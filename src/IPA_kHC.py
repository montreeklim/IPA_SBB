import xpress as xp
import numpy as np
from scipy.linalg import null_space
import itertools
import time
import pickle
import pandas as pd
from dataclasses import dataclass, field
import gc

np.set_printoptions(precision=3, suppress=True)
xp.init('C:/xpressmp/bin/xpauth.xpr') # license path for laptop

@dataclass
class ProblemData:
    dataset: np.ndarray
    n_planes: int
    tol: float = 1e-4
    master_rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))
    all_starts: list = field(init=False)
    M: int = field(init=False)
    N: int = field(init=False)
    K: int = field(init=False)
    a: np.ndarray = field(init=False)
    BigM: float = field(init=False)
    gamma_bound: float = field(init=False)

    def __post_init__(self):
        self.M, self.N = self.dataset.shape
        self.K = self.n_planes
        self.a = self.dataset
        h = np.max(self.a)
        self.BigM = h * np.sqrt(self.N)
        self.gamma_bound = self.N * h + h * np.sqrt(self.N)
        # Pre-generate random starts
        self.all_starts = [generate_random_matrix(self.K, self.N, self.master_rng) for _ in range(100)]

def generate_random_matrix(K, N, rng):
    # Generate a K x N matrix with random values from a normal distribution
    A = rng.standard_normal((K, N))
    gamma = rng.standard_normal(K)
    # Normalize each row of A
    row_norms = np.linalg.norm(A, axis=1, keepdims=True)
    A_normalized = A / row_norms
    return A_normalized, gamma


def starting_points(pdata, starts_list: list) -> list:
    """
    Generate feasible starting MIP-solutions using warm starts and 
    vectorized random refinement.
    """
    start_sols = []
    rows_idx = np.arange(pdata.M)

    for w_init, g_init in starts_list:
        # 1. Initialize geometry
        w, gamma, assign = _find_feasible_geometry(pdata, w_init, g_init)
        
        x = np.zeros((pdata.M, pdata.K), dtype=int)
        x[rows_idx, assign] = 1

        # 2. Alternating Minimization
        for _ in range(100):
            prev_assign = assign.copy()
            
            distances = np.abs(pdata.a @ w.T - gamma)
            assign = np.argmin(distances, axis=1)
            
            if np.array_equal(prev_assign, assign):
                break
                
            x.fill(0)
            x[rows_idx, assign] = 1

            w_list, g_list, _ = compute_w_gamma_y(
                pdata.a, x.ravel(), w, pdata.M, pdata.K, pdata.BigM
            )
            w = np.array(w_list).reshape(pdata.K, pdata.N)
            gamma = np.array(g_list)

        # Enforce Symmetry Breaking
        # Rename clusters so that the first point seen defines the next cluster ID
        cluster_map = {}
        next_id = 0
        
        for i in range(pdata.M):
            c = assign[i]
            if c not in cluster_map:
                cluster_map[c] = next_id
                next_id += 1
                
        # Catch any empty clusters
        for c in range(pdata.K):
            if c not in cluster_map:
                cluster_map[c] = next_id
                next_id += 1
                
        # Apply the new mapping
        new_w = np.zeros_like(w)
        new_g = np.zeros_like(gamma)
        for old_c, new_c in cluster_map.items():
            new_w[new_c] = w[old_c]
            new_g[new_c] = gamma[old_c]
            
        w = new_w
        gamma = new_g
        assign = np.array([cluster_map[c] for c in assign])
        
        x.fill(0)
        x[rows_idx, assign] = 1

        # Compute Exact 'y' Variables
        # y[i] is exactly | w[assign[i]] * a[i] - gamma[assign[i]] |
        # We calculate this using advanced indexing
        dps = np.sum(pdata.a * w[assign], axis=1)
        y = np.abs(dps - gamma[assign])

        # 3. Pack solution
        full_vector = np.concatenate([w.ravel(), gamma, x.ravel(), y])
        start_sols.append(np.round(full_vector, 10))

    return start_sols

def _find_feasible_geometry(pdata, w_start, g_start, batch_size=20, max_tries=100):
    """
    Ensures we have a starting (w, gamma) where every cluster has >= N points.
    """
    A, N, K = pdata.a, pdata.N, pdata.K
    rng = pdata.master_rng
    
    # Check if the provided warm start is already feasible
    assign = np.argmin(np.abs(A @ w_start.T - g_start), axis=1)
    if np.all(np.bincount(assign, minlength=K) >= N):
        return w_start.copy(), g_start.copy(), assign

    # Vectorized search for a random feasible start
    for _ in range(max_tries):
        # Generate Batch
        Ws = rng.standard_normal((batch_size, K, N))
        Ws /= np.linalg.norm(Ws, axis=2, keepdims=True)
        Gs = rng.standard_normal((batch_size, K))

        # Vectorized Distance & Assignment: (Batch, M, K)
        dots = np.matmul(A[None, :, :], Ws.transpose(0, 2, 1))
        dists = np.abs(dots - Gs[:, None, :])
        batch_assigns = np.argmin(dists, axis=2)

        # Vectorized Count Check: Compare each point's assignment to cluster indices
        # Resulting shape: (Batch, K)
        counts = (batch_assigns[:, :, None] == np.arange(K)).sum(axis=1)
        
        valid_idx = np.where(np.all(counts >= N, axis=1))[0]
        if valid_idx.size > 0:
            best_i = valid_idx[0]
            return Ws[best_i], Gs[best_i], batch_assigns[best_i]

    return w_start, g_start, assign


def inverse_power_method(A, x0, tol=1e-4, max_iter=30, reg=1e-10):
    """
    Uses the inverse power method to find the eigenvalue of A closest to zero.
    
    Parameters:
      A       : numpy.ndarray, the input square matrix (assumed invertible)
      x0      : numpy.ndarray, an initial guess vector (nonzero)
      tol     : float, tolerance for convergence
      max_iter: int, maximum number of iterations
      reg     : float, regularization parameter to avoid singularity
      
    Returns:
      eigenvalue : float, the approximated eigenvalue of A
      eigenvector: numpy.ndarray, the corresponding eigenvector (normalized)
      iterations : int, the number of iterations performed
    """
    x = x0 / np.linalg.norm(x0)
    eigenvalue_old = 0.0

    for i in range(max_iter):
        try:
            y = np.linalg.solve(A, x)
        except np.linalg.LinAlgError:
            # If A is singular, solve the modified system: (A + reg * I) y = x
            y = np.linalg.solve(A + reg * np.eye(A.shape[0]), x)
        x = y / np.linalg.norm(y)
        eigenvalue = x.T @ A @ x
        if np.abs(eigenvalue - eigenvalue_old) < tol:
            break
        eigenvalue_old = eigenvalue

    return eigenvalue, -x

import numpy as np

def create_n_simplex(n):
    """Create a numpy array containing extreme points of an n-simplex using a Helmert matrix."""
    # Initialize an (n+1) x n matrix to hold the vertices as rows
    V_T = np.zeros((n + 1, n))
    
    # Build the sub-matrix of the Helmert transformation (transposed)
    for k in range(1, n + 1):
        val = 1.0 / np.sqrt(k * (k + 1))
        V_T[0:k, k - 1] = val
        V_T[k, k - 1] = -k * val
        
    # Normalize each row (vertex) to lie on the unit sphere
    V_T = V_T / np.linalg.norm(V_T, axis=1, keepdims=True)
    
    return V_T


def split_data(data, K, N):
    """Splits data into K parts of length N."""
    return np.array([data[i*N : (i+1)*N] for i in range(K)])

def up_extension_constraint(
    vertices_matrix, 
    cond_thresh: float = 1e7, 
    null_rcond: float = 1e-4, 
    dup_atol: float = 1e-6
) -> np.ndarray:
    """
    Create a numpy array containing coefficients for adding constraints 
    in new nodes based on a matrix of extreme points.
    """
    E = np.asarray(vertices_matrix)
    n_rows, n_cols = E.shape
    e = np.ones(n_rows)
    
    # 1. Compute the first constraint coefficient 'a'
    # Only attempt exact solve if square and well-conditioned
    if n_rows == n_cols and np.linalg.cond(E) < cond_thresh:
        a = np.linalg.solve(E, e)
    else:
        a = np.linalg.pinv(E) @ e
        
    # Store rows in a standard Python list (much faster than vstack in a loop)
    coeffs_list = [a]
    
    # 2. Compute null space bases for subsets
    # (Original logic skipped the last row, iterating to n_rows - 1)
    for i in range(n_rows - 1):
        E_sub = np.delete(E, i, axis=0)
        null_basis = null_space(E_sub, rcond=null_rcond)
        
        # null_space returns basis as columns, we want them as rows
        if null_basis.size > 0:
            coeffs_list.append(null_basis.T)
            
    # Combine everything into a single array at once
    a_coeff = np.vstack(coeffs_list)
    
    # 3. Filter out near-duplicates (and sign-flipped duplicates)
    unique_rows = []
    for row in a_coeff:
        # Check against already accepted unique rows
        is_dup = any(
            np.allclose(row, u, atol=dup_atol) or np.allclose(row, -u, atol=dup_atol)
            for u in unique_rows
        )
        if not is_dup:
            unique_rows.append(row)
            
    return np.array(unique_rows)


def ProjectOnBall(w):
    """Project the obtained solution onto the ball through the origin."""
    norm_w = np.linalg.norm(w)
    if norm_w == 0 or np.isnan(norm_w):
        # Handle zero or NaN case gracefully
        return w  # or return some default value
    else:
        # Perform the normalization
        return w / norm_w
    
def safe_tolist(arr):
    return arr.tolist() if hasattr(arr, "tolist") else arr

def compute_w_gamma_y(a, x, w_old, rows, cols, BigM):
    """
    Recalculate the optimal solution for hyperplane clustering given an integer solution x.
    """
    # Ensure a and x are NumPy arrays and reshape x into a 2D (rows x cols) binary matrix.
    a = np.asarray(a)
    x = np.asarray(x).reshape(rows, cols)

    # For each cluster j, extract the subarray of points where x[i,j]==1.
    subarrays_per_j = {j: a[x[:, j].astype(bool)] for j in range(cols)}

    # Initialize lists to store hyperplane parameters.
    w_list = []
    gamma_list = []

    # Process each cluster j.
    for j in range(cols):
        subarray = subarrays_per_j[j]
        n_points = subarray.shape[0]
        if n_points == 0:
            w_list.append(w_old[j])
            gamma_list.append(0)
            continue

        # Compute the projection matrix
        P = np.eye(n_points) - np.ones((n_points, n_points)) / n_points

        # Compute B = subarray^T * P * subarray.
        B_j = subarray.T @ P @ subarray

        # Use the inverse power method to compute the smallest eigenpair.
        eigenvalue, w_j = inverse_power_method(B_j, w_old[j], tol=1e-4, max_iter=30)

        # Compute gamma for cluster j
        gamma_j = np.sum(subarray @ w_j) / n_points

        w_list.append(w_j)
        gamma_list.append(gamma_j)

    # Compute y for each point.
    y = np.zeros(rows)
    for j, (w_j, gamma_j) in enumerate(zip(w_list, gamma_list)):
        indices = (x[:, j] == 1)
        if np.any(indices):
            dp = a[indices] @ w_j
            term1 = dp - gamma_j
            term2 = -dp + gamma_j
            y[indices] = np.maximum(0, np.maximum(term1, term2))

    w_concat = np.concatenate(w_list)

    return w_concat.tolist(), gamma_list, y.tolist()

    
def create_problem(pdata: ProblemData) -> xp.problem:
    """
    Build and return an Xpress model for the hyperplane‐clustering MIP,
    pulling all data (M, N, K, a, BigM, gamma_bound) from `pd`.
    """
    prob = xp.problem()

    # unpack once from pd
    M, N, K    = pdata.M, pdata.N, pdata.K
    a, BigM    = pdata.a, pdata.BigM
    gamma_bound = pdata.gamma_bound

    # --- variables ---
    w     = prob.addVariables(K, N, lb=-1, ub=1, name="w")
    gamma = prob.addVariables(K, lb=-gamma_bound, ub=gamma_bound, name="gamma")
    x     = prob.addVariables(M, K, vartype=xp.binary, name="x")
    y     = prob.addVariables(M, lb=0, name="y")

    # --- symmetry breaking on x ---
    for i in range(M):
        for j in range(K):
            if j > i:
                x[i, j].ub = 0

    # --- assignment + big-M constraints ---
    for i in range(M):
        # each point must go in exactly one of clusters 0..min(i,K-1)
        prob.addConstraint(
            xp.Sum(x[i, j] for j in range(min(i+1, K))) == 1
        )
        for j in range(K):
            if j <= i:
                # y[i] ≥ | w[j]·a[i] – γ[j] |  via two half-spaces
                prob.addConstraint(
                    y[i] >= xp.Dot(w[j], a[i]) - gamma[j] - BigM*(1 - x[i, j])
                )
                prob.addConstraint(
                    y[i] >= xp.Dot(-w[j], a[i]) + gamma[j] - BigM*(1 - x[i, j])
                )

    # --- norm constraints on each hyperplane ---
    for j in range(K):
        prob.addConstraint(
            xp.Sum(w[j, t]*w[j, t] for t in range(N)) <= 1
        )

    # --- objective: minimize sum of squared residuals ---
    prob.setObjective(
        xp.Sum(y[i]*y[i] for i in range(M)),
        sense=xp.minimize
    )

    return prob


def cbchecksol(prob, data, soltype, cutoff):
    """
    Reject any node-solution whose hyperplanes aren't all on the unit-ball.
    If they are off-ball but nonzero, compute a heuristic solution via Mangasarian
    and stash it in data['refuse_sol'] for prenode_callback to add later.
    """
    pdata = data["pd"]
    M, N, K = pdata.M, pdata.N, pdata.K
    BigM, tol = pdata.BigM, pdata.tol
    
    # 1. Early exit if presolve hasn't produced an LP solution
    if (prob.attributes.presolvestate & 128) == 0:
        return (1, 0)

    # 2. Fetch the continuous solution safely
    try:
        # Cast immediately to a NumPy array for fast slicing later
        sol = np.array(prob.getCallbackSolution(prob.getVariable()))
    except Exception: # Avoid bare 'except:' to prevent catching KeyboardInterrupt
        return (1, cutoff)

    # 3. Extract and reshape w (assuming w_idxs are contiguous)
    w_idxs = data["w_idxs"]
    w_arr = sol[w_idxs[0] : w_idxs[-1] + 1].reshape(K, N)

    # 4. Vectorized check for unit-norm hyperplanes
    norms = np.linalg.norm(w_arr, axis=1)
    if np.all(np.abs(norms - 1.0) < tol):
        return (0, cutoff)

    # 5. Add closed-form heuristic solution
    # Check if every row in w_arr has at least one element > 1e-4
    if np.all(np.any(np.abs(w_arr) > 1e-4, axis=1)):
        x_idxs = data["x_idxs"]
        x_flat = sol[x_idxs[0] : x_idxs[-1] + 1]
        
        # Compute new heuristic parameters
        new_w, new_gamma, new_y = compute_w_gamma_y(pdata.a, x_flat, w_arr, M, K, BigM)
        
        # Fast vectorized check against incumbent
        new_y_arr = np.asarray(new_y)
        if np.sum(new_y_arr**2) < prob.attributes.mipbestobjval:
            
            # Use np.concatenate instead of list addition (+). 
            # This is significantly faster and more memory-efficient.
            new_sol = np.concatenate([
                np.ravel(new_w), 
                np.ravel(new_gamma), 
                np.ravel(x_flat), 
                new_y_arr
            ])
            
            # Round and append in one step
            data["refuse_sol"].append(np.round(new_sol, 10).tolist())

    # 6. Reject the current node solution
    return (1, cutoff)


def prenode_callback(prob, data):
    """
    Before diving into a new node, inject any heuristic MIP-solutions
    stored in data['refuse_sol'] (from cbchecksol), then clear the list.
    """
    # Use .get() safely in case the key is missing, defaulting to an empty list
    refuse = data.get("refuse_sol", [])
    
    for sol in refuse:
        prob.addmipsol(sol)
        
    # Clear the list in-place rather than creating a new list object
    refuse.clear()
    
    return 0

def cbbranch(prob, data, branch):
    """
    Branching callback: at node 1 build the full (N+1)^K branchobj;
    at other nodes decide whether to branch on x or w, and if w build
    a small branchobj on the chosen “ball face.”
    """
    pdata = data["pd"]
    N, K = pdata.N, pdata.K
    tol = pdata.tol
    node = prob.attributes.currentnode
    rng_node = np.random.default_rng(42 + node)

    # =========================================================================
    # 1. ROOT NODE: Build the large (N+1)^K branching object
    # =========================================================================
    if node == 1:
        bo = xp.branchobj(prob, isoriginal=True)
        bo.addbranches((N + 1) ** K)

        # Build and stash the initial simplex
        init_simplex = create_n_simplex(N)
        data["initial_polytope"] = init_simplex

        # Precompute submatrices and coefficients for each face i=0..N
        data["submatrix"]      = {}
        data["a_coeff"]        = {}
        data["extreme_points"] = {}

        for i in range(N + 1):
            face = np.delete(init_simplex, i, axis=0)
            data["submatrix"][i]      = face
            data["extreme_points"][i] = face

            coeffs = up_extension_constraint(face)
            # Flip sign so max(face @ c) >= 0
            for j, c in enumerate(coeffs):
                if np.max(face @ c) < 1e-6:
                    coeffs[j] = -c
            data["a_coeff"][i] = coeffs

        # Add rows for every K-tuple of faces
        powers = [(N + 1) ** (K - k - 1) for k in range(K)]
        
        for combo in itertools.product(range(N + 1), repeat=K):
            idx = sum(combo[k] * powers[k] for k in range(K))
            
            # Each hyperplane k
            for k, face_id in enumerate(combo):
                # Assuming w_vars corresponds directly to these indices natively
                w_vars = np.arange(k * N, (k + 1) * N)
                
                for j, coeff in enumerate(data["a_coeff"][face_id]):
                    rhs = 1 if j == 0 else 0
                    bo.addrows(
                        idx, ['G'], [rhs], [0, N * K], w_vars, coeff
                    )
                    
            # Stash the combined extreme_points for later use
            data["extreme_points"][idx] = data["submatrix"][combo[0]]

        bo.setpriority(100)
        return bo

    # =========================================================================
    # 2. NON-ROOT NODES: Decide whether and how to branch
    # =========================================================================
    
    # Only proceed if LP presolve has run
    if (prob.attributes.presolvestate & 128) == 0:
        return branch

    # Fetch the current LP solution safely
    try:
        sol = prob.getCallbackSolution(prob.getVariable())
    except Exception:  # Catch Exception, not everything, to allow KeyboardInterrupt
        return branch

    # Reshape the w-part natively (assumes w_idxs is sorted and contiguous)
    w_idxs = data["w_idxs"]
    flat_w = np.array(sol[w_idxs[0] : w_idxs[-1] + 1])
    w_arr = flat_w.reshape(K, N)

    # Fast vectorized calculation of norms
    norms = np.linalg.norm(w_arr, axis=1)
    
    # If all hyperplanes are on the ball, skip custom branching
    if np.all(np.abs(norms - 1.0) < tol):
        return branch

    # Pick the "smallest-norm" ball
    ball_id = int(np.argmin(norms))

    # Initialize node data dictionary
    nd = data.setdefault("node_data", {})
    nd[node] = {"w_array": w_arr, "ball_id": ball_id}

    # Optional: skip if distances already tiny
    dist = nd[node].get("distance", [])
    if dist and max(dist) <= 1e-6:
        return branch

    # Bound/gap test (with protection against division by zero)
    dual = prob.getAttrib("bestbound")
    if dual <= tol:
        nd[node]["branch_on_w"] = False
        return branch

    mipobj = prob.getAttrib("mipobjval")
    
    # Safeguard against mipobj being zero or infinity
    if abs(mipobj) < 1e-9 or np.isinf(mipobj):
        gap = 1.0 
    else:
        gap = abs((mipobj - dual) / mipobj)

    # Randomly choose x-branch vs w-branch
    branch_on_w = rng_node.random() >= max(gap, 1 - gap)
    nd[node]["branch_on_w"] = branch_on_w
    if not branch_on_w:
        return branch

    # =========================================================================
    # 3. Build a small branchobj on the chosen ball face
    # =========================================================================
    face = data["extreme_points"][ball_id]
    proj_w = ProjectOnBall(w_arr[ball_id])
    face2 = np.vstack((face, proj_w))
    
    # The new facet 'face2' is defined by N+1 points in N-dimensional space.
    # It is degenerate if these points do not span an N-dimensional affine space.
    if np.linalg.matrix_rank(face2, tol=1e-4) < N:
        # Prune the node due to degeneracy
        bo = xp.branchobj(prob, isoriginal=True)
        bo.addbranches(0)
        return bo
        
    # The facet is valid, generate constraints
    try:
        coeffs2 = up_extension_constraint(face2)
    except Exception:
        return branch # Fallback if constraint generation fails

    bo = xp.branchobj(prob, isoriginal=True)
    bo.addbranches(N)
    w_vars = np.arange(ball_id * N, (ball_id + 1) * N)

    for i in range(N):
        # Drop row i
        for j, cf in enumerate(coeffs2):
            rhs = 1 if j == 0 else 0
            
            # Recalculate max to check if sign needs flipping
            test_matrix = np.vstack((np.delete(face2, i, 0), proj_w))
            if j > 0 and np.max(test_matrix @ cf) < 1e-6:
                cf = -cf
                
            bo.addrows(i, ['G'], [rhs], [0, N * K], w_vars, cf)
            
    return bo

def cbnewnode(prob, data, parentnode, newnode, branch):
    """
    When a new node is created:
     - if its parent was the root (parentnode==1), 
       pick up the precomputed extreme-points face 'branch'
     - otherwise, if the parent branched on x, we inherit unchanged
     - if the parent branched on w, we remove row 'branch' from the face,
       append the projected w, and record distances.
    """
    # 1. Early exit: only act once the LP presolve has given us a solution
    if (prob.attributes.presolvestate & 128) == 0:
        return 0

    pdata = data["pd"]
    N = pdata.N
    node_data = data.setdefault("node_data", {})

    # 2. Case 1: Child of the root
    # Directly assign the dict rather than allocating an empty {} first
    if parentnode == 1:
        node_data[newnode] = {
            "face_matrix": data["extreme_points"][branch],
            "ball_id": branch
        }
        return 0

    # Fetch parent data safely (returns empty dict if missing to prevent KeyError)
    parent = node_data.get(parentnode, {})
    if not parent:
        node_data[newnode] = {}
        return 0

    # 3. If parent branched on x, inherit state exactly
    if parent.get("branch_on_w") is False:
        node_data[newnode] = parent.copy()
        return 0

    # 4. Parent branched on w: "remove one row + project" update
    w_arr = parent["w_array"]
    ball_id = parent["ball_id"]

    # Original face for this ball (from root node precomputation)
    orig_face = data["extreme_points"][ball_id]

    # Remove the 'branch'-th row and project the chosen hyperplane onto the ball
    subface = np.delete(orig_face, branch, axis=0)
    pi_w = ProjectOnBall(w_arr[ball_id])

    # Overwrite with the new face
    updated_face = np.vstack((subface, pi_w))

    # 5. Vectorized distance calculation
    # Computes norm of difference for the first N rows in C-space
    distances = np.linalg.norm(updated_face[:N] - orig_face[:N], axis=1).tolist()

    # 6. Copy parent's bookkeeping and update specific fields
    new_state = parent.copy()
    new_state["face_matrix"] = updated_face
    new_state["ball_id"] = ball_id
    new_state["distance"] = distances
    
    node_data[newnode] = new_state

    return 0

    
def solve(pdata: ProblemData) -> xp.problem:
    """
    Build, configure, and solve the Xpress model for the given ProblemData.
    All sizes and parameters come from pdata—no globals.
    """
    # Unpack once
    N, K    = pdata.N, pdata.K
    # 1) build the model
    prob = create_problem(pdata)

    # 2) grab variable‐index lists
    all_vars     = prob.getVariable()
    w_idxs        = [i for i,v in enumerate(all_vars) if v.name.startswith("w")]
    gamma_idxs    = [i for i,v in enumerate(all_vars) if v.name.startswith("gamma")]
    x_idxs        = [i for i,v in enumerate(all_vars) if v.name.startswith("x")]
    y_idxs        = [i for i,v in enumerate(all_vars) if v.name.startswith("y")]

    # 3) precompute starting solutions
    starts = starting_points(pdata, pdata.all_starts)

    # 4) prepare the shared data dict for callbacks
    data = {
        "pd": pdata,
        "w_idxs": w_idxs,
        "gamma_idxs": gamma_idxs,
        "x_idxs": x_idxs,
        "y_idxs": y_idxs,
        "refuse_sol": starts,
        "extreme_points": {},
        "submatrix": {},
        "a_coeff": {},
        "node_data": {}
    }

    # 5) register callbacks
    prob.addcbpreintsol(cbchecksol, data, 2)
    prob.addcbprenode(prenode_callback, data, 1)
    prob.addcbchgbranchobject(cbbranch, data, 2)
    prob.addcbnewnode(cbnewnode, data, 2)

    # 6) solver controls (all using local N,K)
    
    # best first
    prob.controls.backtrack       = 5
    prob.controls.nodeselection  = 4
    prob.controls.backtracktie   = 5
    prob.controls.breadthfirst   = (N + 1) ** K + 1
    
    # best bound
    # prob.controls.backtrack       = 3
    # prob.controls.nodeselection  = 2
    # prob.controls.backtracktie   = 1
    
    # depth first search
    # prob.controls.nodeselection  = 5
    
    
    prob.controls.timelimit      = 3600
    prob.controls.randomseed     = 42
    prob.controls.deterministic  = 1
    # prob.controls.threads        = 1
    # prob.controls.maxnode = 10000
    
    # to avoid numerical errors with LP
    # prob.controls.miprelstop = 1e-4 
    # prob.controls.scaling = 1
    # prob.controls.feastol = 1e-7
    
    start_time = time.time()
    
    # 7) run
    prob.mipoptimize()
    computation_time = time.time() - start_time

    return prob, computation_time

if __name__ == "__main__":

    TOL      = 1e-4
    N_STARTS = 100
    RESUME_IDX = 0

    # TARGET = (22, 2, 2)  # (m, n, k) to benchmark

    datasets_filenames = [
        "LowDim_low_noise.pkl",
        "LowDim_medium_noise.pkl",
        "LowDim_high_noise.pkl",
    ]

    for filename in datasets_filenames:
        print(f"Processing {filename} …")
        with open(filename, "rb") as f:
            datasets_dict = pickle.load(f)
        print(f"  Loaded {len(datasets_dict)} instances\n")

        results = []
        for idx, ((m, n, k), data_array) in enumerate(datasets_dict.items()):
            # if (m, n, k) != TARGET:
            #     continue

            problem_data = ProblemData(
                dataset  = data_array,
                n_planes = k,
                tol      = TOL
            )
            problem_data.all_starts = [
                generate_random_matrix(k, n, problem_data.master_rng)
                for _ in range(N_STARTS)
            ]

            prob, computation_time = solve(problem_data)
            duration = round(computation_time, 3)

            nodes       = prob.attributes.nodes
            lower_bound = prob.attributes.bestbound
            upper_bound = prob.attributes.objval

            print(f"  Instance {idx} (m={m},n={n},k={k}):")
            print(f"    Nodes:       {nodes}")
            print(f"    Time:        {duration}s")
            print(f"    Lower Bound: {lower_bound:.6f}")
            print(f"    Upper Bound: {upper_bound:.6f}")

            del prob
            gc.collect()

            results.append({
                "m": m, "n": n, "k": k,
                "Nodes": nodes,
                "Time": duration,
                "LowerBound": lower_bound,
                "UpperBound": upper_bound,
            })

        if results:
            df = pd.DataFrame(results)
            outname = f"results_{filename[:-4]}_IPA.xlsx"
            df.to_excel(outname, index=False)
            print(f"  → Saved final batch to {outname}\n")


# if __name__ == "__main__":

#     # You can adjust tol, number of random starts, batch size, etc.
#     TOL        = 1e-4
#     N_STARTS   = 100
#     # BATCH_SIZE = 4
#     RESUME_IDX = 0

#     datasets_filenames = [        
#         # 'Vision_Instances_hard.pkl',
#         # 'Vision_Instances.pkl',
#         'test_instances.pkl'
#         # "LowDim_no_noise.pkl",
#         # "LowDim_low_noise.pkl",
#         # "LowDim_medium_noise.pkl",
#         # "LowDim_high_noise.pkl",
#         # "HighDim_low_noise.pkl",
#         # "HighDim_medium_noise.pkl",
#         # "HighDim_high_noise.pkl"
#         # "Hyperplane_Instances_low_noise.pkl",
#         # "Hyperplane_Instances_medium_noise.pkl",
#         # "Hyperplane_Instances_high_noise.pkl",
#     ]

#     for filename in datasets_filenames:
#         print(f"Processing {filename} …")
#         with open(filename, "rb") as f:
#             datasets_dict = pickle.load(f)
#         print(f"  Loaded {len(datasets_dict)} instances\n")

#         results = []
#         for idx, ((m, n, k), data_array) in enumerate(datasets_dict.items()):
#             # if idx >= 3:
#             #     continue
            
#             # build your ProblemData
#             problem_data = ProblemData(
#                 dataset  = data_array,
#                 n_planes = k,
#                 tol      = TOL
#                 )
#             # override the random starts if you like
#             problem_data.all_starts = [
#                 generate_random_matrix(k, n, problem_data.master_rng)
#                 for _ in range(N_STARTS)
#                 ]

#             # solve it
#             prob, computation_time = solve(problem_data)
#             duration = round(computation_time, 3)
            
#             # 3) collect metrics
#             nodes          = prob.attributes.nodes
#             lower_bound    = prob.attributes.bestbound
#             upper_bound    = prob.attributes.mipbestobjval
            
#             # optimality_gap = abs((upper_bound - lower_bound) / upper_bound)

#             print(f"  Instance {idx} (m={m},n={n},k={k}):")
#             print(f"    Nodes:          {nodes}")
#             print(f"    Time:           {duration}s")
#             print(f"    Lower Bound:    {lower_bound:.6f}")
#             print(f"    Upper Bound:    {upper_bound:.6f}")
#             # print(f"    Optimality Gap: {optimality_gap*100:.4f}%\n")

#             del prob
#             gc.collect()

#             results.append({
#                 "m": m,
#                 "n": n,
#                 "k": k,
#                 "Nodes": nodes,
#                 "Time": duration,
#                 "LowerBound": lower_bound,
#                 "UpperBound": upper_bound,
#                 # "Gap": optimality_gap,
#             })
            
#         # any leftover
#         if results:
#             df = pd.DataFrame(results)
#             outname = f"results_{filename[:-4]}_refactor.xlsx"
#             # outname = f"results_{filename[:-4]}_best_bound.xlsx"
#             df.to_excel(outname, index=False)
#             print(f"  → Saved final batch to {outname}\n")