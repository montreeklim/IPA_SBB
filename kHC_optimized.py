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

def starting_points(pdata: ProblemData, starts_list):
    """
    Generate feasible starting MIP‐solutions (w, gamma, x, y) for hyperplane clustering,
    using only data from `pd` and the provided starts_list.  Uses a vectorized batch
    approach to find a valid random start (each cluster ≥ N points).
    """
    M, N, K     = pdata.M, pdata.N, pdata.K
    A           = pdata.a              # (M × N)
    tol         = pdata.tol
    BigM        = pdata.BigM
    rng         = pdata.master_rng

    rows_idx    = np.arange(M)
    start_sols  = []

    for w0, gamma0 in starts_list:
        # 1) warm‐start hyperplane params
        w, gamma = w0.copy(), gamma0.copy()

        # 2a) initial assignment
        dot        = A @ w.T
        dist       = np.abs(dot - gamma)      # (M × K)
        assign     = np.argmin(dist, axis=1)  # length‐M

        # 2b) vectorized “find-feasible-start” loop
        B = 20  # batch size: tune as needed
        while True:
            # draw B random (W,Γ) pairs
            Ws     = rng.standard_normal((B, K, N))
            Gammas = rng.standard_normal((B, K))
            norms  = np.linalg.norm(Ws, axis=2, keepdims=True)  # (B,K,1)
            Ws     = Ws / norms

            # compute dists: (B, M, K)
            dots   = np.matmul(A[None,:,:], Ws.transpose(0,2,1))
            dists  = np.abs(dots - Gammas[:,None,:])

            # assignments: (B, M)
            assigns_batch = np.argmin(dists, axis=2)

            # counts[b,k] = #points in cluster k for batch b → (B, K)
            counts = np.stack(
                [(assigns_batch == k).sum(axis=1) for k in range(K)],
                axis=1
            )

            valid_mask = np.all(counts >= N, axis=1)  # (B,)
            if valid_mask.any():
                idx   = np.argmax(valid_mask)
                w     = Ws[idx]
                gamma = Gammas[idx]
                assign = assigns_batch[idx]
                break
            # else repeat

        # 2c) alternating minimization (unchanged)
        prev_assign = assign.copy()
        for _ in range(50):
            w_old = w.copy()
            dot   = A @ w.T
            dist  = np.abs(dot - gamma)
            assign = np.argmin(dist, axis=1)

            x = np.zeros((M, K), dtype=int)
            x[rows_idx, assign] = 1

            w_list, gamma_list, y = compute_w_gamma_y(A, x.ravel(), w_old, M, K, BigM)
            w     = np.array(w_list).reshape(K, N)
            gamma = np.array(gamma_list)

            if np.linalg.norm(prev_assign - assign) < tol:
                break
            prev_assign = assign.copy()

        # 2d) pack into one flat solution vector
        new_sol = np.concatenate([w.ravel(), gamma, x.ravel(), y])
        start_sols.append(np.round(new_sol, 10))

    return start_sols


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

def create_n_simplex(n):
    """Create a numpy array contains extreme points of n-simplex."""
    E = np.identity(n+1)
    c = np.ones(n+1)/(n+1)
    A = E.copy()
    for i in range(n):
        A[i] = E[i]-c
    A = A[0:n, :].T
    
    U, _ = np.linalg.qr(A)
    e = U.copy()
    for i in range(n+1):
        e[i] = np.linalg.pinv(U)@E[i]
        # normalize
        e[i] = e[i]/np.linalg.norm(e[i])
    return e

def append_zeros_and_ones(E):
    """Create a numpy array for augmented matrix E to check if 0 in affine(E)"""
    n = E.shape[0]
    
    # Create a column vector of zeros of size n
    zeros_column = np.zeros((n, 1))
    
    # Create a row vector of ones of size n, and append a zero
    ones_row = np.concatenate((np.ones(n), [0]))
    
    # Stack E with the zeros_column vertically, and concatenate ones_row horizontally
    new_array = np.vstack((np.hstack((E, zeros_column)), ones_row))
    
    return new_array

def split_data(data, K, N):
    """Splits data into K parts of length N."""
    return np.array([data[i*N : (i+1)*N] for i in range(K)])

def up_extension_constraint(vertices_matrix):
    """Create a numpy array contains coefficient for adding constraints in new nodes based on matrix of extreme points."""
    # Convert the vertices_matrix to a numpy array
    E = np.array(vertices_matrix)
    # Check condition number of new_matrix
    cond_number = np.linalg.cond(E)
    e = np.ones(len(E))
    if cond_number >= 1e+7: # Learning to Scale MIP, Timo and Gregor, treat cond_number < 1e+7 as normal for MIP
        a = np.linalg.pinv(E) @ e
    else:
        a = np.linalg.solve(E, e)
    
    # Compute the coefficient 'a' for the first constraint
    a_coeff = np.empty((0, len(E[0])))  # Initialize an empty array to store coefficients
    a_coeff = np.vstack((a_coeff, a))
    
    # Compute coefficients for additional constraints
    n = E.shape[0] - 1  # Number of extreme points

    for i in range(n):
        # Compute the null space basis of the subarray by removing the i-th row
        null_basis = null_space(np.delete(E, i, axis=0), rcond=1e-4)
        
        # Append each null space vector as a constraint coefficient
        a_coeff = np.vstack((a_coeff, null_basis.T))
    
    # Check redundant
    # Identify and remove near-redundant rows
    unique = []

    for row in a_coeff:
        # inline “near‐duplicate or sign‐flipped near‐duplicate” test
        is_dup = any(
            np.allclose(row, u, atol=1e-6)    # same
            or np.allclose(row, -u, atol=1e-6) # flipped
            for u in unique
            )
        if not is_dup:
            unique.append(row)

    unique_rows = np.vstack(unique)

    # Convert the list of unique rows back to a NumPy array
    A_reduced = np.array(unique_rows)
    
    return A_reduced

def check_all_balls(pdata: ProblemData, w_array: np.ndarray) -> bool:
    """
    Verify that every hyperplane weight vector in w_array lies (within tol)
    on the unit ball.

    Parameters
    ----------
    pd      : ProblemData
        Contains pd.tol, the acceptance tolerance.
    w_array : np.ndarray, shape (K, N)
        The stack of weight vectors to check.

    Returns
    -------
    bool
        True if ∥w_i∥ ∈ [1–tol, 1+tol] for all i; False otherwise.
    """
    tol = pdata.tol
    # vectorized norm check
    norms = np.linalg.norm(w_array, axis=1)
    return np.all(np.abs(norms - 1) < tol)


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

    Parameters
    ----------
    a : array_like
        Points to be classified, assumed to be a NumPy array of shape (rows, dim).
    x : array_like
        A flat list/array representing a binary assignment matrix (rows x cols) where 
        x[i*cols + j] == 1 if point a[i] is in cluster j.
    w_old : list or array
        A list/array of warm-start hyperplane parameters for each cluster (each is a vector of length N).
    rows : int
        Number of rows (points).
    cols : int
        Number of clusters.
    BigM : float
        A sufficiently large constant (not used in this implementation).

    Returns
    -------
    w : list
        Concatenated optimal hyperplane parameters for each cluster.
    gamma : list
        Optimal gamma values (one per cluster).
    y : list
        The computed y values (one per point).
    """
    import numpy as np

    # Ensure a and x are NumPy arrays and reshape x into a 2D (rows x cols) binary matrix.
    a = np.asarray(a)
    x = np.asarray(x).reshape(rows, cols)

    # For each cluster j, extract the subarray of points where x[i,j]==1.
    # This uses boolean indexing and avoids manual index conversion.
    subarrays_per_j = {j: a[x[:, j].astype(bool)] for j in range(cols)}

    # Initialize lists to store hyperplane parameters.
    w_list = []
    gamma_list = []

    # Process each cluster j.
    for j in range(cols):
        subarray = subarrays_per_j[j]
        n = subarray.shape[0]
        if n == 0:
            # If no points are assigned, use the warm start and gamma = 0.
            w_list.append(w_old[j])
            gamma_list.append(0)
            continue

        # Compute the projection matrix:
        #   P = I - (1/n) * ones(n, n)
        # This is equivalent to I - (e e^T)/n.
        P = np.eye(n) - np.ones((n, n)) / n

        # Compute B = subarray^T * P * subarray.
        B_j = subarray.T @ P @ subarray

        # Use the inverse power method (with warm start) to compute the smallest eigenpair.
        eigenvalue, w_j = inverse_power_method(B_j, w_old[j], tol=1e-4, max_iter=30)

        # Compute gamma for cluster j:
        # Since e.T @ subarray is equivalent to summing the rows of subarray, we can write:
        gamma_j = np.sum(subarray @ w_j) / n

        w_list.append(w_j)
        gamma_list.append(gamma_j)

    # Compute y for each point.
    # Assuming each point is assigned to exactly one cluster (i.e. each row of x has a single 1),
    # we vectorize over clusters.
    y = np.zeros(rows)
    for j, (w_j, gamma_j) in enumerate(zip(w_list, gamma_list)):
        # Get indices where point i is assigned to cluster j.
        indices = (x[:, j] == 1)
        if np.any(indices):
            # Compute dot products for all assigned points at once.
            dp = a[indices] @ w_j
            term1 = dp - gamma_j
            term2 = -dp + gamma_j
            # For each point, y is the maximum of (0, term1, term2).
            y[indices] = np.maximum(0, np.maximum(term1, term2))

    # Concatenate the per-cluster hyperplane parameters into one flat vector.
    # (This assumes each w_j is of length N, where N is defined globally.)
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
    Reject any node‐solution whose hyperplanes aren’t all on the unit‐ball.
    If they are off‐ball but nonzero, compute a heuristic solution via Mangasarian
    and stash it in data['refuse_sol'] for prenode_callback to add later.
    """
    pdata       = data["pd"]
    M, N, K  = pdata.M, pdata.N, pdata.K
    a, BigM  = pdata.a, pdata.BigM
    tol      = pdata.tol
    w_idxs   = data["w_idxs"]
    x_idxs   = data["x_idxs"]
    refuse   = data["refuse_sol"]

    # only act once presolve has produced an LP solution
    if (prob.attributes.presolvestate & 128) == 0:
        return (1, 0)

    # fetch the continuous solution
    try:
        sol = prob.getCallbackSolution(prob.getVariable())
    except:
        return (1, cutoff)

    # Extract and reshape the w-part
    w_flat = sol[min(w_idxs) : max(w_idxs) + 1]
    w_arr  = split_data(w_flat, K, N)

    # 1) if all hyperplanes are already unit-norm, accept or skip
    on_ball = all(abs(np.linalg.norm(w_arr[i]) - 1) < tol for i in range(K))
    if on_ball:
        # If the LP obj is 0, we still reject so branching continues
        if prob.attributes.lpobjval == 0:
            return (1, cutoff)
        # Otherwise accept this solution
        return (0, cutoff)

    # 2) if every w-vector is nonzero but some are off-ball, record a heuristic
    nonzero = np.any(np.abs(w_arr) > 1e-4, axis=1)
    if np.all(nonzero):
        # extract x, recompute (w,gamma,y) via the MIP-free routine
        x_flat = sol[min(x_idxs) : max(x_idxs) + 1]
        new_w, new_gamma, new_y = compute_w_gamma_y(a, x_flat, w_arr, M, K, BigM)
        # flatten and round
        new_w, new_gamma, x_flat, new_y = [
            safe_tolist(arr) for arr in (new_w, new_gamma, x_flat, new_y)
        ]
        new_sol = np.round(new_w + new_gamma + x_flat + new_y, 10).tolist()
        refuse.append(new_sol)

    # In all other cases we reject
    return (1, cutoff)


def prenode_callback(prob, data):
    """
    Before diving into a new node, inject any heuristic MIP‐solutions
    we stored in data['refuse_sol'] (from cbchecksol), then clear the list.
    """
    refuse = data["refuse_sol"]
    if refuse:
        for sol in refuse:
            prob.addmipsol(sol)
        # clear for the next node
        data["refuse_sol"] = []
    return 0

def cbbranch(prob, data, branch):
    """
    Branching callback: at node 1 build the full (N+1)^K branchobj;
    at other nodes decide whether to branch on x or w, and if w build
    a small branchobj on the chosen “ball face.”
    """
    pdata       = data["pd"]
    N, K     = pdata.N, pdata.K
    tol      = pdata.tol
    node     = prob.attributes.currentnode
    rng_node = np.random.default_rng(42 + node)

    # --- ROOT NODE: build the big (N+1)^K branching object ---
    if node == 1:
        bo = xp.branchobj(prob, isoriginal=True)
        bo.addbranches((N + 1) ** K)

        # build and stash the initial simplex
        init_simplex = create_n_simplex(N)
        data["initial_polytope"] = init_simplex

        # for each face i=0..N compute its submatrix and coeffs
        data["submatrix"]      = {}
        data["a_coeff"]        = {}
        data["extreme_points"] = {}

        for i in range(N + 1):
            face = np.delete(init_simplex, i, axis=0)
            data["submatrix"][i]      = face
            data["extreme_points"][i] = face

            coeffs = up_extension_constraint(face)
            # flip sign so max(face @ c) >= 0
            for j, c in enumerate(coeffs):
                if np.max(face @ c) < 1e-6:
                    coeffs[j] = -c
            data["a_coeff"][i] = coeffs

        # now for every K‐tuple of faces add rows
        powers = [(N + 1) ** (K - k - 1) for k in range(K)]
        for combo in itertools.product(range(N+1), repeat=K):
            idx = sum(combo[k] * powers[k] for k in range(K))
            # each hyperplane k
            for k, face_id in enumerate(combo):
                w_vars = np.arange(k * N, (k + 1) * N)
                for j, coeff in enumerate(data["a_coeff"][face_id]):
                    rhs = 1 if j == 0 else 0
                    bo.addrows(
                        idx,
                        ['G'],
                        [rhs],
                        [0, N * K],
                        w_vars,
                        coeff
                    )
            # also stash the combined extreme_points if you need them later
            data["extreme_points"][idx] = data["submatrix"][combo[0]]

        bo.setpriority(100)
        return bo

    # --- NON-ROOT NODES: decide whether and how to branch ---
    # only if LP presolve has run
    if (prob.attributes.presolvestate & 128) == 0:
        return branch

    # fetch the current LP solution
    try:
        sol = prob.getCallbackSolution(prob.getVariable())
    except:
        return branch

    # reshape the w‐part from the stored indices
    w_idxs = data["w_idxs"]
    flat_w = sol[min(w_idxs): max(w_idxs) + 1]
    w_arr  = split_data(flat_w, K, N)

    # if all are already on the ball, accept or skip
    on_ball = all(abs(np.linalg.norm(w_arr[i]) - 1) < tol for i in range(K))
    if on_ball:
        # if lp obj==0, force branching; otherwise accept
        if prob.attributes.lpobjval == 0:
            return branch
        return branch  # or `return (0,0)` if you want to accept here

    # pick the “smallest‐norm” ball
    norms   = np.linalg.norm(w_arr, axis=1)
    ball_id = int(np.argmin(norms))

    nd = data.setdefault("node_data", {})
    nd[node] = {"w_array": w_arr, "ball_id": ball_id}

    # optional: skip if distances already tiny
    dist = nd[node].get("distance", [])
    if dist and max(dist) <= 1e-6:
        return branch

    # bound/gap test
    dual = prob.getAttrib("bestbound")
    if dual <= tol:
        nd[node]["branch_on_w"] = False
        return branch

    mipobj = prob.getAttrib("mipobjval")
    gap    = abs((mipobj - dual) / mipobj)

    # randomly choose x‐branch vs w‐branch
    branch_on_w = rng_node.random() >= max(gap, 1-gap)
    nd[node]["branch_on_w"] = branch_on_w
    if not branch_on_w:
        return branch

    # now build a small branchobj on the chosen ball face
    face     = data["extreme_points"][ball_id]
    proj_w   = ProjectOnBall(w_arr[ball_id])
    face2    = np.vstack((face, proj_w))

    try:
        coeffs2 = up_extension_constraint(face2)
    except:
        return branch

    bo = xp.branchobj(prob, isoriginal=True)
    bo.addbranches(N)
    w_vars = np.arange(ball_id * N, (ball_id + 1) * N)

    for i in range(N):
        # drop row i
        for j, cf in enumerate(coeffs2):
            rhs = 1 if j == 0 else 0
            if j > 0 and np.max(np.vstack((np.delete(face2, i, 0), proj_w)) @ cf) < 1e-6:
                cf = -cf
            bo.addrows(i, ['G'], [rhs], [0, N*K], w_vars, cf)

    return bo



def cbnewnode(prob, data, parentnode, newnode, branch):
    """
    When a new node is created:
     - if its parent was the root (parentnode==1), 
       we pick up the precomputed extreme‐points face 'branch'
     - otherwise, if the parent branched on x, we inherit unchanged
     - if the parent branched on w, we remove row 'branch' from the face,
       append the projected w, and record distances.
    """
    pdata       = data["pd"]
    N        = pdata.N
    node_data = data.setdefault("node_data", {})

    # initialize storage for this new node
    node_data[newnode] = {}

    # only act once the LP presolve has given us a solution
    if (prob.attributes.presolvestate & 128) == 0:
        return 0

    # Case 1: child of the root—branch index directly maps to a ball face
    if parentnode == 1:
        face = data["extreme_points"][branch]  # precomputed at root
        node_data[newnode]["face_matrix"] = face
        node_data[newnode]["ball_id"]      = branch
        return 0

    # Otherwise, look at what the parent did
    parent = node_data[parentnode]

    # If parent branched on x (branch flag False), just inherit
    if parent.get("branch") is False:
        node_data[newnode] = parent.copy()
        return 0

    # Parent branched on w: do the “remove one row + project” update
    w_arr   = parent["w_array"]
    ball_id = parent["ball_id"]

    # original face for this ball
    orig_face = data["extreme_points"][ball_id]

    # remove the 'branch'-th row
    subface = np.delete(orig_face, branch, axis=0)
    # project the chosen hyperplane onto the ball
    pi_w    = ProjectOnBall(w_arr[ball_id])

    # copy all of parent’s bookkeeping
    node_data[newnode] = parent.copy()
    # overwrite with the new face
    updated_face = np.vstack((subface, pi_w))
    node_data[newnode]["face_matrix"] = updated_face
    node_data[newnode]["ball_id"]      = ball_id

    # record how far each point moved
    dists = [
        np.linalg.norm(updated_face[i] - orig_face[i])
        for i in range(N)
    ]
    node_data[newnode]["distance"] = dists

    return 0

def cb_usersolnotify(prob, data, solname, status):
    """
    This callback is invoked *after* Xpress has tried to process
    a user‐solution added via prob.addmipsol().
    - solname is the string name you gave when you called addmipsol.
    - status is an integer describing the outcome:
        0 = error, 
        1 = feasible, 
        2 = feasible after reoptimize,
        … up to 8 = dropped.
    """
    print(f"[UserSolNotify] solution '{solname}' status={status}")

def cbnodelpsolved(prob, data):
    """
    After an LP solve at a node, occasionally invoke a MIP-free heuristic
    to produce a feasible solution if the LP lower bound is still too small.
    """
    pdata        = data["pd"]
    M, N, K   = pdata.M, pdata.N, pdata.K
    a, BigM   = pdata.a, pdata.BigM
    tol       = pdata.tol
    x_idxs    = data["x_idxs"]
    node_data = data.setdefault("node_data", {})
    refuse    = data["refuse_sol"]

    lower = prob.getAttrib("bestbound")
    rng   = np.random.default_rng(1234 + prob.attributes.currentnode)

    # only run when deep enough, a random chance, and LB < tol
    if (prob.attributes.currentnode < (N+1)**(K+1)
        or rng.random() >= 0.1
        or lower >= tol):
        return 0

    # fetch the LP solution
    try:
        sol = prob.getCallbackSolution(prob.getVariable())
    except:
        return 0

    # extract x and build assignment vector
    flat_x = sol[min(x_idxs) : max(x_idxs) + 1]
    x_mat  = split_data(flat_x, M, K)
    mask   = (x_mat == 1)
    result = np.where(mask.any(axis=1), np.argmax(mask, axis=1), -1)

    node = prob.attributes.currentnode
    nd   = node_data.setdefault(node, {})

    # --- compute (or reuse) hyperplane weights w and gammas ---
    if "heuristic_w" in nd:
        w     = nd["heuristic_w"]
        gamma = np.empty(K)
        for i in range(K):
            pts = a[result == i]
            if pts.shape[0] == 0:
                return 0
            gamma[i] = np.sum(pts @ w[i]) / pts.shape[0]
    else:
        w     = np.empty((K, N))
        gamma = np.empty(K)
        for i in range(K):
            pts = a[result == i]
            if pts.shape[0] == 0:
                return 0
            P       = np.eye(pts.shape[0]) - np.ones((pts.shape[0], pts.shape[0]))/pts.shape[0]
            B       = pts.T @ P @ pts
            q       = nd["w_array"][i]
            _, wj   = inverse_power_method(B, q, tol=1e-4, max_iter=30)
            gamma[i] = np.sum(pts @ wj) / pts.shape[0]
            w[i, :]  = wj

    # --- refine any unassigned points by alternating minimization ---
    unmask   = result < 0
    pts_un   = a[unmask]
    prev_ass = np.zeros(pts_un.shape[0])
    rows_idx = np.arange(M)

    for _ in range(100):
        w_old       = w.copy()
        dps         = pts_un @ w.T
        dists       = np.abs(dps - gamma)
        assigns     = np.argmin(dists, axis=1)
        result[unmask] = assigns

        # rebuild x and flatten
        x_full = np.zeros((M, K), dtype=int)
        x_full[rows_idx, result] = 1
        flat_x = x_full.ravel()

        # recompute (w, gamma, y)
        w_list, gamma_list, y = compute_w_gamma_y(a, flat_x, w_old, M, K, BigM)
        w     = np.array(w_list).reshape(K, N)
        gamma = np.array(gamma_list)

        if np.linalg.norm(prev_ass - assigns) < tol:
            break
        prev_ass = assigns.copy()

    # --- pack into a new MIP solution and stash it ---
    new_sol = np.concatenate([w.ravel(), gamma.ravel(), flat_x, np.asarray(y).ravel()])
    refuse.append(np.round(new_sol, 10).tolist())

    # remember for next time
    nd["heuristic_w"] = w

    return 0
    
def create_problem_defaults(n_planes = 3, dataset = None):
    """Read the given problem or create simple linear classifier problem."""
    prob = xp.problem()

    M = dataset.shape[0]
    N = dataset.shape[1]
    K = n_planes
    a = dataset

    h = np.max(a)
    BigM = h*np.sqrt(N)
    gamma_bound = N*h + h*np.sqrt(N)

    # Create variables using addVariables
    w = prob.addVariables(K, N, lb=-1, ub=1, name='w')
    gamma = prob.addVariables(K, lb=-gamma_bound, ub=gamma_bound, name='gamma')
    x = prob.addVariables(M, K, vartype=xp.binary, name='x')
    y = prob.addVariables(M, lb=0, name='y')

    # Add symmetry breaking on x
    for m in range(M):
        for k in range(K):
            if k > m:
                x[m, k].ub = 0

    # Add constraints
    for i in range(M):
        prob.addConstraint(xp.Sum(x[i,j] for j in range(min(i+1,K))) == 1)
        for j in range(K):
            if j <= i:
                prob.addConstraint(y[i] >= xp.Dot(w[j], a[i]) - gamma[j] - BigM*(1 - x[i,j]))
                prob.addConstraint(y[i] >= xp.Dot(-w[j], a[i]) + gamma[j] - BigM*(1 - x[i,j]))

    # Add norm constraints
    for j in range(K):
        prob.addConstraint(xp.Sum(w[j, i]*w[j, i] for i in range(N)) == 1 )

    # set objective
    prob.setObjective(xp.Sum(y[i]*y[i] for i in range(M)), sense=xp.minimize)
    
    return prob

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
    prob.addcbnodelpsolved(cbnodelpsolved, data, 2)
    # prob.addcbusersolnotify(cb_usersolnotify, data, 1)

    # 6) solver controls (all using local N,K)
    prob.controls.backtrack       = 5
    prob.controls.backtracktie   = 5
    prob.controls.timelimit      = 1800
    prob.controls.randomseed     = 42
    prob.controls.deterministic  = 1
    prob.controls.nodeselection  = 4
    prob.controls.breadthfirst   = (N + 1) ** K + 1
    # prob.controls.threads        = 1
    
    start_time = time.time()
    # 7) run
    prob.mipoptimize()
    computation_time = time.time() - start_time

    return prob, computation_time

if __name__ == "__main__":

    # You can adjust tol, number of random starts, batch size, etc.
    TOL        = 1e-4
    N_STARTS   = 100
    BATCH_SIZE = 2
    RESUME_IDX = 0

    datasets_filenames = [
        # "LowDim_with_noise.pkl",
        # "LowDim_no_noise.pkl",
        # "HighDim_with_noise.pkl",
        "HighDim_no_noise.pkl"
    ]

    for filename in datasets_filenames:
        print(f"Processing {filename} …")
        with open(filename, "rb") as f:
            datasets_dict = pickle.load(f)
        print(f"  Loaded {len(datasets_dict)} instances\n")

        results = []
        for idx, ((m, n, k), data_array) in enumerate(datasets_dict.items()):
            # if idx >= 2:
            #     continue
            
            # build your ProblemData
            problem_data = ProblemData(
                dataset  = data_array,
                n_planes = k,
                tol      = TOL
                )
            # override the random starts if you like
            problem_data.all_starts = [
                generate_random_matrix(k, n, problem_data.master_rng)
                for _ in range(N_STARTS)
                ]

            # solve it
            prob, computation_time = solve(problem_data)
            duration = round(computation_time, 3)
            
            # 3) collect metrics
            nodes     = prob.attributes.nodes
            bestbound = prob.attributes.bestbound

            print(f"  Instance {idx} (m={m},n={n},k={k}):")
            print(f"    nodes   = {nodes}")
            print(f"    best LB = {bestbound:.6f}")
            print(f"    time    = {duration}s\n")
            
            del prob
            gc.collect()

            results.append({
                "m": m,
                "n": n,
                "k": k,
                "Nodes": nodes,
                "BestLB": bestbound,
                "Time": duration
            })

            # batch‐write every BATCH_SIZE
            if len(results) == BATCH_SIZE:
                df = pd.DataFrame(results)
                outname = f"results_{filename[:-4]}_batch_{idx//BATCH_SIZE}.xlsx"
                df.to_excel(outname, index=False)
                print(f"  → Saved batch to {outname}\n")
                results = []

        # any leftover
        if results:
            df = pd.DataFrame(results)
            outname = f"results_{filename[:-4]}_batch_final.xlsx"
            df.to_excel(outname, index=False)
            print(f"  → Saved final batch to {outname}\n")