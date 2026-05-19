import xpress as xp
import numpy as np
from scipy.linalg import null_space, eigh
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
        self.all_starts = [generate_random_matrix(self.K, self.N, self.master_rng) for _ in range(100)]

def generate_random_matrix(K, N, rng):
    A = rng.standard_normal((K, N))
    gamma = rng.standard_normal(K)
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
        w, gamma, assign = _find_feasible_geometry(pdata, w_init, g_init)

        x = np.zeros((pdata.M, pdata.K), dtype=int)
        x[rows_idx, assign] = 1

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

        cluster_map = {}
        next_id = 0

        for i in range(pdata.M):
            c = assign[i]
            if c not in cluster_map:
                cluster_map[c] = next_id
                next_id += 1

        for c in range(pdata.K):
            if c not in cluster_map:
                cluster_map[c] = next_id
                next_id += 1

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

        dps = np.sum(pdata.a * w[assign], axis=1)
        y = np.abs(dps - gamma[assign])

        full_vector = np.concatenate([w.ravel(), gamma, x.ravel(), y])
        start_sols.append(np.round(full_vector, 10))

    return start_sols

def _find_feasible_geometry(pdata, w_start, g_start, batch_size=20, max_tries=100):
    """
    Ensures we have a starting (w, gamma) where every cluster has >= N points.
    """
    A, N, K = pdata.a, pdata.N, pdata.K
    rng = pdata.master_rng

    assign = np.argmin(np.abs(A @ w_start.T - g_start), axis=1)
    if np.all(np.bincount(assign, minlength=K) >= N):
        return w_start.copy(), g_start.copy(), assign

    for _ in range(max_tries):
        Ws = rng.standard_normal((batch_size, K, N))
        Ws /= np.linalg.norm(Ws, axis=2, keepdims=True)
        Gs = rng.standard_normal((batch_size, K))

        dots = np.matmul(A[None, :, :], Ws.transpose(0, 2, 1))
        dists = np.abs(dots - Gs[:, None, :])
        batch_assigns = np.argmin(dists, axis=2)

        counts = (batch_assigns[:, :, None] == np.arange(K)).sum(axis=1)

        valid_idx = np.where(np.all(counts >= N, axis=1))[0]
        if valid_idx.size > 0:
            best_i = valid_idx[0]
            return Ws[best_i], Gs[best_i], batch_assigns[best_i]

    return w_start, g_start, assign


def create_n_simplex(n):
    """Create a numpy array containing extreme points of an n-simplex using a Helmert matrix."""
    V_T = np.zeros((n + 1, n))

    for k in range(1, n + 1):
        val = 1.0 / np.sqrt(k * (k + 1))
        V_T[0:k, k - 1] = val
        V_T[k, k - 1] = -k * val

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

    if n_rows == n_cols and np.linalg.cond(E) < cond_thresh:
        a = np.linalg.solve(E, e)
    else:
        a = np.linalg.pinv(E) @ e

    coeffs_list = [a]

    for i in range(n_rows - 1):
        E_sub = np.delete(E, i, axis=0)
        null_basis = null_space(E_sub, rcond=null_rcond)

        if null_basis.size > 0:
            coeffs_list.append(null_basis.T)

    a_coeff = np.vstack(coeffs_list)

    unique_rows = []
    for row in a_coeff:
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
        return w
    else:
        return w / norm_w

def safe_tolist(arr):
    return arr.tolist() if hasattr(arr, "tolist") else arr

def compute_w_gamma_y(a, x, w_old, rows, cols, BigM):
    """
    Recalculate the optimal solution for hyperplane clustering given an integer solution x.
    """
    a = np.asarray(a)
    x = np.asarray(x).reshape(rows, cols)

    subarrays_per_j = {j: a[x[:, j].astype(bool)] for j in range(cols)}

    w_list = []
    gamma_list = []

    for j in range(cols):
        subarray = subarrays_per_j[j]
        n_points = subarray.shape[0]
        if n_points == 0:
            w_list.append(w_old[j])
            gamma_list.append(0)
            continue

        P = np.eye(n_points) - np.ones((n_points, n_points)) / n_points
        B_j = subarray.T @ P @ subarray

        # scipy.linalg.eigh finds the smallest eigenpair of the symmetric B_j directly
        _, vecs = eigh(B_j, subset_by_index=[0, 0])
        w_j = vecs[:, 0]

        gamma_j = np.sum(subarray @ w_j) / n_points

        w_list.append(w_j)
        gamma_list.append(gamma_j)

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

    M, N, K    = pdata.M, pdata.N, pdata.K
    a, BigM    = pdata.a, pdata.BigM
    gamma_bound = pdata.gamma_bound

    w     = prob.addVariables(K, N, lb=-1, ub=1, name="w")
    gamma = prob.addVariables(K, lb=-gamma_bound, ub=gamma_bound, name="gamma")
    x     = prob.addVariables(M, K, vartype=xp.binary, name="x")
    y     = prob.addVariables(M, lb=0, name="y")

    for i in range(M):
        for j in range(K):
            if j > i:
                x[i, j].ub = 0

    for i in range(M):
        prob.addConstraint(
            xp.Sum(x[i, j] for j in range(min(i+1, K))) == 1
        )
        for j in range(K):
            if j <= i:
                prob.addConstraint(
                    y[i] >= xp.Dot(w[j], a[i]) - gamma[j] - BigM*(1 - x[i, j])
                )
                prob.addConstraint(
                    y[i] >= xp.Dot(-w[j], a[i]) + gamma[j] - BigM*(1 - x[i, j])
                )

    for j in range(K):
        prob.addConstraint(
            xp.Sum(w[j, t]*w[j, t] for t in range(N)) <= 1
        )

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

    if (prob.attributes.presolvestate & 128) == 0:
        return (1, 0)

    try:
        sol = np.array(prob.getCallbackSolution(prob.getVariable()))
    except Exception:
        return (1, cutoff)

    w_idxs = data["w_idxs"]
    w_arr = sol[w_idxs[0] : w_idxs[-1] + 1].reshape(K, N)

    norms = np.linalg.norm(w_arr, axis=1)
    if np.all(np.abs(norms - 1.0) < tol):
        # If the LP obj is 0, we still reject
        # sometimes the solver finds feasible solution with 0 objective but it does not have norm = 1
        if prob.attributes.lpobjval == 0:
            return (1, cutoff)
        # Otherwise accept this solution
        return (0, cutoff)

    if np.all(np.any(np.abs(w_arr) > 1e-4, axis=1)):
        x_idxs = data["x_idxs"]
        x_flat = sol[x_idxs[0] : x_idxs[-1] + 1]

        new_w, new_gamma, new_y = compute_w_gamma_y(pdata.a, x_flat, w_arr, M, K, BigM)

        new_y_arr = np.asarray(new_y)
        if np.sum(new_y_arr**2) < prob.attributes.mipbestobjval:

            new_sol = np.concatenate([
                np.ravel(new_w),
                np.ravel(new_gamma),
                np.ravel(x_flat),
                new_y_arr
            ])

            data["refuse_sol"].append(np.round(new_sol, 10).tolist())

    return (1, cutoff)


def prenode_callback(prob, data):
    """
    Before diving into a new node, inject any heuristic MIP-solutions
    stored in data['refuse_sol'] (from cbchecksol), then clear the list.
    Only inject solutions that are strictly better than the current incumbent.
    """
    refuse = data.get("refuse_sol", [])
    if not refuse:
        return 0

    mipobj = prob.attributes.mipbestobjval
    y_idxs = data["y_idxs"]

    for sol in refuse:
        sol_arr = np.asarray(sol)
        if np.sum(sol_arr[y_idxs] ** 2) < mipobj:
            prob.addmipsol(sol)

    refuse.clear()

    return 0


def _build_simplex_data(data, N, K):
    """
    Initialise simplex geometry on the first branching call.
    Populates data['initial_polytope'], data['submatrix'], data['a_coeff'].
    """
    init_simplex = create_n_simplex(N)
    data["initial_polytope"] = init_simplex
    data["submatrix"] = {}
    data["a_coeff"] = {}

    for i in range(N + 1):
        face = np.delete(init_simplex, i, axis=0)  # shape (N, N)
        data["submatrix"][i] = face

        coeffs = up_extension_constraint(face)
        for j, c in enumerate(coeffs):
            if np.max(face @ c) < 1e-6:
                coeffs[j] = -c
        data["a_coeff"][i] = coeffs


def cbbranch(prob, data, branch):
    """
    Branching callback implementing two phases:

    INIT PHASE (init_depth < K):
        Branch (N+1)-way for ball `init_depth`, assigning one simplex face per
        branch.  This replaces the original (N+1)^K root fanout: nodes are now
        created lazily one ball at a time, so the solver can prune bad
        initialisation combinations via LP bounds before generating all combos.

    IPA PHASE (init_depth == K):
        LP-guided spatial branching — same logic as before, but the current
        node's face for the chosen ball is read from face_per_ball[ball_id]
        (set cumulatively by cbnewnode) rather than from a global dict indexed
        by the initial simplex face number.
    """
    pdata = data["pd"]
    N, K = pdata.N, pdata.K
    tol = pdata.tol
    node = prob.attributes.currentnode

    nd = data.setdefault("node_data", {})
    node_info = nd.get(node, {})
    # Root (node==1) has no entry yet → init_depth defaults to 0.
    # Non-init IPA nodes store init_depth == K explicitly.
    init_depth = node_info.get("init_depth", 0 if node == 1 else K)

    # =========================================================================
    # INIT PHASE: assign one simplex face per ball, one ball per tree level.
    # =========================================================================
    if init_depth < K:
        if "initial_polytope" not in data:
            _build_simplex_data(data, N, K)

        ball_to_init = init_depth
        bo = xp.branchobj(prob, isoriginal=True)
        bo.addbranches(N + 1)

        w_vars = np.arange(ball_to_init * N, (ball_to_init + 1) * N)
        for face_id in range(N + 1):
            for j, coeff in enumerate(data["a_coeff"][face_id]):
                rhs = 1 if j == 0 else 0
                bo.addrows(face_id, ['G'], [rhs], [0, N * K], w_vars, coeff)

        bo.setpriority(100)
        return bo

    # =========================================================================
    # IPA PHASE: LP-guided branching on x or w.
    # =========================================================================
    if (prob.attributes.presolvestate & 128) == 0:
        return branch

    try:
        sol = prob.getCallbackSolution(prob.getVariable())
    except Exception:
        return branch

    w_idxs = data["w_idxs"]
    flat_w = np.array(sol[w_idxs[0] : w_idxs[-1] + 1])
    w_arr = flat_w.reshape(K, N)

    norms = np.linalg.norm(w_arr, axis=1)

    if np.all(np.abs(norms - 1.0) < tol):
        return branch

    ball_id = int(np.argmin(norms))

    # Preserve face_per_ball written by cbnewnode when merging in w_array/ball_id.
    existing = nd.get(node, {})
    nd[node] = {**existing, "w_array": w_arr, "ball_id": ball_id}

    dist = nd[node].get("distance", [])
    if dist and max(dist) <= 1e-6:
        return branch

    dual = prob.getAttrib("bestbound")
    if dual <= tol:
        nd[node]["branch_on_w"] = False
        return branch

    mipobj = prob.getAttrib("mipobjval")

    if abs(mipobj) < 1e-9 or np.isinf(mipobj):
        gap = 1.0
    else:
        gap = abs((mipobj - dual) / mipobj)

    rng_node = np.random.default_rng(42 + node)
    branch_on_w = rng_node.random() >= max(gap, 1 - gap)
    nd[node]["branch_on_w"] = branch_on_w
    if not branch_on_w:
        return branch

    # Read the per-ball evolved face for ball_id.
    # face_per_ball holds a separate (N x N) face matrix for each of the K balls,
    # updated cumulatively by cbnewnode as the tree deepens.
    face_per_ball = nd[node].get("face_per_ball")
    face = face_per_ball[ball_id] if face_per_ball is not None else data["submatrix"][0]

    proj_w = ProjectOnBall(w_arr[ball_id])
    face2 = np.vstack((face, proj_w))

    if np.linalg.matrix_rank(face2, tol=1e-4) < N:
        bo = xp.branchobj(prob, isoriginal=True)
        bo.addbranches(0)
        return bo

    try:
        coeffs2 = up_extension_constraint(face2)
    except Exception:
        return branch

    bo = xp.branchobj(prob, isoriginal=True)
    bo.addbranches(N)
    w_vars = np.arange(ball_id * N, (ball_id + 1) * N)

    for i in range(N):
        for j, cf in enumerate(coeffs2):
            rhs = 1 if j == 0 else 0
            test_matrix = np.vstack((np.delete(face2, i, 0), proj_w))
            if j > 0 and np.max(test_matrix @ cf) < 1e-6:
                cf = -cf
            bo.addrows(i, ['G'], [rhs], [0, N * K], w_vars, cf)

    return bo


def cbnewnode(prob, data, parentnode, newnode, branch):
    """
    When a new node is created, propagate the correct per-ball face state.

    INIT PHASE (parent's init_depth < K):
        No LP state required.  Record which simplex face was assigned to
        ball `parent_init_depth` and increment init_depth for the child.

    IPA PHASE (parent's init_depth == K):
        Requires LP state.  If parent branched on w, update only
        face_per_ball[ball_id] with the refined face; all other balls
        inherit the parent's face unchanged.
    """
    pdata = data["pd"]
    N, K = pdata.N, pdata.K
    node_data = data.setdefault("node_data", {})

    parent = node_data.get(parentnode, {})
    # Root has no entry → its children begin init_depth at 0.
    # Non-root nodes without the key are in IPA phase (default K).
    parent_init_depth = parent.get("init_depth", 0 if parentnode == 1 else K)

    # =========================================================================
    # INIT PHASE: propagate face assignment and advance init_depth.
    # =========================================================================
    if parent_init_depth < K:
        # Each init-phase node records only the faces assigned so far.
        # branch is the face_id (0..N) chosen by Xpress for ball parent_init_depth.
        new_faces = list(parent.get("face_per_ball", [None] * K))
        new_faces[parent_init_depth] = data["submatrix"][branch]
        node_data[newnode] = {
            "face_per_ball": new_faces,
            "init_depth": parent_init_depth + 1,
        }
        return 0

    # =========================================================================
    # IPA PHASE: requires LP state to propagate w_array and evolve the face.
    # =========================================================================
    if (prob.attributes.presolvestate & 128) == 0:
        return 0

    if not parent:
        node_data[newnode] = {}
        return 0

    # Parent branched on x: inherit all state unchanged.
    if parent.get("branch_on_w") is False:
        node_data[newnode] = parent.copy()
        return 0

    # Parent branched on w: update only the face for the chosen ball.
    w_arr = parent["w_array"]
    ball_id = parent["ball_id"]

    # face_per_ball tracks a separate (N x N) face for each ball.
    # Update ball_id's face; the other K-1 balls inherit their faces unchanged.
    parent_faces = parent.get("face_per_ball", [None] * K)
    new_faces = list(parent_faces)

    orig_face = parent_faces[ball_id]
    subface = np.delete(orig_face, branch, axis=0)   # (N-1, N)
    pi_w = ProjectOnBall(w_arr[ball_id])              # (N,)
    new_faces[ball_id] = np.vstack((subface, pi_w))   # (N, N)

    distances = np.linalg.norm(
        new_faces[ball_id][:N] - orig_face[:N], axis=1
    ).tolist()

    new_state = parent.copy()
    new_state["face_per_ball"] = new_faces
    new_state["ball_id"] = ball_id
    new_state["distance"] = distances
    node_data[newnode] = new_state

    return 0


def solve(pdata: ProblemData) -> xp.problem:
    """
    Build, configure, and solve the Xpress model for the given ProblemData.
    All sizes and parameters come from pdata—no globals.
    """
    N, K = pdata.N, pdata.K

    prob = create_problem(pdata)

    all_vars   = prob.getVariable()
    w_idxs     = [i for i, v in enumerate(all_vars) if v.name.startswith("w")]
    gamma_idxs = [i for i, v in enumerate(all_vars) if v.name.startswith("gamma")]
    x_idxs     = [i for i, v in enumerate(all_vars) if v.name.startswith("x")]
    y_idxs     = [i for i, v in enumerate(all_vars) if v.name.startswith("y")]

    starts = starting_points(pdata, pdata.all_starts)

    # submatrix/a_coeff are populated lazily by cbbranch on first call.
    # extreme_points is no longer used — face_per_ball replaces it.
    data = {
        "pd": pdata,
        "w_idxs": w_idxs,
        "gamma_idxs": gamma_idxs,
        "x_idxs": x_idxs,
        "y_idxs": y_idxs,
        "refuse_sol": starts,
        "submatrix": {},
        "a_coeff": {},
        "node_data": {},
    }

    prob.addcbpreintsol(cbchecksol, data, 2)
    prob.addcbprenode(prenode_callback, data, 1)
    prob.addcbchgbranchobject(cbbranch, data, 2)
    prob.addcbnewnode(cbnewnode, data, 2)

    prob.controls.backtrack      = 5
    prob.controls.nodeselection  = 4
    prob.controls.backtracktie   = 5
    prob.controls.breadthfirst   = (N + 1) ** K + 1

    prob.controls.timelimit      = 3600
    prob.controls.randomseed     = 42
    prob.controls.deterministic  = 1
    # prob.controls.threads      = 1
    # prob.controls.maxnode      = 10000

    # to avoid numerical errors with LP
    # prob.controls.miprelstop = 1e-4
    # prob.controls.scaling    = 1
    # prob.controls.feastol    = 1e-7

    start_time = time.time()
    prob.mipoptimize()
    computation_time = time.time() - start_time

    return prob, computation_time


if __name__ == "__main__":

    TOL      = 1e-4
    N_STARTS = 100
    RESUME_IDX = 0

    TARGET = (22, 2, 2)  # (m, n, k) to benchmark

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
            if (m, n, k) != TARGET:
                continue

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
            outname = f"results_{filename[:-4]}_ISCA.xlsx"
            df.to_excel(outname, index=False)
            print(f"  → Saved final batch to {outname}\n")
