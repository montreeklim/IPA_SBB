import xpress as xp
import numpy as np
from scipy.linalg import null_space
import itertools
import time
import pickle
import pandas as pd
import re  # For extracting numbers from filenames
# import matplotlib.pyplot as plt
# import gurobipy as gp

np.set_printoptions(precision=3, suppress=True)
xp.init('C:/xpressmp/bin/xpauth.xpr') # license path for laptop

def generate_random_matrix(K, N, rng):
    # Generate a K x N matrix with random values from a normal distribution
    A = rng.standard_normal((K, N))
    gamma = rng.standard_normal(K)
    # Normalize each row of A
    row_norms = np.linalg.norm(A, axis=1, keepdims=True)
    A_normalized = A / row_norms
    return A_normalized, gamma

def starting_points(a, starts_list):
    M, _      = a.shape
    rows_idx  = np.arange(M)
    start_sols = []

    for w0, gamma0 in starts_list:
        w, gamma = w0.copy(), gamma0.copy()

        # 2a) initial assignment
        dot = a @ w.T
        dist = np.abs(dot - gamma)
        assign = np.argmin(dist, axis=1)
        valid_assignment = False
        while not valid_assignment:
            # Generate new hyperplane parameters
            w, gamma = generate_random_matrix(K, N, master_rng )
            dot_products = a @ w.T
            distances = np.abs(dot_products - gamma)  # gamma broadcasts along rows.
            assignments = np.argmin(distances, axis=1)
            
            # Check that each cluster has at least min_points_per_cluster points.
            valid_assignment = True
            for cluster in range(K):
                if np.sum(assignments == cluster) < N:
                    valid_assignment = False
                    break
        
        max_iter = 50
        prev_assign = assign.copy()
        for _ in range(max_iter):
            w_old = w.copy()
            dot   = a @ w.T
            dist  = np.abs(dot - gamma)
            assign = np.argmin(dist, axis=1)

            # build x_flat, call compute_w_gamma_y, etc.
            x      = np.zeros((M, K), int)
            x[rows_idx, assign] = 1
            w, gamma, y = compute_w_gamma_y(a, x.ravel(), w_old, M, K, BigM)
            w     = np.array(w).reshape(K, N)
            gamma = np.array(gamma)
            if np.linalg.norm(prev_assign - assign) < tol:
                break
            prev_assign = assign.copy()

        # 2c) pack into one vector
        new_sol = np.concatenate([w.ravel(), gamma, x.ravel(), y])
        start_sols.append(np.round(new_sol, 10))
    return start_sols

# def starting_points(a, n_points=10):
#     start_sols = []
#     M, _ = a.shape  # Number of data points
#     rows_idx = np.arange(M)
    
#     for i in range(n_points):
#         valid_assignment = False
#         # Repeat until the assignment meets the minimum point criteria per cluster.
#         while not valid_assignment:
#             # Generate new hyperplane parameters
#             w, gamma = generate_random_matrix(K, N)
#             dot_products = a @ w.T
#             distances = np.abs(dot_products - gamma)  # gamma broadcasts along rows.
#             assignments = np.argmin(distances, axis=1)
            
#             # Check that each cluster has at least min_points_per_cluster points.
#             valid_assignment = True
#             for cluster in range(K):
#                 if np.sum(assignments == cluster) < N:
#                     valid_assignment = False
#                     break
        
#         max_iter = 50
#         prev_assignment = assignments.copy()
#         for iteration in range(max_iter):
#             w_old = w.copy()
#             dot_products = a @ w.T
#             distances = np.abs(dot_products - gamma)
#             assignments = np.argmin(distances, axis=1)
            
#             # Create an assignment matrix (M x K)
#             x = np.zeros((M, K), dtype=int)
#             x[rows_idx, assignments] = 1
#             x_flat = x.ravel()
            
#             # Update hyperplane parameters and error vector y.
#             w, gamma, y = compute_w_gamma_y(a, x_flat, w_old, M, K, BigM)
#             w = np.asarray(w).reshape(K, N)
#             gamma = np.asarray(gamma)
            
#             # Convergence check: if assignments have changed very little, stop iterating.
#             if np.linalg.norm(prev_assignment - assignments) < tol:
#                 break
#             prev_assignment = assignments.copy()
        
#         # Concatenate all parts into a new solution vector.
#         new_sol = np.concatenate([w.ravel(), gamma.ravel(), x_flat, np.asarray(y).ravel()])
#         rounded_new_sol = np.round(new_sol, 10)
#         start_sols.append(rounded_new_sol)
    
#     return start_sols


def inverse_power_method(A, x0, tol=1e-6, max_iter=1000, reg=1e-10):
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

# Function to extract the last number from the filename (used for n_planes)
def get_n_planes_from_filename(filename):
    numbers = re.findall(r"\d+", filename)
    return int(numbers[-1])  # Return the last number in the filename

def get_n_planes_from_tuple(key_tuple):
    """
    Extracts the number of planes from the (m, n, k) tuple.
    Assuming n_planes corresponds to 'k'.
    """
    _, _, k = key_tuple
    return k


def ball_idx_to_combination(ball_idx, n, k):
    """
    Converts a given ball_idx back into the original combination.
    
    Parameters:
        ball_idx: The integer representing the combination (ball_idx).
        n: The upper bound of the range (n+1 values).
        k: The number of loops or elements in the combination.
    
    Returns:
        combination: The list representing the original combination.
    """
    combination = []
    base = n + 1  # Since values range from 0 to n (inclusive)
    
    for i in range(k):
        # Calculate the current index element from ball_idx
        power = base ** (n - i)
        element = ball_idx // power
        combination.append(element)
        
        # Reduce ball_idx by the contribution of this element
        ball_idx -= element * power

    return combination

def gram_schmidt(A):
    """Orthogonalize a set of vectors stored as the columns of matrix A."""
    # Compute the QR decomposition of A
    Q, _ = np.linalg.qr(A)
    return Q

def create_n_simplex(n):
    """Create a numpy array contains extreme points of n-simplex."""
    E = np.identity(n+1)
    c = np.ones(n+1)/(n+1)
    A = E.copy()
    for i in range(n):
        A[i] = E[i]-c
    A = A[0:n, :].T
    
    U = gram_schmidt(A)
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

def check_zero_norms(w_array):
    """
    Checks if any element in each row (1D sub-array) of w_array has a zero norm,
    using np.linalg.norm for calculating norms.

    Args:
        w_array: A NumPy array of any shape.

    Returns:
        A NumPy array of booleans, with the same number of rows as w_array.
        Each element indicates whether the corresponding row has at least one
        element with a non-zero norm.
    """
    global tol

    # Calculate norms for each element
    norms = np.linalg.norm(w_array, axis=-1)

    # Check if any norm in each row is non-zero
    at_least_one_non_zero = np.any(norms < tol)

    return at_least_one_non_zero


def are_rows_near_duplicates(row1, row2, tol=1e-6):
    '''
    Function to check if rows are nearly identical

    Parameters
    ----------
    row1 : np.array
    row2 : np.array
    tol : float, optional
        The default is 1e-6.

    Returns
    -------
    bool
        True if two rows are nearly identical, False otherwise

    '''
    if np.linalg.norm(row1 - row2) < tol or np.linalg.norm(row1 + row2) < tol:
        return True
    else:
        return False

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
    unique_rows = []
    for i in range(a_coeff.shape[0]):
        is_duplicate = False
        for unique_row in unique_rows:
            if are_rows_near_duplicates(a_coeff[i], unique_row):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_rows.append(a_coeff[i])

    # Convert the list of unique rows back to a NumPy array
    A_reduced = np.array(unique_rows)
    
    return A_reduced

def CheckOnBall(w):
    """Check if the obtained solution norm is almost one or not."""
    # tol = 1e-6 if w close to pi(w) accept the 
    global tol
    if np.abs(np.linalg.norm(w) - 1) < tol:
        return True
    else:
        return False

def check_all_balls(w_array):
    """Check if the obtained solution norm is almost one or not."""
    # tol = 1e-6 if w close to pi(w) accept the solution
    global tol
    for i in range(len(w_array)):    
        if np.abs(np.linalg.norm(w_array[i]) - 1) >= tol:
            return False
    return True

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
        eigenvalue, w_j = inverse_power_method(B_j, w_old[j], tol=1e-6, max_iter=100)

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

# def create_gurobi_model(n_planes = 3, dataset = None):
        
#     model = gp.Model()
    
#     global M, N, K, a, BigM
        
#     M = dataset.shape[0]
#     N = dataset.shape[1]
#     K = n_planes
#     a = dataset
 
#     h = np.max(a)
#     BigM = h*np.sqrt(N)
#     gamma_bound = N*h + h*np.sqrt(N)

#     # Gurobi variable creation
#     w = model.addVars(K, N, name='w', lb=-1, ub=1)
#     gamma = model.addVars(K, name="gamma", lb=-gamma_bound, ub = gamma_bound)
#     x = model.addVars(M, K, vtype=gp.GRB.BINARY, name='x')
#     y = model.addVars(M, name='y', lb=0)

#     # Gurobi constraint creation
#     # Add constraints
#     for i in range(M):
#         model.addConstr(sum(x[i,j] for j in range(min(i+1,K))) == 1)
#         for j in range(K):
#             if j<=i:
#                 model.addConstr(y[i] >= gp.quicksum(w[j,k]*a[i][k] for k in range(N)) - gamma[j] - BigM*(1-x[i,j]))
#                 model.addConstr(y[i] >= gp.quicksum(-w[j,k]*a[i][k] for k in range(N)) + gamma[j] - BigM*(1-x[i,j]))

#         # Gurobi quadratic norm constraints
#         for j in range(K):
#             model.addQConstr(sum(w[j, i] * w[j, i] for i in range(N)) >= 1)

#         # Gurobi objective
#         model.setObjective(sum(y[i]*y[i] for i in range(M)), sense=gp.GRB.MINIMIZE)
#     return model # Handle the case where a filename is provided

    
def create_problem(n_planes = 3, dataset = None):
    """Read the given problem or create simple linear classifier problem."""
    prob = xp.problem()
    global M, N, K, a, BigM

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
        prob.addConstraint(xp.Sum(w[j, i]*w[j, i] for i in range(N)) <= 1 )

    # set objective
    prob.setObjective(xp.Sum(y[i]*y[i] for i in range(M)), sense=xp.minimize)

    global all_variables, w_variables_idxs, gamma_variables_idxs, x_variables_idxs, y_variables_idxs, refuse_sol, x_vars, new_list
    refuse_sol = starting_points(a, all_starts)
    all_variables = prob.getVariable()
    w_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("w")]
    gamma_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("gamma")]
    x_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("x")]
    y_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("y")]
    new_list = w_variables_idxs + x_variables_idxs + y_variables_idxs

    return prob

# def cbchecksol(prob, data, soltype, cutoff):
#     """
#     Intercept each integer solution, check your 'ball' condition, and
#     – if it fails but is valid (all_non_zero) – immediately inject the
#     Mangasarian‐heuristic solution as a new incumbent.
#     """
#     global BigM, a, tol, w_variables_idxs, x_variables_idxs, K, N

#     # Only run after presolve is done
#     if (prob.attributes.presolvestate & 128) == 0:
#         return (1, cutoff)

#     # Get the candidate solution
#     try:
#         sol = prob.getCallbackSolution(prob.getVariable())
#     except:
#         return (1, cutoff)

#     # Extract w and check norms
#     w_sol   = sol[min(w_variables_idxs): max(w_variables_idxs)+1]
#     w_array = split_data(w_sol, K, N)
#     if check_all_balls(w_array):
#         # on‐ball: accept
#         return (0, cutoff)

#     # off‐ball but non‐zero in every row → build a Mangasarian candidate
#     non_zero_check = np.any(np.abs(w_array) > tol, axis=1)
#     if np.all(non_zero_check):
#         # build the new solution
#         x_sol = sol[min(x_variables_idxs): max(x_variables_idxs) + 1]
#         new_w, new_gamma, new_y = compute_w_gamma_y(a, x_sol, w_array, M, K, BigM)

#         # prepare the flat Python list
#         new_sol = (
#             list(new_w) +
#             list(new_gamma) +
#             list(x_sol) +
#             list(new_y)
#         )
#         new_sol_round = [round(v, 10) for v in new_sol]

#         # **INJECT IT IMMEDIATELY** and don’t keep it around
#         prob.addmipsol(new_sol_round)

#         # accept the new incumbent
#         return (0, cutoff)

#     # anything else: reject the original
#     return (1, cutoff)


def cbchecksol(prob, data, soltype, cutoff):
    """Callback function to reject the solution if it is not on the ball and accept otherwise."""
    # print('Enter Preintsol Callback')
    try:
        global BigM, a, tol
        if (prob.attributes.presolvestate & 128) == 0:
            return (1, 0)


        # Retrieve node solution
        try:
            sol = prob.getCallbackSolution(prob.getVariable())
        except:
            return (1, cutoff)

        w_sol = sol[min(w_variables_idxs): max(w_variables_idxs)+1]
        w_array = split_data(w_sol, K, N)
        non_zero_check = np.any(np.abs(w_array) > 1e-4, axis=1)
        all_non_zero = np.all(non_zero_check)

        if check_all_balls(w_array):
            refuse = 0
            
            if prob.attributes.lpobjval == 0:
                return (1, cutoff)
        elif all_non_zero:
            refuse = 1
            # Add a feasible solution from Mangasarian from leaf node
            x_sol = sol[min(x_variables_idxs): max(x_variables_idxs) + 1]
            new_w, new_gamma, new_y = compute_w_gamma_y(a, x_sol, w_array, M, K, BigM)

            # Convert arrays to lists if needed.
            new_w, new_gamma, x_sol, new_y = [safe_tolist(arr) for arr in (new_w, new_gamma, x_sol, new_y)]

            # Concatenate the lists to form the new solution.
            new_sol = new_w + new_gamma + x_sol + new_y
            new_sol_round = np.array([round(a, 10) for a in new_sol])
            refuse_sol.append(new_sol_round) 
        else:
            refuse = 1

        return (refuse, cutoff)

    except Exception:
        return (1, cutoff)

def prenode_callback(prob, data):
    # print('Enter Prenode Callback')
    global refuse_sol

    if len(refuse_sol) > 0:
        for sol in refuse_sol:
            # print('Before add mip sol')
            # data['last_refuse_sol'] = refuse_sol.copy()
            # values = sol[new_list]
            # print('Injecting Solution = ', sol)
            prob.addmipsol(sol)
            # prob.addmipsol(
            #     mipsolval = sol,
            #     # mipsolcol = new_list,
            #     # solname   = "heuristic"
            #     )
            # print('After add mip sol')
        refuse_sol = []
        # remove callback
        # prob.removecbprenode()
        
    return 0

def cbbranch(prob, data, branch):
    # print('Enter Branching Callback')
    global initial_polytope, extreme_points
    currentnode = prob.attributes.currentnode
    # make a fresh RNG whose seed is a function of the node ID
    rng_node = np.random.default_rng(  42 + currentnode )

    if currentnode == 1:
        # Root node: build full branching object.
        bo = xp.branchobj(prob, isoriginal=True)
        bo.addbranches((N + 1) ** K)
        initial_polytope = create_n_simplex(N)
        extreme_points = {}
        a_coeff = {}
        submatrix = {}

        # Precompute submatrices, extreme points, and their associated constraint coefficients.
        for i in range(N + 1):
            submat = np.delete(initial_polytope, i, axis=0)
            submatrix[i] = submat
            extreme_points[i] = submat
            coeff = up_extension_constraint(submat)
            # Vectorized dot-products: adjust sign if all dot-products are very close to zero.
            for j in range(len(coeff)):
                # Use np.dot on the entire submatrix rather than a list comprehension.
                dot_products = submat @ coeff[j]
                if np.max(dot_products) < 1e-6:
                    coeff[j] = -coeff[j]
            a_coeff[i] = coeff

        # Precompute powers for computing ball_idx.
        power_factors = [(N + 1) ** (K - k - 1) for k in range(K)]
        values = range(N + 1)
        for combination in itertools.product(values, repeat=K):
            ball_idx = sum(combination[k] * power_factors[k] for k in range(K))
            extreme_points[ball_idx] = {}
            for k in range(K):
                w_ball_idx = np.arange(k * N, (k + 1) * N)
                coeff_list = a_coeff[combination[k]]
                for j, coeff in enumerate(coeff_list):
                    rhs_value = 1 if j == 0 else 0
                    bo.addrows(ball_idx, ['G'], [rhs_value], [0, N * K], w_ball_idx, coeff)
                extreme_points[ball_idx][k] = submatrix[combination[k]]
        bo.setpriority(100)
        return bo

    else:
        # Non-root node branch processing.
        # Skip if presolvestate flag is not set.
        if (prob.attributes.presolvestate & 128) == 0:
            return branch

        try:
            sol = prob.getCallbackSolution(prob.getVariable())
        except Exception:
            return branch

        # Extract and reshape w part of the solution.
        w_sol = sol[min(w_variables_idxs): max(w_variables_idxs) + 1]
        w_array = split_data(w_sol, K, N)
        split_index = split_data(w_variables_idxs, K, N)

        if check_all_balls(w_array):
            return branch

        norms = np.linalg.norm(w_array, axis=-1)
        ball_idx = int(np.argmin(norms))
        w_ball_idx = list(split_index[ball_idx])

        node_data = data[prob.attributes.currentnode]
        node_data['w_array'] = w_array
        node_data['ball_idx'] = ball_idx

        try:
            if max(node_data['distance']) <= 1e-6:
                return branch
        except Exception:
            pass

        dual_bound = prob.getAttrib("bestbound")
        if dual_bound <= tol:
            node_data['branch'] = False
            return branch
        else:
            best_solution = prob.getAttrib("mipobjval")
            mip_gap = abs((best_solution - dual_bound) / best_solution)

        # Randomized decision: branch on x or w.
        if rng_node.random() < max(mip_gap, 1 - mip_gap):
            node_data['branch'] = False
            return branch
        else:
            node_data['branch'] = True

        # Project the w_array onto the ball.
        pi_w_array = [ProjectOnBall(w_j) for w_j in w_array]
        initial_points = node_data[ball_idx]
        new_matrix = append_zeros_and_ones(initial_points)

        rank_initial = np.linalg.matrix_rank(initial_points, tol=1e-6)
        if rank_initial < N or np.linalg.matrix_rank(new_matrix, tol=1e-6) != rank_initial:
            return branch

        bo = xp.branchobj(prob, isoriginal=True)
        bo.addbranches(N)
        # For each possible row deletion (each branch)
        for i in range(N):
            submat = np.delete(initial_points, i, axis=0)
            # Append the projected w for the current ball.
            extreme_temp = np.concatenate((submat, [pi_w_array[ball_idx]]), axis=0)
            try:
                a_coeff = up_extension_constraint(extreme_temp)
            except Exception:
                return branch
            for j, coeff in enumerate(a_coeff):
                if j == 0:
                    bo.addrows(i, ['G'], [1], [0, N * K], w_ball_idx, coeff)
                else:
                    dot_products = extreme_temp @ coeff
                    if np.max(dot_products) < 1e-6:
                        coeff = -coeff
                    bo.addrows(i, ['G'], [0], [0, N * K], w_ball_idx, coeff)
        return bo

    return branch


def cbnewnode(prob, data, parentnode, newnode, branch):
    # print('Enter NEWNODE callback')
    data[newnode] = {}

    if (prob.attributes.presolvestate & 128) == 0:
        return 0

    if parentnode == 1:
        data[newnode] = extreme_points[branch]
    else:
        # print('Not in the root node')
        # if parentnode is branch on x
        if not data[parentnode]['branch']:
            # print('Branch on x')
            data[newnode] = data[parentnode]
            return 0

        # IPA branch on w
        w_array = data[parentnode]['w_array']
        ball_idx = data[parentnode]['ball_idx']

        initial_polytope = data[parentnode][ball_idx]
        submatrix = np.delete(initial_polytope, branch, axis=0)
        pi_w = ProjectOnBall(w_array[ball_idx])
        for key in data[parentnode].keys():
            data[newnode][key] = data[parentnode][key]
        data[newnode][ball_idx] = np.vstack((submatrix, pi_w))

        distance = [np.linalg.norm(data[newnode][ball_idx][n] - data[parentnode][ball_idx][n]) for n in range(N)]
        data[newnode]['distance'] = distance

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
    # e.g. if you only want to re-inject ones that were fixed by reoptimize:
    # if status == 2:
    #     # grab your repaired vector back from data and re-add it
    #     repaired = data.get('to_inject', None)
    #     if repaired is not None:
    #         prob.addmipsol(repaired, name=solname + "_retry")

# def cbnodelpsolved(prob, data):

#     depth = prob.attributes.nodedepth
#     time_until = time.time() - start_time
#     lower_bound = prob.getAttrib("bestbound")
#     upper_bound = prob.getAttrib("mipobjval")
#     data_dict[data['node_count']] = {'depth':depth, 'time': time_until, 'lower_bound':lower_bound, 'upper_bound':upper_bound}
#     rng = np.random.default_rng( 1234 + prob.attributes.currentnode )
    # Only execute if conditions are met.
    # or lower_bound >= 1e-6
    # if prob.attributes.currentnode < (N + 1) ** (K + 1) or rng_heuristic.random() >= 0.1:
    # if prob.attributes.currentnode < (N + 1) ** (K + 1) or rng.random() >= 0.1 or lower_bound >= 1e-6:
    #     return 0

    # try:
    #     # Retrieve the callback solution and extract the x portion.
    #     sol = prob.getCallbackSolution(prob.getVariable())
    #     min_idx, max_idx = min(x_variables_idxs), max(x_variables_idxs) + 1
    #     x_sol = split_data(sol[min_idx:max_idx], M, K)

    #     # Determine cluster assignments using a binary mask.
    #     mask = (x_sol == 1)
    #     result = np.where(mask.any(axis=1), np.argmax(mask, axis=1), -1)

    #     # Use a local reference to the current node's data.
    #     currentnode = prob.attributes.currentnode
    #     current_data = data[currentnode]

    #     # Compute hyperplane weights and gamma values.
    #     if 'heuristic_w' in current_data:
    #         w = current_data['heuristic_w']
    #         gamma = np.empty(K)
    #         for i in range(K):
    #             sub_a = a[result == i]
    #             n = sub_a.shape[0]
    #             if n == 0:
    #                 return 0
    #             gamma[i] = np.sum(sub_a @ w[i]) / n
    #     else:
    #         w = np.empty((K, N))
    #         gamma = np.empty(K)
    #         for i in range(K):
    #             sub_a = a[result == i]
    #             n = sub_a.shape[0]
    #             if n == 0:
    #                 return 0
    #             # Compute projection matrix: P = I - ones((n,n))/n.
    #             P = np.eye(n) - np.ones((n, n)) / n
    #             B = sub_a.T @ P @ sub_a
    #             q = current_data['w_array'][i]
    #             _, w_j = inverse_power_method(B, q, tol=1e-6, max_iter=100)
    #             gamma[i] = np.sum(sub_a @ w_j) / n
    #             w[i, :] = w_j

    #     # Process unassigned points (those with result < 0).
    #     mask_unassigned = result < 0
    #     unassign_points = a[mask_unassigned]
    #     prev_assignment = np.zeros(unassign_points.shape[0])
    #     max_iter = 100
    #     rows_idx = np.arange(M)  # Precompute row indices for x reconstruction.

    #     for iteration in range(max_iter):
    #         w_old = w.copy()
    #         # Calculate distances from each unassigned point to each hyperplane.
    #         dot_products = unassign_points @ w.T
    #         distances = np.abs(dot_products - gamma)  # gamma broadcasts along rows.
    #         assignments = np.argmin(distances, axis=1)
    #         result[mask_unassigned] = assignments

    #         # Reconstruct x as an M x K binary matrix.
    #         x = np.zeros((M, K), dtype=int)
    #         x[rows_idx, result] = 1
    #         x_flat = x.ravel()

    #         # Update hyperplane parameters (and obtain error vector y).
    #         w, gamma, y = compute_w_gamma_y(a, x_flat, w_old, M, K, BigM)
    #         w = np.asarray(w).reshape(K, N)
    #         gamma = np.asarray(gamma)

    #         # Convergence check: if assignments change little, stop iterating.
    #         if np.linalg.norm(prev_assignment - assignments) < tol:
    #             break
    #         prev_assignment = assignments.copy()

    #     # Concatenate all parts into a new solution.
    #     new_sol = np.concatenate([w.ravel(), gamma.ravel(), x_flat, np.asarray(y).ravel()])
    #     rounded_new_sol = np.round(new_sol, 10)
    #     refuse_sol.append(rounded_new_sol.tolist())

    #     # Update data with the current heuristic weights.
    #     current_data['heuristic_w'] = w
    #     return 0
    # except Exception:
    #     return 0

def solveprob(prob, initial_refuse=None):
    # global refuse_sol
    # If the user provided an initial test solution, stick it in refuse_sol now
    # if initial_refuse is not None:
    #     refuse_sol = initial_refuse.copy()
        
    data = {}
    # data[1] = {}
    data['node_count'] = 0
    data['x_sol']= []
    # data['dataframe'] = {}
    prob.addcbpreintsol(cbchecksol, data, 2)
    prob.addcbchgbranchobject(cbbranch, data, 2)
    prob.addcbnewnode(cbnewnode, data, 2)
    prob.addcbprenode(prenode_callback, data, 1)
    prob.addcbusersolnotify(cb_usersolnotify, data, 1)
    # prob.addcbnodelpsolved(cbnodelpsolved, data, 2)
    # prob.controls.outputlog = 0
    # prob.controls.feastol = 1e-4
    # prob.controls.feastoltarget = 1e-4
    # prob.controls.optimalitytoltarget = 1e-4
    prob.controls.refineops = 531
    # prob.controls.presolve = 0
    prob.controls.backtrack = 5
    prob.controls.backtracktie = 5
    prob.controls.timelimit = 180
    prob.controls.randomseed = 42
    prob.controls.deterministic = 1
    prob.controls.nodeselection = 4
    prob.controls.breadthfirst = (N+1)**K + 1
    prob.controls.threads = 1
    # prob.controls.miplog = -100
    # prob.controls.maxnode = 100

    prob.mipoptimize("")
    # return data
    
def create_problem_defaults(n_planes = 3, dataset = None):
    """Read the given problem or create simple linear classifier problem."""
    prob = xp.problem()
    global M, N, K, a, BigM

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

if __name__ == '__main__':
    tol = 1e-4
    # List of dataset filenames
    datasets_filenames = [
        "LowDim_with_noise.pkl",
        # "LowDim_no_noise.pkl",
        # "HighDim_with_noise.pkl",
        # "HighDim_no_noise.pkl"
    ]
    
    # Number of instances per batch
    batch_size = 2
    # Total instances already processed in previous batches (2 batches x 5 instances per batch)
    resume_from_instance = 0 * batch_size

    for filename in datasets_filenames:
        print(f"Processing dataset file: {filename}")

        # Load the dataset
        with open(filename, "rb") as f:
            datasets_dict = pickle.load(f)
        print(f"Loaded {len(datasets_dict)} datasets from '{filename}'.\n")
        
        combined_data = []
        instance_counter = 0  # counts processed instances in this run (not total)
        batch_counter = resume_from_instance // batch_size

        # Make a list of key-value pairs so that we can use an index to resume
        dataset_items = list(datasets_dict.items())
        
        # Iterate over each (m, n, k) tuple and its corresponding data starting from the resume index
        for idx, (key_tuple, data_array) in enumerate(dataset_items):
            if idx < resume_from_instance:
                # Skip instances that were already processed in previous batches
                continue

            m, n, k = key_tuple
            # if (m,n,k) != (18,3,2) and (m,n,k) != (22,2,2):
            if (m,n,k) != (14,3,2):
                continue
            
            # print(m, n, k)
            master_rng = np.random.default_rng(42)
            n_points = 100   # however many starting points you want
            # suppose K and N are already defined for your dataset
            all_starts = [ generate_random_matrix(k, n, master_rng)
                          for _ in range(n_points) ]

            # Initialize a list to store DataFrames for each scenario
            dfs = []
            for scenario in range(1):  # Adjust the number of scenarios if needed
                # global data_dict
                # rng_branch = np.random.default_rng(seed=42 + scenario)
                # rng_heuristic = np.random.default_rng(seed=123 + scenario)
                # data_dict = {}
                print(f"--- Scenario {scenario + 1} ---")
                # np.random.seed(scenario)  # For reproducibility
                
                num_nodes = []
                mip_bound = []
                start_time = time.time()

                # Create and solve the problem for the current scenario
                prob = create_problem(n_planes=k, dataset=data_array)
                solveprob(prob)
                
                nodes = prob.attributes.nodes
                bestbound = prob.attributes.bestbound
                solve_time = round(time.time() - start_time, 3)
                num_nodes.append(nodes)
                mip_bound.append(bestbound)

                # Create the DataFrame for this scenario
                df = pd.DataFrame({
                    "Objective": mip_bound,
                    "Nodes": num_nodes,
                    "Time": solve_time
                })
                dfs.append(df)
                print(f"Results for Scenario {scenario + 1}:\n{df}\n")
    
        #     # Default experiment for the current instance
        #     default_objective = []
        #     default_nodes = []
        #     default_time = []
        #     start_time = time.time()
            
        #     prob_default = create_problem_defaults(n_planes=k, dataset=data_array)
        #     prob_default.controls.timelimit = 1800
        #     # prob_default.optimize('x')
            
        #     default_bestbound = prob_default.attributes.bestbound
        #     default_nodes_count = prob_default.attributes.nodes
        #     default_elapsed_time = round(time.time() - start_time, 3)
        #     default_objective.append(default_bestbound)
        #     default_nodes.append(default_nodes_count)
        #     default_time.append(default_elapsed_time)
            
        #     # Calculate averages across scenarios
        #     average_nodes = []
        #     average_time = []
        #     for i in range(len(dfs[0])):
        #         nodes_values = [
        #             dfs[0]["Nodes"][i] if dfs[0]["Nodes"][i] is not None else np.nan,
        #             dfs[1]["Nodes"][i] if dfs[1]["Nodes"][i] is not None else np.nan,
        #             dfs[2]["Nodes"][i] if dfs[2]["Nodes"][i] is not None else np.nan
        #         ]
        #         times_values = [
        #             dfs[0]["Time"][i] if dfs[0]["Time"][i] is not None else np.nan,
        #             dfs[1]["Time"][i] if dfs[1]["Time"][i] is not None else np.nan,
        #             dfs[2]["Time"][i] if dfs[2]["Time"][i] is not None else np.nan
        #         ]
        #         avg_nodes = int(np.nanmean(nodes_values)) if not np.all(np.isnan(nodes_values)) else None
        #         avg_time = round(np.nanmean(times_values), 3) if not np.all(np.isnan(times_values)) else None
        #         average_nodes.append(avg_nodes)
        #         average_time.append(avg_time)
            
        #     # Combine results for this instance
        #     for i in range(len(dfs[0])):
        #         combined_data.append([
        #             m,  # m parameter
        #             n,  # n parameter
        #             k,  # k parameter
        #             # Scenario 1
        #             dfs[0].loc[i, "Objective"], dfs[0].loc[i, "Nodes"], dfs[0].loc[i, "Time"],
        #             # Scenario 2
        #             dfs[1].loc[i, "Objective"], dfs[1].loc[i, "Nodes"], dfs[1].loc[i, "Time"],
        #             # Scenario 3
        #             dfs[2].loc[i, "Objective"], dfs[2].loc[i, "Nodes"], dfs[2].loc[i, "Time"],
        #             # Averages
        #             average_nodes[i], average_time[i],
        #             # Default Experiment
        #             default_objective[i], default_nodes[i], default_time[i]
        #         ])
        #         instance_counter += 1

        #         # Save every batch_size instances
        #         if instance_counter % batch_size == 0:
        #             formatted_df = pd.DataFrame(combined_data, columns=[
        #                 "m", "n", "k",  # Dataset parameters
        #                 "1 - Objective", "1 - Nodes", "1 - Time",
        #                 "2 - Objective", "2 - Nodes", "2 - Time",
        #                 "3 - Objective", "3 - Nodes", "3 - Time",
        #                 "Average Nodes", "Average Time",
        #                 "Default - Objective", "Default - Nodes", "Default - Time"
        #             ])
        #             output_filename = f"results_batch_{batch_counter}.xlsx"
        #             formatted_df.to_excel(output_filename, index=False)
        #             print(f"Batch {batch_counter}: Saved {batch_size} instances to '{output_filename}'.\n")
        #             combined_data = []  # Reset for the next batch
        #             batch_counter += 1

        # # Save any remaining instances that didn't complete a full batch
        # if combined_data:
        #     formatted_df = pd.DataFrame(combined_data, columns=[
        #         "m", "n", "k",  # Dataset parameters
        #         "1 - Objective", "1 - Nodes", "1 - Time",
        #         "2 - Objective", "2 - Nodes", "2 - Time",
        #         "3 - Objective", "3 - Nodes", "3 - Time",
        #         "Average Nodes", "Average Time",
        #         "Default - Objective", "Default - Nodes", "Default - Time"
        #     ])
        #     # output_filename = f"results_batch_{batch_counter}.xlsx"
        #     # formatted_df.to_excel(output_filename, index=False)
        #     # print(f"Final batch: Saved remaining {len(combined_data)} instance(s) to '{output_filename}'.\n")


# if __name__ == '__main__':
    
#     tol = 1e-4
#     # List of dataset filenames
#     datasets_filenames = [
#         # "instances_10_2_2.pkl",
#         "instances_10_2_3.pkl",
#         # "instances_12_2_3.pkl",
#         # "instances_14_2_3.pkl",
#         # "instances_12_3_2.pkl"
#         ]

#     for filename in datasets_filenames:
#         print(f"Processing dataset: {filename}")

#         # Load the dataset
#         with open(filename, "rb") as f:
#             datasets = pickle.load(f)

#         # Extract n_planes from the filename
#         n_planes = get_n_planes_from_filename(filename)
#         print(f"Using n_planes = {n_planes}")

#         # Initialize lists to store DataFrames for each scenario
#         dfs = []

#         # Run experiments for three different configurations/scenarios
#         for scenario in range(1):  # Adjust the number of scenarios if needed
#             # np.random.seed(scenario)
#             global rng_branch, rng_heuristic, data_dict
#             rng_branch = np.random.default_rng(seed=42 + scenario)
#             rng_heuristic = np.random.default_rng(seed=123 + scenario)
#             num_nodes = []
#             mip_bound = []
#             solve_time = []
#             start_time = time.time()
#             data_dict = {}
            
#             # Test one instance
#             dataset = datasets[9]
#             # gurobi_model = create_gurobi_model(n_planes=n_planes, dataset=dataset)
#             # gurobi_model.setParam('OutputFlag', 0)
#             # gurobi_model.optimize()
#             # global gurobi_sol
#             # gurobi_sol = gurobi_model.x
#             prob = create_problem(n_planes=n_planes, dataset=dataset)
#             solveprob(prob)
#             df = pd.DataFrame.from_dict(data_dict, orient='index')
#             df.index.name = 'index'
#             df = df.reset_index()
#             print(prob.attributes.nodes)
#             print(prob.attributes.bestbound)
#             print(time.time() - start_time)
            
# #             # # First, filter the DataFrame for rows with upper_bound < 1
# #             # df_filtered = df[df['upper_bound'] < 2]

# #             # # For the depth plot, ensure that for each unique depth only the row with the highest index is kept.
# #             # # Sorting by index ensures that the last occurrence is the one with the highest index.
# #             # df_depth = df_filtered.drop_duplicates(subset=['depth'], keep='last')
# #             # df_depth = df_depth.sort_values(by='depth')

# #             # # Create a figure with 3 subplots
# #             # fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# #             # # Plot 1: x-axis = index (using df_filtered)
# #             # axes[0].plot(df_filtered.index, df_filtered['lower_bound'], label='Lower Bound', color='blue')
# #             # # axes[0].plot(df_filtered.index, df_filtered['upper_bound'], label='Upper Bound', color='red')
# #             # axes[0].set_xlabel('Index')
# #             # axes[0].set_ylabel('Bound Value')
# #             # axes[0].set_title('Bounds vs. Index (upper_bound < 1)')
# #             # axes[0].legend()

# #             # # Plot 2: x-axis = depth (using df_depth)
# #             # axes[1].plot(df_depth['depth'], df_depth['lower_bound'], label='Lower Bound', color='blue')
# #             # # axes[1].plot(df_depth['depth'], df_depth['upper_bound'], label='Upper Bound', color='red')
# #             # axes[1].set_xlabel('Depth')
# #             # axes[1].set_ylabel('Bound Value')
# #             # axes[1].set_title('Bounds vs. Depth (upper_bound < 1, one row per depth with highest index)')
# #             # axes[1].legend()

# #             # # Plot 3: x-axis = time (using df_filtered)
# #             # axes[2].plot(df_filtered['time'], df_filtered['lower_bound'], label='Lower Bound', color='blue')
# #             # # axes[2].plot(df_filtered['time'], df_filtered['upper_bound'], label='Upper Bound', color='red')
# #             # axes[2].set_xlabel('Time')
# #             # axes[2].set_ylabel('Bound Value')
# #             # axes[2].set_title('Bounds vs. Time (upper_bound < 1)')
# #             # axes[2].legend()

# #             # plt.tight_layout()
# #             # plt.show()

# #             for i, dataset in enumerate(datasets):
# #                 # print(f"Scenario {scenario}, solving instance {i}")
# #                 # Create gurobi model and solve to optimal
# #                 # gurobi_model = create_gurobi_model(n_planes=n_planes, dataset=dataset)
# #                 # gurobi_model.setParam('OutputFlag', 0)
# #                 # gurobi_model.optimize()
# #                 # global gurobi_sol
# #                 # gurobi_sol = gurobi_model.x
# #                 print(f"Scenario {scenario}, solving instance {i}")
# #                 start_time = time.time()
# #                 prob = create_problem(n_planes=n_planes, dataset=dataset)
# #                 solveprob(prob)
# #                 num_nodes.append(prob.attributes.nodes)
# #                 mip_bound.append(prob.attributes.bestbound)
# #                 solve_time.append(time.time() - start_time)
            
# #             print('Objective ', mip_bound)
# #             print("IPA Nodes ", num_nodes)
# #             print('IPA time ', solve_time)

# #             # Create the DataFrame for this scenario
# #             df = pd.DataFrame({
# #                 "Objective": mip_bound,
# #                 "IPA Nodes": num_nodes,
# #                 "IPA Time": solve_time
# #             })
# #             dfs.append(df)
# #             print(f"Results for Scenario {scenario}:")
# #             print(df)

# #         # Calculate averages
# #         average_nodes = [int(np.mean([dfs[0]["IPA Nodes"][i], dfs[1]["IPA Nodes"][i], dfs[2]["IPA Nodes"][i]])) for i in range(len(dfs[0]))]
# #         average_time = [round(np.mean([dfs[0]["IPA Time"][i], dfs[1]["IPA Time"][i], dfs[2]["IPA Time"][i]]), 3) for i in range(len(dfs[0]))]

# #         # Combine results into a single DataFrame for output
# #         combined_data = []
# #         for i in range(len(dfs[0])):
# #             combined_data.append([
# #                 i,  # Instance
# #                 dfs[0].loc[i, "Objective"], dfs[0].loc[i, "IPA Nodes"], dfs[0].loc[i, "IPA Time"],  # Scenario 1
# #                 dfs[1].loc[i, "Objective"], dfs[1].loc[i, "IPA Nodes"], dfs[1].loc[i, "IPA Time"],  # Scenario 2
# #                 dfs[2].loc[i, "Objective"], dfs[2].loc[i, "IPA Nodes"], dfs[2].loc[i, "IPA Time"],  # Scenario 3
# #                 average_nodes[i], average_time[i]  # Averages
# #             ])

# #         # Create a nicely formatted DataFrame
# #         formatted_df = pd.DataFrame(combined_data, columns=[
# #             "Instance",
# #             "Scenario 1 - Objective", "Scenario 1 - IPA Nodes", "Scenario 1 - IPA Time",
# #             "Scenario 2 - Objective", "Scenario 2 - IPA Nodes", "Scenario 2 - IPA Time",
# #             "Scenario 3 - Objective", "Scenario 3 - IPA Nodes", "Scenario 3 - IPA Time",
# #             "Average Nodes", "Average Time"
# #         ])

# #         # Generate output filename
# #         # output_filename = "results_Mangasarian_heuristic.xlsx"
# #         output_filename = f"results_Mangasarian_heuristic_{filename.split('.')[0].split('_', 1)[1]}.xlsx"  # e.g., results_10_2_3.xlsx
# #         formatted_df.to_excel(output_filename, index=False)
# #         print(f"Results saved to {output_filename}\n")