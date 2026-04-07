import xpress as xp
import numpy as np
from scipy.linalg import null_space, solve
import itertools
import time
import pickle
import pandas as pd
import re  # For extracting numbers from filenames

np.set_printoptions(precision=3, suppress=True)
xp.init('C:/Apps/Anaconda3/Lib/site-packages/xpress/license/community-xpauth.xpr') # license path for desktop
# xp.init('C:/Users/montr/anaconda3/Lib/site-packages/xpress/license/community-xpauth.xpr') # license path for laptop

def inverse_power_method(A, x0, tol=1e-6, max_iter=1000):
    """
    Uses the inverse power method to find the eigenvalue of A closest to zero
    (i.e., the smallest in magnitude) and its corresponding eigenvector.

    Parameters:
      A       : numpy.ndarray, the input square matrix (assumed invertible)
      x0      : numpy.ndarray, an initial guess vector (nonzero)
      tol     : float, tolerance for convergence (default 1e-6)
      max_iter: int, maximum number of iterations (default 1000)

    Returns:
      eigenvalue : float, the approximated eigenvalue of A
      eigenvector: numpy.ndarray, the corresponding eigenvector (normalized)
      iterations : int, the number of iterations performed
    """
    # Normalize the initial guess
    x = x0 / np.linalg.norm(x0)
    eigenvalue_old = 0.0

    for i in range(max_iter):
        # Solve A * y = x for y
        y = np.linalg.solve(A, x)
        # Normalize the new vector
        x = y / np.linalg.norm(y)
        # Compute the Rayleigh quotient to approximate the eigenvalue
        eigenvalue = x.T @ A @ x
        # Check for convergence (change in eigenvalue below tolerance)
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
    
# def compute_w_gamma_y(a, x, w_old, rows, cols, BigM):
#     '''
#     A function to recalculate optimal solution of hyperplane clustering given an integer solution x

#     Parameters
#     ----------
#     a : list, array
#         points to be classified
#     x : list, array
#         x_ij = 1 iff point a[i] is in cluster j
#     w_old : array of w
#     rows : int
#         number of rows
#     cols : int
#         number of columns
#     BigM : float
#         Big M value that sufficiently large

#     Returns
#     -------
#     w : list
#         optimal value of w
#     gamma : list
#         optimal value of gamma
#     y : list
#         optimal value of y

#     '''
#     # Reshape the flat list into a 2D array
#     matrix = [x[i * cols:(i + 1) * cols] for i in range(rows)]
#     # print(matrix)

#     # Identify which x_ij is 1 for each j
#     indices_per_j = {}
#     for j in range(cols):
#         indices_per_j[j] = [i + 1 for i in range(rows) if matrix[i][j] == 1]

#     # Convert 1-based indices to 0-based and extract subarrays
#     subarrays_per_j = {}
#     for j, indices in indices_per_j.items():
#         zero_based_indices = [idx - 1 for idx in indices]  # Convert to 0-based
#         subarrays_per_j[j] = a[zero_based_indices]         # Extract rows from `a`

#     B = {}
#     w = []
#     gamma = []

#     for j, subarray in subarrays_per_j.items():
#         n = subarray.shape[0]
#         if n == 0:  # Handle empty subarrays
#             continue

#         I = np.eye(n)  # Identity matrix of size n x n
#         e = np.ones((n, 1))  # Column vector of ones of size n x 1

#         # Compute projection matrix P = I - e e^T / n
#         P = I - (e @ e.T) / n

#         # Compute the final value: A^T * P * A
#         B[j] = subarray.T @ P @ subarray
#         # print(B[j].shape)

#         # Compute the eigenvalues and eigenvectors of B[j]
#         # eigenvalues, eigenvectors = np.linalg.eigh(B[j])

#         # Smallest eigenvalue and its corresponding eigenvector
#         # smallest_eigenvalue_index = np.argmin(eigenvalues)
#         # w_j = eigenvectors[:, smallest_eigenvalue_index]
#         # q = w_old[j]
#         # print('Warm start point = ', q)
        
#         # q = np.random.rand(B[j].shape[0])
#         # print('Random q = ', q)
        
#         eigenvalue, w_j = inverse_power_method(B[j], w_old[j], tol=1e-3, max_iter=100)
        
        
#         gamma_j = (e.T @ subarray @ w_j) / n
#         # Append results to lists
#         w.extend(w_j)
#         gamma.append(gamma_j[0])  # gamma_j is a 1-element array

#     # Convert w and gamma back into per-j lists for computation
#     w_per_j = [np.array(w[i * N:(i + 1) * N]) for i in range(cols)]
#     gamma_per_j = gamma


#     # Compute y[i] for all rows and columns
#     y = np.zeros(rows)
#     for i in range(rows):
#         for j in range(cols):
#             w_j = w_per_j[j]
#             gamma_j = gamma_per_j[j]
#             x_ij = matrix[i][j]  # Binary variable x_ij
#             if x_ij == 0:
#                 continue
#             # term1 = (w_j.T @ a[i] - gamma_j)[0]
#             # term2 = (-w_j.T @ a[i] + gamma_j)[0]
#             # y[i] += max(0, term1, term2)

#             term1 = w_j.T @ a[i] - gamma_j 
#             term2 = -w_j.T @ a[i] + gamma_j 
#             y[i] += max(0, term1, term2)
            
#     return w, gamma, y

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
        eigenvalue, w_j = inverse_power_method(B_j, w_old[j], tol=1e-3, max_iter=100)

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

    global all_variables, w_variables_idxs, gamma_variables_idxs, x_variables_idxs, y_variables_idxs, refuse_sol, x_vars
    refuse_sol = []
    all_variables = prob.getVariable()
    w_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("w")]
    gamma_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("gamma")]
    x_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("x")]
    # x_vars = [var for ind, var in enumerate(all_variables) if var.name.startswith("x")]
    y_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("y")]
    # print('W indices = ', w_variables_idxs)
    # print('Gamma indices = ', gamma_variables_idxs)

    return prob

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
            # Recompute w, gamma, y based on the current x_sol
            # x_sol = sol[min(x_variables_idxs): max(x_variables_idxs) + 1]
            # new_w, new_gamma, new_y = compute_w_gamma_y(a, x_sol, M, K, BigM)

            # # Convert to lists if needed
            # new_w = list(np.array(new_w).flatten()) if isinstance(new_w, np.ndarray) else new_w
            # new_gamma = list(new_gamma) if isinstance(new_gamma, np.ndarray) else new_gamma
            # x_sol = list(x_sol) if isinstance(x_sol, np.ndarray) else x_sol
            # new_y = list(new_y) if isinstance(new_y, np.ndarray) else new_y

            # new_sol = new_w + new_gamma + x_sol + new_y

            # w_norm = np.linalg.norm(w_array, axis=1)
            # if min(w_norm) >= 1e-6:
            #     refuse_sol.append(new_sol)
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
            prob.addmipsol(sol)
        refuse_sol = []
    
    return 0

def cbbranch(prob, data, branch):
    global initial_polytope, extreme_points
    currentnode = prob.attributes.currentnode

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
        if rng_branch.random() < max(mip_gap, 1 - mip_gap):
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


# def cbbranch(prob, data, branch):
#     # print('Enter ChangeBranch callback')
#     global initial_polytope
#     global extreme_points

#     if prob.attributes.currentnode == 1:
#         # print('We are in the root node')
#         bo = xp.branchobj(prob, isoriginal=True)
#         bo.addbranches((N + 1)**K)
#         initial_polytope = create_n_simplex(N)
#         extreme_points = {}
#         a_coeff = {}
#         submatrix = {}
#         # print('Variables created')
#         for i in range(N + 1):
#             submatrix[i] = np.delete(initial_polytope, i, axis=0)
#             extreme_points[i] = submatrix[i]
#             coeff = up_extension_constraint(submatrix[i])
#             for j in range(len(coeff)):
#                 dot_products = [np.dot(coeff[j], row) for row in submatrix[i]]
#                 if max(dot_products) < 1e-6:
#                     coeff[j] = -coeff[j]
#             a_coeff[i] = coeff
#         values = range(N + 1)
#         # print('Submatrix done')

#         for combination in itertools.product(values, repeat=K):
#             ball_idx = sum([combination[k] * ((N+1) ** (K - k - 1)) for k in range(K)])
#             extreme_points[ball_idx] = {}
#             for k in range(K):
#                 w_ball_idx = np.arange(k*N, (k+1)*N)
#                 for j in range(len(a_coeff[combination[k]])):
#                     rhs_value = 1 if j == 0 else 0
#                     bo.addrows(ball_idx, ['G'], [rhs_value], [0, N*K], w_ball_idx, a_coeff[combination[k]][j])
#                 extreme_points[ball_idx][k] = submatrix[combination[k]]
#         bo.setpriority(100)
#         # print('The branching object at the root node is created')
#         return bo
#     else:
#         # print('In the else of changebranch callback')
#         sol = []

#         if (prob.attributes.presolvestate & 128) == 0:
#             return branch

#         # Retrieve node solution
#         try:
#             sol = prob.getCallbackSolution(prob.getVariable())
#             # print('The relaxation solution is ', sol)
#         except:
#             return branch

#         w_sol = sol[min(w_variables_idxs): max(w_variables_idxs)+1]
#         w_array = split_data(w_sol, K, N)
#         split_index = split_data(w_variables_idxs, K, N)
#         # print('W array = ', w_array)
#         # print(check_all_balls(w_array))

#         if check_all_balls(w_array):
#             # print('The solution is on ball ', w_array)
#             return branch
#         # print('After check all balls')

#         norms = np.linalg.norm(w_array, axis=-1)
#         # print('norm = ', norms)
#         ball_idx = np.argmin(norms)
#         # print('ball_idx = ', ball_idx)
#         w_ball_idx = list(split_index[ball_idx])
#         # print('w ball idx = ', w_ball_idx)
        
#         # Store necessary info in data:
#         data[prob.attributes.currentnode]['w_array'] = w_array
#         data[prob.attributes.currentnode]['ball_idx'] = ball_idx

#         try:
#             max_dist = max(data[prob.attributes.currentnode]['distance'])
#             if max_dist <= 1e-6:
#                 return branch
#         except:
#             pass

#         dual_bound = prob.getAttrib("bestbound")
#         # print('dual_bound = ', dual_bound)
#         if dual_bound <= tol:
#             data[prob.attributes.currentnode]['branch'] = False
#             return branch
#         else:
#             # print('Before get mipobjval')
#             best_solution = prob.getAttrib("mipobjval")
#             # print('best_solution = ', best_solution)
#             mip_gap = abs((best_solution - dual_bound) / best_solution)

#         # rand = np.random.random()
#         # toss a coin to branch on x or w
#         if rng_branch.random() < max(mip_gap, 1-mip_gap):
#             data[prob.attributes.currentnode]['branch'] = False
#             return branch
#         else:
#             data[prob.attributes.currentnode]['branch'] = True

#         pi_w_array = [ProjectOnBall(w_j) for w_j in w_array]
#         initial_points = data[prob.attributes.currentnode][ball_idx]
#         new_matrix = append_zeros_and_ones(initial_points)

#         if np.linalg.matrix_rank(initial_points, tol=1e-6) < N or np.linalg.matrix_rank(new_matrix, tol=1e-6) != np.linalg.matrix_rank(initial_points, tol=1e-6):
#             return branch

#         bo = xp.branchobj(prob, isoriginal=True)
#         bo.addbranches(N)
#         for i in range(N):
#             submatrix = np.delete(initial_points, i, axis=0)
#             extreme_points = np.concatenate((submatrix, [pi_w_array[ball_idx]]), axis=0)
#             try:
#                 a_coeff = up_extension_constraint(extreme_points)
#             except:
#                 return branch
#             for j in range(len(a_coeff)):
#                 if j == 0:
#                     bo.addrows(i, ['G'], [1], [0, N*K], w_ball_idx, a_coeff[j])
#                 else:
#                     dot_products = [np.dot(a_coeff[j], row) for row in extreme_points]
#                     if max(dot_products) < 1e-6:
#                         a_coeff[j] = - a_coeff[j]
#                     bo.addrows(i, ['G'], [0], [0, N*K], w_ball_idx, a_coeff[j])
#         # print('User branching object spatial branching on w')
#         return bo


#     return branch

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
        
        # print('New node callback: Brnaching on W')
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

def cbnodelpsolved(prob, data):
    # Only execute if current node and heuristic check pass.
    if prob.attributes.currentnode >= (N + 1) ** (K + 1) and rng_heuristic.random() < 1:
        try:
            # print(data[prob.attributes.currentnode]['w_array'])
            # Get the current callback solution and extract the x portion.
            sol = prob.getCallbackSolution(prob.getVariable())
            x_sol = sol[min(x_variables_idxs): max(x_variables_idxs) + 1]
            x_sol = split_data(x_sol, M, K)

            # Create a binary mask and determine cluster assignments.
            mask = (x_sol == 1)
            result = np.where(mask.any(axis=1), np.argmax(mask, axis=1), -1)

            # Preallocate arrays for the hyperplane parameters:
            # w is a KxN matrix; gamma is a length-K vector.
            w = np.empty((K, N))
            gamma = np.empty(K)

            # Loop over each hyperplane (cluster) and compute its parameters.
            for i in range(K):
                mask_i = (result == i)
                sub_a = a[mask_i]
                n = sub_a.shape[0]
                if n == 0:
                    return 0

                # Instead of explicitly forming an identity matrix and ones vector each time,
                # we create e and use np.eye(n) to form the projection matrix.
                e = np.ones((n, 1))
                P = np.eye(n) - (e @ e.T) / n
                B = sub_a.T @ P @ sub_a

                # Use the provided initial vector for this hyperplane.
                q = data[prob.attributes.currentnode]['w_array'][i]
                eigenvalue, w_j = inverse_power_method(B, q, tol=1e-3, max_iter=100)

                # Compute gamma_j = (e.T @ sub_a @ w_j) / n.
                # Since e.T @ sub_a sums the rows of sub_a, this equals:
                gamma_j = np.sum(sub_a @ w_j) / n

                w[i, :] = w_j  # Store as the i-th row.
                gamma[i] = gamma_j

            # Process the unassigned points (those with result < 0).
            mask_unassigned = result < 0
            unassign_points = a[mask_unassigned]
            previous_assignment = np.zeros(unassign_points.shape[0])
            max_iter = 20

            # Iterate to refine hyperplane parameters by reassigning unassigned points.
            for iteration in range(max_iter):
                # Make a copy of w to track changes.
                w_old = w.copy()

                # Calculate distances from each unassigned point to each hyperplane.
                dot_products = unassign_points @ w.T
                # Broadcasting: subtract gamma (of shape (K,)) from each column.
                distances = np.abs(dot_products - gamma)
                assignments = np.argmin(distances, axis=1)
                result[mask_unassigned] = assignments

                # Reconstruct x as an M x K binary matrix.
                x = np.zeros((M, K), dtype=int)
                x[np.arange(M), result] = 1
                x_flat = x.flatten()

                # Update the hyperplanes (and obtain the error vector y) using an external function.
                w, gamma, y = compute_w_gamma_y(a, x_flat, w_old, M, K, BigM)
                w = np.array(w).reshape(K, N)
                gamma = np.array(gamma)

                # If the change in assignments is small, we have converged.
                if np.linalg.norm(previous_assignment - assignments) < tol:
                    break
                previous_assignment = assignments.copy()
            
            # Update data
            # data[prob.attributes.currentnode]['w_array'] = w
            

            # Concatenate all parts into a new solution.
            new_sol = np.concatenate([w.flatten(), gamma.flatten(), x_flat, np.array(y).flatten()])
            refuse_sol.append(new_sol.tolist())

            return 0
        except Exception:
            return 0


# def cbnodelpsolved(prob, data):
#     # rng_heuristic.random() < 0 === Mangasarian
#     # rng_heuristic.random() < 0.1 === 10% calculate new sol
#     # rng_heuristic.random() < 1 === always
#     if prob.attributes.currentnode >= (N+1)**(K+1) and rng_heuristic.random() < 1:
#         try:
#             # print('Data at current node is ', data[prob.attributes.currentnode])
#             # print('ENTER TRY')
#             sol = prob.getCallbackSolution(prob.getVariable())
#             x_sol = sol[min(x_variables_idxs): max(x_variables_idxs)+1]
#             x_sol = split_data(x_sol, M, K)
#             # print('x = ', x_sol)
#             # print('Obtain x_sol')
            
#             # store x sol and check if it is in data
#             # if x_sol in data['x_sol']:
#             #     return 0
#             # else:
#             #     data[x_sol].append(x_sol)
#             # print('Obtain x_sol')
        
#             # to find which hyperplane point i belong to
#             # return -1 if unassigned
#             mask = x_sol == 1
#             # print('Mask = ', mask)
#             result = np.where(mask.any(axis=1), np.argmax(mask, axis=1), -1)
#             # print('Obtain result')
#             # print('Result = ', result)
        
#             # iterate 0 to K-1 to create hyperplane for each assigned points 
#             w = []
#             gamma = []
#             for i in range(K):
#                 # print('i = ', i)
#                 mask = result == i
#                 # print(mask)
#                 # sub_x = x_sol[mask]
#                 sub_a = a[mask]
#                 # find a hyperplane
#                 n = sub_a.shape[0]
#                 if n == 0:
#                     return 0
#                 I = np.eye(n)  # Identity matrix of size n x n
#                 e = np.ones((n, 1))  # Column vector of ones of size n x 1
#                 # print('Code works here')

#                 # Compute projection matrix P = I - e e^T / n
#                 P = I - (e @ e.T) / n
#                 # Compute the final value: A^T * P * A
#                 B = sub_a.T @ P @ sub_a
                
#                 q = data[prob.attributes.currentnode]['w_array'][i]
#                 # print('Initial vector = ', q)
#                 # q = np.random.rand(B.shape[0])
#                 # print('Random vector = ', q)
#                 eigenvalue, w_j = inverse_power_method(B, q, tol=1e-3, max_iter=100)
#                 # print('Code works here')
#                 # Compute the eigenvalues and eigenvectors of B[j]
#                 # eigenvalues, eigenvectors = np.linalg.eigh(B)
                
#                 # # Smallest eigenvalue and its corresponding eigenvector
#                 # smallest_eigenvalue_index = np.argmin(eigenvalues)
#                 # # print('Code works here')
#                 # w_j = eigenvectors[:, smallest_eigenvalue_index]
                
                
#                 gamma_j = (e.T @ sub_a @ w_j) / n
#                 # print(w_j)
#                 # print(gamma_j)
#                 w.extend(w_j)
#                 gamma.append(gamma_j[0])  # gamma_j is a 1-element array
#                 # print('Code works here')
#                 # print(w)
            
#             # print('Outside the for loop')
            
#             # print(w)
#             w = np.array(w).reshape(K, N) # change to N K if not correct
#             # print('Code works here')
#             gamma = np.array(gamma)
#             # print('Code works here')
            
#             # Now we have initial hyperplanes
#             # assign unassigned points
#             mask = result < 0
#             unassign_points = a[mask]
#             # print('Calculate unassigned points')
#             previous_assignment = np.zeros(unassign_points.shape[0])
            
            
#             max_iter = 20
#             for iteration in range(max_iter):
#                 w_old = w
#                 dot_products = np.dot(unassign_points, w.T)
#                 distances = np.abs(dot_products - gamma)  # Shape (M, K)
#                 assignments = np.argmin(distances, axis=1)
#                 result[mask] = assignments
        
#                 # reconstruct x
#                 x = np.zeros((M, K), dtype=int)
#                 rows = np.arange(M)
#                 x[rows, result] = 1
#                 x = x.flatten()

#                 # recompute hyperplane
#                 w, gamma, y = compute_w_gamma_y(a, x, w_old, M, K, BigM)

#                 if np.linalg.norm(previous_assignment - assignments) < tol:
#                     # print('Optimal hyperplane with distances = ', sum([ele**2 for ele in y]))
#                     break
#                 else:
#                     # print('Update hyperplane with distances = ', sum([ele**2 for ele in y]))
#                     previous_assignment = assignments.copy()
#                     # print('Update hyperplane with distances = ', sum([ele**2 for ele in y]))
                
#                 w = np.array(w).reshape(K, N)
#                 gamma = np.array(gamma)
#             # if it reach max_iter we need to convert w into list
#             # if iteration == max_iter-1:
#             # w = w.flatten().tolist()
#             # gamma = [float(val[0]) if isinstance(val, np.ndarray) else float(val) for val in gamma]

#             # gamma = gamma.tolist()
#             # print('Code works here')
#             # print('W out of loop ', w)
            
#             # add the final mipsol
#             new_sol = list(w) + list(gamma) + list(x) + list(y)
#             # print('New sol = ', new_sol)
#             # print('Length = ', len(new_sol))
#             # print('GET NEW SOL')
#             refuse_sol.append(new_sol)
            
#             return 0
#         except:
#             return 0

def solveprob(prob):
    data = {}
    data[1] = {}
    data['x_sol']= []
    prob.addcbpreintsol(cbchecksol, data, 2)
    prob.addcbchgbranchobject(cbbranch, data, 2)
    prob.addcbnewnode(cbnewnode, data, 2)
    prob.addcbprenode(prenode_callback, data, 1)
    prob.addcbnodelpsolved(cbnodelpsolved, data, 2)
    prob.controls.outputlog = 0
    # prob.controls.presolve = 0
    prob.controls.branchchoice = 1
    prob.controls.backtrack = 2
    prob.controls.backtracktie = 1
    prob.controls.timelimit = 900
    prob.controls.randomseed = 42
    prob.controls.deterministic = 1
    # prob.controls.threads = 1
    # prob.controls.maxnode = 500

    prob.mipoptimize("")
    
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

# if __name__ == '__main__':
#     tol = 1e-4
#     # List of dataset filenames
#     datasets_filenames = [
#         "HighDim.pkl",
#     ]

#     for filename in datasets_filenames:
#         print(f"Processing dataset file: {filename}")

#         # Load the dataset
#         with open(filename, "rb") as f:
#             datasets_dict = pickle.load(f)
#         print(f"Loaded {len(datasets_dict)} datasets from '{filename}'.\n")
#         # count = 0
#         combined_data = []
#         # Iterate over each (m, n, k) tuple and its corresponding data
#         for key_tuple, data_array in datasets_dict.items():
#             # if count >= 3:
#             #     break
#             # count += 1
#             m, n, k = key_tuple
#             print(f"Processing dataset with parameters (m, n, k) = ({m}, {n}, {k})")

#             # Initialize lists to store DataFrames for each scenario
#             dfs = []

#             # Run experiments for three different configurations/scenarios
#             for scenario in range(3):  # Adjust the number of scenarios if needed
#                 print(f"--- Scenario {scenario + 1} ---")
#                 np.random.seed(scenario)  # Ensure reproducibility per scenario
#                 num_nodes = []
#                 mip_bound = []
#                 solve_time = []

#                 start_time = time.time()

#                 # Create and solve the problem for the current scenario
#                 prob = create_problem(n_planes=k, dataset=data_array)
#                 solveprob(prob)

#                 nodes = prob.attributes.nodes
#                 bestbound = prob.attributes.bestbound
#                 solve_time = round(time.time() - start_time, 3)
#                 num_nodes.append(nodes)
#                 mip_bound.append(bestbound)

#                 # Create the DataFrame for this scenario
#                 df = pd.DataFrame({
#                     "Objective": mip_bound,
#                     "Nodes": num_nodes,      # Renamed from "IPA Nodes" to "Nodes"
#                     "Time": solve_time       # Renamed from "IPA Time" to "Time"
#                 })
#                 dfs.append(df)
#                 print(f"Results for Scenario {scenario + 1}:\n{df}\n")

#             # Initialize lists to store default experiment results
#             # default_objective = []
#             # default_nodes = []
#             # default_time = []

#             # start_time = time.time()

#             # # Create and solve the default problem
#             # prob_default = create_problem_defaults(n_planes=k, dataset=data_array)
#             # prob_default.controls.timelimit = 900 
#             # prob_default.optimize('x')

#             # default_bestbound = prob_default.attributes.bestbound
#             # default_nodes_count = prob_default.attributes.nodes
#             # default_elapsed_time = round(time.time() - start_time, 3)
#             # default_objective.append(default_bestbound)
#             # default_nodes.append(default_nodes_count)
#             # default_time.append(default_elapsed_time)

#             # Calculate averages across scenarios
#             average_nodes = []
#             average_time = []
#             for i in range(len(dfs[0])):
#                 nodes_values = [
#                     dfs[0]["Nodes"][i] if dfs[0]["Nodes"][i] is not None else np.nan,
#                     dfs[1]["Nodes"][i] if dfs[1]["Nodes"][i] is not None else np.nan,
#                     dfs[2]["Nodes"][i] if dfs[2]["Nodes"][i] is not None else np.nan
#                 ]
#                 times_values = [
#                     dfs[0]["Time"][i] if dfs[0]["Time"][i] is not None else np.nan,
#                     dfs[1]["Time"][i] if dfs[1]["Time"][i] is not None else np.nan,
#                     dfs[2]["Time"][i] if dfs[2]["Time"][i] is not None else np.nan
#                 ]

#                 # Compute mean, ignoring NaN
#                 avg_nodes = int(np.nanmean(nodes_values)) if not np.all(np.isnan(nodes_values)) else None
#                 avg_time = round(np.nanmean(times_values), 3) if not np.all(np.isnan(times_values)) else None

#                 average_nodes.append(avg_nodes)
#                 average_time.append(avg_time)

#             # Combine results into a single DataFrame for output
#             for i in range(len(dfs[0])):
#                 combined_data.append([
#                     m,  # m parameter
#                     n,  # n parameter
#                     k,  # k parameter
#                     # Scenario 1
#                     dfs[0].loc[i, "Objective"], dfs[0].loc[i, "Nodes"], dfs[0].loc[i, "Time"],
#                     # Scenario 2
#                     dfs[1].loc[i, "Objective"], dfs[1].loc[i, "Nodes"], dfs[1].loc[i, "Time"],
#                     # Scenario 3
#                     dfs[2].loc[i, "Objective"], dfs[2].loc[i, "Nodes"], dfs[2].loc[i, "Time"],
#                     # Averages
#                     average_nodes[i], average_time[i]
#                     # Default Experiment
#                     # default_objective[i], default_nodes[i], default_time[i]
#                 ])
            


#         # Create a nicely formatted DataFrame without the "Instance" column
#         formatted_df = pd.DataFrame(combined_data, columns=[
#             "m", "n", "k",  # Dataset parameters
#             # Scenario 1
#             "1 - Objective", "1 - Nodes", "1 - Time",
#             # Scenario 2
#             "2 - Objective", "2 - Nodes", "2 - Time",
#             # Scenario 3
#             "3 - Objective", "3 - Nodes", "3 - Time",
#             # Averages
#             "Average Nodes", "Average Time"
#             # Default Experiment
#             # "Default - Objective", "Default - Nodes", "Default - Time"
#         ])

#         # Generate output filename based on (m, n, k)
#         output_filename = "results_HighDim.xlsx"
#         # output_filename = "results_high_dim.xlsx"
#         formatted_df.to_excel(output_filename, index=False)
#         print(f"Results saved to '{output_filename}'.\n")


if __name__ == '__main__':
    tol = 1e-4
    # List of dataset filenames
    datasets_filenames = [
        # "instances_10_2_2.pkl",
        "instances_10_2_3.pkl",
        # "instances_12_2_3.pkl",
        # "instances_14_2_3.pkl"
        # "instances_12_3_2.pkl"
        ]

    for filename in datasets_filenames:
        print(f"Processing dataset: {filename}")

        # Load the dataset
        with open(filename, "rb") as f:
            datasets = pickle.load(f)

        # Extract n_planes from the filename
        n_planes = get_n_planes_from_filename(filename)
        print(f"Using n_planes = {n_planes}")

        # Initialize lists to store DataFrames for each scenario
        dfs = []

        # Run experiments for three different configurations/scenarios
        for scenario in range(1):  # Adjust the number of scenarios if needed
            # np.random.seed(scenario)
            global rng_branch, rng_heuristic
            rng_branch = np.random.default_rng(seed=42 + scenario)
            rng_heuristic = np.random.default_rng(seed=123 + scenario)
            num_nodes = []
            mip_bound = []
            solve_time = []
            start_time = time.time()
            
            # Test one instance
            dataset = datasets[0]
            prob = create_problem(n_planes=n_planes, dataset=dataset)
            solveprob(prob)
            print(prob.attributes.nodes)
            print(prob.attributes.bestbound)
            print(time.time() - start_time)

        #     for i, dataset in enumerate(datasets):
        #         print(f"Scenario {scenario}, solving instance {i}")
        #         start_time = time.time()
        #         prob = create_problem(n_planes=n_planes, dataset=dataset)
        #         solveprob(prob)
        #         num_nodes.append(prob.attributes.nodes)
        #         mip_bound.append(prob.attributes.bestbound)
        #         solve_time.append(time.time() - start_time)
            
        #     # print('Objective ', mip_bound)
        #     # print("IPA Nodes ", num_nodes)
        #     # print('IPA time ', solve_time)

        #     # Create the DataFrame for this scenario
        #     df = pd.DataFrame({
        #         "Objective": mip_bound,
        #         "IPA Nodes": num_nodes,
        #         "IPA Time": solve_time
        #     })
        #     dfs.append(df)
        #     print(f"Results for Scenario {scenario}:")
        #     print(df)

        # # Calculate averages
        # average_nodes = [int(np.mean([dfs[0]["IPA Nodes"][i], dfs[1]["IPA Nodes"][i], dfs[2]["IPA Nodes"][i]])) for i in range(len(dfs[0]))]
        # average_time = [round(np.mean([dfs[0]["IPA Time"][i], dfs[1]["IPA Time"][i], dfs[2]["IPA Time"][i]]), 3) for i in range(len(dfs[0]))]

        # # Combine results into a single DataFrame for output
        # combined_data = []
        # for i in range(len(dfs[0])):
        #     combined_data.append([
        #         i,  # Instance
        #         dfs[0].loc[i, "Objective"], dfs[0].loc[i, "IPA Nodes"], dfs[0].loc[i, "IPA Time"],  # Scenario 1
        #         dfs[1].loc[i, "Objective"], dfs[1].loc[i, "IPA Nodes"], dfs[1].loc[i, "IPA Time"],  # Scenario 2
        #         dfs[2].loc[i, "Objective"], dfs[2].loc[i, "IPA Nodes"], dfs[2].loc[i, "IPA Time"],  # Scenario 3
        #         average_nodes[i], average_time[i]  # Averages
        #     ])

        # # Create a nicely formatted DataFrame
        # formatted_df = pd.DataFrame(combined_data, columns=[
        #     "Instance",
        #     "Scenario 1 - Objective", "Scenario 1 - IPA Nodes", "Scenario 1 - IPA Time",
        #     "Scenario 2 - Objective", "Scenario 2 - IPA Nodes", "Scenario 2 - IPA Time",
        #     "Scenario 3 - Objective", "Scenario 3 - IPA Nodes", "Scenario 3 - IPA Time",
        #     "Average Nodes", "Average Time"
        # ])

        # # Generate output filename
        # # output_filename = "results_Mangasarian_heuristic.xlsx"
        # output_filename = f"results_Mangasarian_heuristic_{filename.split('.')[0].split('_', 1)[1]}.xlsx"  # e.g., results_10_2_3.xlsx
        # formatted_df.to_excel(output_filename, index=False)
        # print(f"Results saved to {output_filename}\n")