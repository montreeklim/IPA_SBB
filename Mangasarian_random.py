import xpress as xp
import numpy as np
from scipy.linalg import null_space
import itertools
import time

np.set_printoptions(precision=3, suppress=True)

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

# Function to check if rows are nearly identical
def are_rows_near_duplicates(row1, row2, tol=1e-6):
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

def compute_w_gamma_y(a, x, rows, cols, BigM):
    # Reshape the flat list into a 2D array
    matrix = [x[i * cols:(i + 1) * cols] for i in range(rows)]

    # Identify which x_ij is 1 for each j
    indices_per_j = {}
    for j in range(cols):
        indices_per_j[j] = [i + 1 for i in range(rows) if matrix[i][j] == 1]

    # Convert 1-based indices to 0-based and extract subarrays
    subarrays_per_j = {}
    for j, indices in indices_per_j.items():
        zero_based_indices = [idx - 1 for idx in indices]  # Convert to 0-based
        subarrays_per_j[j] = a[zero_based_indices]         # Extract rows from `a`

    B = {}
    w = []
    gamma = []

    for j, subarray in subarrays_per_j.items():
        n = subarray.shape[0]
        if n == 0:  # Handle empty subarrays
            continue

        I = np.eye(n)  # Identity matrix of size n x n
        e = np.ones((n, 1))  # Column vector of ones of size n x 1

        # Compute projection matrix P = I - e e^T / n
        P = I - (e @ e.T) / n

        # Compute the final value: A^T * P * A
        B[j] = subarray.T @ P @ subarray

        # Compute the eigenvalues and eigenvectors of B[j]
        eigenvalues, eigenvectors = np.linalg.eigh(B[j])

        # Smallest eigenvalue and its corresponding eigenvector
        smallest_eigenvalue_index = np.argmin(eigenvalues)
        w_j = eigenvectors[:, smallest_eigenvalue_index]
        gamma_j = (e.T @ subarray @ w_j) / n

        # Append results to lists
        w.extend(w_j)
        gamma.append(gamma_j[0])  # gamma_j is a 1-element array

    # Convert w and gamma back into per-j lists for computation
    w_per_j = [np.array(w[i * 2:(i + 1) * 2]) for i in range(cols)]
    gamma_per_j = gamma

    # Compute y[i] for all rows and columns
    y = np.zeros(rows)
    for i in range(rows):
        for j in range(cols):
            w_j = w_per_j[j]
            gamma_j = gamma_per_j[j]
            x_ij = matrix[i][j]  # Binary variable x_ij
            if x_ij == 0:
                continue
            term1 = w_j.T @ a[i] - gamma_j 
            term2 = -w_j.T @ a[i] + gamma_j 
            y[i] += max(0, term1, term2)

    return w, gamma, y
    
def create_problem(filename = None):
    """Read the given problem or create simple linear classifier problem."""

    # Create a new optimization problem
    prob = xp.problem()
    # Disable presolve
    # prob.controls.xslp_presolve = 0
    # prob.presolve()
    global M, N, K, a, BigM

    # Read the problem from a file
    M = 14
    N = 2
    K = 3
    a = np.array([[4.5, 7.1], [4.0, 6.9], [9.3, 8.7], [2.3, 7.1], [5.5, 6.9], [0.0, 1.0], [4.3, 8.5], [4.8, 7.8], [0.0, 0.0], [2.5, 7.1], [7.0, 8.4], [3.4, 7.0], [1.1, 7.0], [9.3, 7.7], [5.5, 7.8], [5.2, 8.5], [7.9, 8.3], [10.0, 10.0], [4.8, 8.2], [8.0, 8.7]])

    h = np.max(a)
    BigM = h*np.sqrt(N)
    gamma_bound = N*h + h*np.sqrt(N)
    
    # Create problem for k-hyperplane clustering without norm constraints
    # Create variables
    w = xp.vars(K, N, name='w', lb=-1, ub=1)
    gamma = xp.vars(K, name="gamma", lb=-gamma_bound, ub = gamma_bound)
    x = xp.vars(M, K, name='x', vartype=xp.binary)
    y = xp.vars(M, name='y', lb=0)
    prob.addVariable(w, gamma, x, y)
    
    # Add symmetry breaking on x
    for m in range(M):
        for k in range(K):
            if k > m:
                x[m, k].ub = 0 
        
    # Add constraints
    for i in range(M):
        prob.addConstraint(sum(x[i,j] for j in range(min(i+1,K))) == 1)
        for j in range(K):
            if j <= i:
                prob.addConstraint(y[i] >= np.dot(w[j], a[i]) - gamma[j] - BigM*(1-x[i,j]))
                prob.addConstraint(y[i] >= np.dot(-w[j], a[i]) + gamma[j] - BigM*(1-x[i,j]))
    
    # Add norm constraints
    for j in range(K):
        prob.addConstraint(sum(w[j, i] * w[j, i] for i in range(N)) <= 1)
    
    # set objective
    prob.setObjective(sum(y[i]*y[i] for i in range(M)), sense = xp.minimize)
    
    global all_variables, w_variables_idxs, gamma_variables_idxs, x_variables_idxs, y_variables_idxs, refuse_sol
    refuse_sol = []
    all_variables = prob.getVariable()
    w_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("w")]
    gamma_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("gamma")]
    x_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("x")]
    y_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("y")]
        
    return prob

def cbchecksol(prob, data, soltype, cutoff):
    # print('ENTER CBCHECKSOL PREINTSOL')
    """Callback function to reject the solution if it is not on the ball and accept otherwise."""
    try:
        global BigM, a, tol
        if (prob.attributes.presolvestate & 128) == 0:
            return (1, 0)
    
        sol = []

        # Retrieve node solution
        try:
            prob.getlpsol(x=sol)
        except:
            return (1, cutoff)

        w_sol = sol[min(w_variables_idxs): max(w_variables_idxs)+1]
        w_array = split_data(w_sol, K, N)
        non_zero_check = np.any(np.abs(w_array) > 1e-4, axis=1)
        all_non_zero = np.all(non_zero_check)
    
        if check_all_balls(w_array):
            refuse = 0
        elif all_non_zero:
            refuse = 1
            w_sol = split_data(sol[min(w_variables_idxs): max(w_variables_idxs) + 1], K, N)
            w_norm = np.linalg.norm(w_sol, axis = 1)
            x_sol = sol[min(x_variables_idxs): max(x_variables_idxs) + 1]
            # print('X = ', x_sol)
            new_w, new_gamma, new_y = compute_w_gamma_y(a, x_sol, M, K, BigM)
            # print('new W = ', new_w)

            # Ensure all variables are Python lists
            new_w = list(new_w) if isinstance(new_w, np.ndarray) else new_w
            new_gamma = list(new_gamma) if isinstance(new_gamma, np.ndarray) else new_gamma
            x_sol = list(x_sol) if isinstance(x_sol, np.ndarray) else x_sol
            new_y = list(new_y) if isinstance(new_y, np.ndarray) else new_y
        
            # new_gamma, new_w, new_y = compute_w_gamma_y(a, x_sol, M, K, BigM)
            new_sol = new_w + new_gamma + x_sol + new_y

            if min(w_norm) >= 1e-6:
                refuse_sol.append(new_sol)
        else:
            refuse = 1
    
        return (refuse, cutoff)
    
    except Exception as e:
        print('Exception in cbchecksol:', e)
        return (1, cutoff)

def prenode_callback(prob, data):
    global refuse_sol
        
    if len(refuse_sol) > 0:
        # print('There are some refused point to be added')
        for sol in refuse_sol:
            # add mip sol
            prob.addmipsol(sol)
            # reset issue_sol
        refuse_sol = []
    return 0



def cbbranch(prob, data, branch):
    # print('ENTER CHGBRANCG CALLBACK')
    """Callback function to create new branching object and add the corresponding constraints."""
    global initial_polytope
    global extreme_points  # Global variable to store constraints for each branch

    if prob.attributes.currentnode == 1:
        bo = xp.branchobj(prob, isoriginal=True)
        bo.addbranches((N + 1)**K)
        initial_polytope = create_n_simplex(N)
        extreme_points = {}
        a_coeff = {}
        submatrix = {}
        for i in range(N + 1):
            submatrix[i] = np.delete(initial_polytope, i, axis=0)
            extreme_points[i] = submatrix[i]
            coeff = up_extension_constraint(submatrix[i])
            for j in range(len(coeff)):
                dot_products = [np.dot(coeff[j], row) for row in submatrix[i]]
                if max(dot_products) < 1e-6:
                    coeff[j] = -coeff[j]
            a_coeff[i] = coeff
        values = range(N + 1)

        for combination in itertools.product(values, repeat=K):
            ball_idx = sum([combination[k] * ((N+1) ** (K - k - 1)) for k in range(K)])
            extreme_points[ball_idx] = {}
            for k in range(K):
                w_ball_idx = np.arange(k*N, (k+1)*N)
                for j in range(len(a_coeff[combination[k]])):
                    rhs_value = 1 if j == 0 else 0
                    bo.addrows(ball_idx, ['G'], [rhs_value], [0, N*K], w_ball_idx, a_coeff[combination[k]][j])
                extreme_points[ball_idx][k] = submatrix[combination[k]]
        bo.setpriority(100)
        return bo
    else:
        sol = []

        if (prob.attributes.presolvestate & 128) == 0:
            return branch

        # Retrieve node solution
        try:
            prob.getlpsol(x=sol)
        except:
            return branch
        
        w_sol = sol[min(w_variables_idxs): max(w_variables_idxs)+1]
        w_array = split_data(w_sol, K, N)
        split_index = split_data(w_variables_idxs, K, N)

        if check_all_balls(w_array):
            return branch
        
        # Find branching varaible with the most smallest norm
        # Choose a ball to branch on
        norms = np.linalg.norm(w_array, axis=-1)
        ball_idx = np.argmin(norms)
        w_ball_idx = list(split_index[ball_idx])
        
        try:
            max_dist = max(data[prob.attributes.currentnode]['distance'])
            if max_dist <= 1e-6:
                # Do not branch on too close node
                return branch
        except:
            pass
                
        # Toss a coin to branch on x with 95%
        if np.random.random() < 0.95:
            data[prob.attributes.currentnode]['branch'] = False
            return branch
        else:
            data[prob.attributes.currentnode]['branch'] = True
            
        pi_w_array = [ProjectOnBall(w_j) for w_j in w_array]
        initial_points = data[prob.attributes.currentnode][ball_idx]
        new_matrix = append_zeros_and_ones(initial_points)
        # This facet is not full rank (two points are too close)
        if np.linalg.matrix_rank(initial_points, tol=1e-6) < N or np.linalg.matrix_rank(new_matrix, tol=1e-6) != np.linalg.matrix_rank(initial_points, tol=1e-6):
            # print('The E matrix is not in full rank ', data[prob.attributes.currentnode])
            return branch
            
        # create new object with n empty branches
        bo = xp.branchobj(prob, isoriginal=True)
        bo.addbranches(N)
        for i in range(N):
            submatrix = np.delete(initial_points, i, axis=0)
            extreme_points = np.concatenate((submatrix, [pi_w_array[ball_idx]]), axis=0)
            try:
                a_coeff = up_extension_constraint(extreme_points)
            except:
                print('Cannot obtain coefficient for constraints')
                return branch
            for j in range(len(a_coeff)):
                if j == 0:
                    bo.addrows(i, ['G'], [1], [0, N*K], w_ball_idx, a_coeff[j])
                else:
                    dot_products = [np.dot(a_coeff[j], row) for row in extreme_points]
                    if max(dot_products) < 1e-6:
                        # Negative case; switch 
                        a_coeff[j] = - a_coeff[j]
                    bo.addrows(i, ['G'], [0], [0, N*K], w_ball_idx, a_coeff[j])
        return bo
    return branch


def cbnewnode(prob, data, parentnode, newnode, branch):
    # print('ENTER CBNEWNODE')
    """Callback function to add data of extreme points to each node. The data[node][ball_index] represents the matrix of extreme points with corresponding node and ball"""
    
    # Create empty dict
    data[newnode] = {}
    
    if (prob.attributes.presolvestate & 128) == 0:
        return 0
    
    # Store data for constraint of new node
    if parentnode == 1:
        data[newnode] = extreme_points[branch]
    else:
        # if parentnode is branch on x
        if not data[parentnode]['branch']:
            data[newnode] = data[parentnode]
            return 0
        
        # IPA branch on w
        sol = []

        # Retrieve node solution
        try:
            prob.getlpsol(x=sol)
        except:
            return 0

        w_sol = sol[min(w_variables_idxs): max(w_variables_idxs)+1]
        w_array = split_data(w_sol, K, N)
        norms = np.linalg.norm(w_array, axis=-1)
        ball_idx = np.argmin(norms)
        
        initial_polytope = data[parentnode][ball_idx]
        submatrix = np.delete(initial_polytope, branch, axis=0)
        pi_w = ProjectOnBall(w_array[ball_idx])
        for key in data[parentnode].keys():
            data[newnode][key] = data[parentnode][key]
        data[newnode][ball_idx] = np.vstack((submatrix, pi_w))
    
        # Calculate distance between newnode and parentnode
        distance = [np.linalg.norm(data[newnode][ball_idx][n] - data[parentnode][ball_idx][n]) for n in range(N)]
        data[newnode]['distance'] = distance
    
    return 0




def solveprob(prob):
    """Function to solve the problem with registered callback functions."""

    data = {}
    data[1] = {}
    prob.addcbpreintsol(cbchecksol, data, 1)
    prob.addcbchgbranchobject(cbbranch, data, 1)
    prob.addcbnewnode(cbnewnode, data, 1)
    prob.addcbprenode(prenode_callback, data, 1)
    prob.controls.outputlog = 0
    # prob.controls.presolve = 0
    # prob.controls.dualstrategy = 7
    # prob.controls.HEUREMPHASIS = 0

    # Adjust tolerances
    # prob.controls.feastol = 1e-8
    # prob.controls.optimalitytol = 1e-8
    # prob.controls.miptol = 1e-4
    # prob.controls.miprelstop = 1e-4      
    
    prob.controls.branchchoice = 1      # Continue on the child node with the best bound.
    prob.controls.backtrack = 2           # Use Breadth-first backtracking strategy
    prob.controls.backtracktie = 1        # In case of ties, select the Earliest created node
    
    prob.controls.timelimit=600
    # prob.controls.maxnode = 500
    
    prob.mipoptimize("")
    
    # print("Solution status:", prob.getProbStatusString())
    
if __name__ == '__main__':
    # np.random.seed(123)
    tol = 1e-4
    # prob = create_problem(filename = 'srncr_20_2_3_2023_ins.dat')
    start_time = time.time()
    prob = create_problem()
    solveprob(prob)
    num_nodes = prob.attributes.nodes
    print("Number of nodes in the tree:", num_nodes)
    mip_bound = prob.attributes.bestbound
    print(f"MIP bound from solver: {mip_bound:.3f}")
    solve_time = time.time()-start_time
    print(f"Solve time: {solve_time:.3f}")



