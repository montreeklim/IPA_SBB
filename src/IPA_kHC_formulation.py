import xpress as xp
import numpy as np
from scipy.linalg import null_space

np.set_printoptions(precision=3, suppress=True)
xp.init('C:/Apps/Anaconda3/lib/site-packages/xpress/license/community-xpauth.xpr')
# xp.init('C:/Users/montr/anaconda3/Lib/site-packages/xpress/license/community-xpauth.xpr') # License path for Laptop

def gram_schmidt(A):
    """Orthogonalize a set of vectors stored as the columns of matrix A."""
    # Compute the QR decomposition of A
    Q, _ = np.linalg.qr(A)
    # Q, _ = np.linalg.qr(A, mode='reduced', rcond=1e-10)
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

    # Calculate norms for each element
    norms = np.linalg.norm(w_array, axis=-1)

    # Check if any norm in each row is non-zero
    at_least_one_non_zero = np.any(norms < 1e-6)

    return at_least_one_non_zero

# This function should be used in a callback 
# H-version of facet relaxation
# verices_matrix = matrix of extreme points

def up_extension_constraint(vertices_matrix):
    """Create a numpy array contains coefficient for adding constraints in new nodes based on matrix of extreme points."""
    # Convert the vertices_matrix to a numpy array
    E = np.array(vertices_matrix)
    # Check condition number of new_matrix
    cond_number = np.linalg.cond(E)
    e = np.ones(len(E))
    if cond_number >= 1e+4:
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
    
    return a_coeff


def CheckOnBall(w):
    """Check if the obtained solution norm is almost one or not."""
    # tol = 1e-6 if w close to pi(w) accept the 
    if np.abs(np.linalg.norm(w) - 1) < tol:
        return True
    else:
        return False

def check_all_balls(w_array):
    """Check if the obtained solution norm is almost one or not."""
    # tol = 1e-6 if w close to pi(w) accept the solution
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

def read_dat_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    M = N = K = 0
    data_list = []
    data_section = False
    
    # Parse the lines
    for line in lines:
        line = line.strip()
        if line.startswith("set M :="):
            M_values = line[len("set M :="):].strip().split()
            M_values = [v for v in M_values if v != ';']
            M = max(map(int, M_values))
        elif line.startswith("set N :="):
            N_values = line[len("set N :="):].strip().split()
            N_values = [v for v in N_values if v != ';']
            N = max(map(int, N_values))
        elif line.startswith("set K :="):
            K_values = line[len("set K :="):].strip().split()
            K_values = [v for v in K_values if v != ';']
            K = max(map(int, K_values))
        elif line.startswith("1 2:="):
            data_section = True
            continue
        elif data_section:
            if line == ';':
                break
            # Correct parsing of data lines
            # print(line)
            parts = line.split()
            # Skip the first part which is the index
            data_row = list(map(float, parts[1:]))
            data_list.append(data_row)
    
    # Convert data_list to numpy array with the correct shape
    data_array = np.array(data_list).reshape(M, N)
    
    return M, N, K, data_array
    
def create_problem(filename = None):
    """Read the given problem or create simple linear classifier problem."""

    # Create a new optimization problem
    prob = xp.problem()
    # Disable presolve
    prob.controls.xslp_presolve = 0
    prob.presolve()

    # Read the problem from a file
    if filename != None:
        # Read dat file to get M, N, K, a
        global M, N, K, a
        # M = #points
        # N = #dimensions
        # K = #hyperplane
        M, N, K, a = read_dat_file(filename)
    else:
        M = 20
        N = 4
        K = 2
        a = np.array([[ 4.8,  4. ,  5. ,  3.9],
       [ 4. ,  4.8, 10. ,  7.1],
       [ 6.9,  9.5,  5.6,  2.6],
       [ 3.6,  4.4,  9.9,  8.9],
       [ 2.7,  0.7,  1.8,  5.1],
       [ 7.6,  2.7,  5.1,  3.8],
       [ 1.8,  0. ,  6.4,  6.8],
       [ 9.4,  2.6,  0.7,  2.2],
       [ 5.5,  2.8,  9.4,  5.9],
       [ 6.5,  9.7,  2.7,  0. ],
       [ 8.9,  3.4,  4.7,  2.7],
       [ 4.5,  9.6,  5.1,  9.7],
       [10. , 10. ,  3.8,  2.5],
       [ 1.8,  4.4,  9.5, 10. ],
       [ 2.6,  3.7,  6.9,  5.5],
       [ 0. ,  8.7,  0. ,  1.6],
       [ 0.9,  9.2,  8.2,  9. ],
       [ 9.9,  4.9,  8.4,  4.9],
       [ 6. ,  3.4,  1. ,  2.7],
       [ 7.3,  1.3,  9.2,  6.8]])
        
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
        
    # Add constraints
    for i in range(M):
        prob.addConstraint(sum(x[i,j] for j in range(min(i+1,K))) == 1)
        for j in range(K):
            if j <= i:
                prob.addConstraint(y[i] >= np.dot(w[j], a[i]) - gamma[j] - BigM*(1-x[i,j]))
                prob.addConstraint(y[i] >= np.dot(-w[j], a[i]) + gamma[j] - BigM*(1-x[i,j]))
    
    # set objective
    prob.setObjective(sum(y[i]*y[i] for i in range(M)), sense = xp.minimize)
    
    global all_variables, w_variables_idxs
    all_variables = prob.getVariable()
    w_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("w")]
        
    return prob

def cbchecksol(prob, data, soltype, cutoff):
    # print('ENTER CBCHECKSOL PREINTSOL')
    """Callback function to reject the solution if it is not on the ball and accept otherwise."""
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

    # Check if the norm is 1
    # If so, we have a feasible solution
    # refuse = 0 if CheckOnBall(sol) else 1
    if check_all_balls(w_array):
        refuse = 0 
        # print('A feasible solution found!')
    else:
        refuse = 1
    # Return with refuse != 0 if solution is rejected, 0 otherwise;
    # and same cutoff
    return (refuse, cutoff)

def cbbranch(prob, data, branch):
    # print('ENTER CBBRANCH CHGBRANCHOBJ')
    """Callback function to create new branching object and add the corresponding constraints."""
    # return new branching object
    sol = []

    if (prob.attributes.presolvestate & 128) == 0:
        return branch

    # Retrieve node solution
    try:
        prob.getlpsol(x=sol)
        # print('The relaxation solution is ', sol[:N*K])
    except:
        return branch
    
    w_sol = sol[min(w_variables_idxs): max(w_variables_idxs)+1]
    w_array = split_data(w_sol, K, N)
    split_index = split_data(w_variables_idxs, K, N)

    if check_all_balls(w_array):
        return branch
    
    # Check if it is on the root node
    norms = np.linalg.norm(w_array, axis=-1)
    # Choose a ball to branch on
    ball_idx = np.argmin(norms)
    w_ball_idx = list(split_index[ball_idx])
    if np.any(norms < 1e-6):
        # create the new object with n+1 empty branches based on an empty ball
        bo = xp.branchobj(prob, isoriginal=True)
        bo.addbranches(N + 1)
        initial_polytope = create_n_simplex(N) # make it global
        for i in range(N + 1):
            # exclude point initial_polytope[i]
            submatrix = np.delete(initial_polytope, i, axis=0)  # Exclude row i
            # derive H-version of facet relaxation
            a_coeff = up_extension_constraint(submatrix)

            # Here is the difference since we need to assign constraints on the correct ball
            for j in range(len(a_coeff)):
                if j == 0:
                    bo.addrows(i, ['G'], [1], [0, N*K], w_ball_idx, a_coeff[j])
                else:
                    dot_products = [np.dot(a_coeff[j], row) for row in submatrix]
                    if max(dot_products) < 1e-6:
                        # Negative case; switch 
                        a_coeff[j] = - a_coeff[j]
                    bo.addrows(i, ['G'], [0], [0, N*K], w_ball_idx, a_coeff[j])
        return bo
    else:
        pi_w_array = [ProjectOnBall(w_j) for w_j in w_array]
        initial_points = data[prob.attributes.currentnode][ball_idx]
        new_matrix = append_zeros_and_ones(initial_points)
        # This facet is not full rank (two points are too close)
        if np.linalg.matrix_rank(initial_points, tol=1e-4) < N or np.linalg.matrix_rank(new_matrix, tol=1e-4) != np.linalg.matrix_rank(initial_points, tol=1e-4):
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
    sol = []

    if (prob.attributes.presolvestate & 128) == 0:
        return 0

    # Retrieve node solution
    try:
        prob.getlpsol(x=sol)
    except:
        return 0
    
    # Create empty dict
    data[newnode] = {}

    w_sol = sol[min(w_variables_idxs): max(w_variables_idxs)+1]
    w_array = split_data(w_sol, K, N)
    norms = np.linalg.norm(w_array, axis=-1)
    ball_idx = np.argmin(norms)
    # print(norms, ball_idx)
    if np.any(norms < 1e-6):
        # in a root node somehow
        initial_polytope = create_n_simplex(N)
        for k in range(K):
            if k == ball_idx:
                data[newnode][ball_idx] = np.delete(initial_polytope, branch, axis=0)
            else:
                try:
                    data[newnode][k] = data[parentnode][k]
                except:
                    data[newnode][k] = initial_polytope
    else:    
        initial_polytope = data[parentnode][ball_idx]
        submatrix = np.delete(initial_polytope, branch, axis=0)
        pi_w = ProjectOnBall(w_array[ball_idx])
        for k in range(K):
            if k == ball_idx:
                data[newnode][ball_idx] = np.vstack((submatrix, pi_w))
            else:
                try:
                    data[newnode][k] = data[parentnode][k]
                except:
                    data[newnode][k] = initial_polytope
    return 0

def cbprojectedsol(prob, data):
    rand = np.random.rand()
    if rand >= 0.1:
        return 0
    
    sol = []

    try:
        prob.getlpsol(x=sol)  # Retrieve node solution
    except:
        return 0

    w_sol = sol[min(w_variables_idxs): max(w_variables_idxs) + 1]

    # Check if solution is feasible
    if np.linalg.norm(w_sol) >= 1e-6:
        projected_sol = np.array(sol) / np.linalg.norm(w_sol)
        prob.addmipsol(projected_sol, all_variables, "Projected Solution")

    return 0

def solveprob(prob):
    """Function to solve the problem with registered callback functions."""

    data = {}
    prob.setControl('feastol', 1e-6)
    prob.addcbpreintsol(cbchecksol, data, 1)
    prob.addcbchgbranchobject(cbbranch, data, 1)
    prob.addcbnewnode(cbnewnode, data, 1)
    # prob.addcbnodelpsolved(cbprojectedsol, data, 1)
    # prob.controls.outputlog = 1
    prob.controls.presolve = 0

    # Adjust tolerances
    # prob.controls.feastol = 1e-6
    # prob.controls.optimalitytol = 1e-6
    # prob.controls.miptol = 1e-6
    
    prob.controls.timelimit=60
    prob.mipoptimize()
    
    print("Solution status:", prob.getProbStatusString())
    
if __name__ == '__main__':
    # np.random.seed(123)
    tol = 1e-4
    # prob = create_problem(filename = 'srncr_20_2_3_2023_ins.dat')
    prob = create_problem()
    solveprob(prob)
    num_nodes = prob.attributes.nodes
    print("Number of nodes in the tree:", num_nodes)
    sol = prob.getSolution()

