import xpress as xp
import numpy as np
from scipy.linalg import null_space
# import re
# from scipy.spatial.distance import cdist
# import pandas as pd
# import line_profiler
np.set_printoptions(precision=3, suppress=True)
# xp.init('C:/Apps/Anaconda3/lib/site-packages/xpress/license/community-xpauth.xpr')
xp.init('C:/Users/montr/anaconda3/Lib/site-packages/xpress/license/community-xpauth.xpr') # License path for Laptop

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
        # M, N, K, a = read_dat_file(filename)
        # print(M, N, K)
        
        # # Create problem for k-hyperplane clustering without norm constraints
        
        # # Create variables
        # w = xp.vars(K, N, name='w', lb=0)
        # gamma = xp.vars(K, name="gamma", lb=0)
        # x = xp.vars(M, K, name='x', vartype=xp.binary)
        # y = xp.vars(M, name='y', lb=0)
        # prob.addVariable(w, gamma, x, y)
        
        # # Add constraints
        # BigM = 10e+6
        # for i in range(M):
        #     prob.addConstraint(sum(x[i,j] for j in range(K)) == 1)
        #     for j in range(K):
        #         prob.addConstraint(y[i] >= np.dot(w[j], a[i]) - gamma[j] - BigM*(1-x[i,j]))
                
        # # set objective
        # prob.setObjective(sum(y[i]*y[i] for i in range(K)), sense = xp.minimize)
        return prob
    else:
        # Create a simple Linear classifier problem
        # n = 3, m = 4, k = 5
        global n 
        n = 2 #dimension
        m = 500 #number of A points
        k = 500 #number of B points

        # Random matrix A, B
        np.random.seed(1012310)
        A = np.random.random(m*n).reshape(m, n)
        B = np.random.random(k*n).reshape(k, n)
        
        # Create variables
        w = xp.vars(n, name='w')
        gamma = xp.var(name="gamma")

        # A-point distance
        y = xp.vars(m, name='y', lb=0)

        # B-point distance
        z = xp.vars(k, name='z', lb=0)

        prob.addVariable(w, gamma, y, z)

        # Add constraints
        for i in range(m):
            # prob.addConstraint(y[i] >= np.dot(-w, A[i]) + gamma)
            prob.addConstraint(y[i] >= np.dot(w, A[i]) - gamma) # swap label

        for j in range(k):
            # prob.addConstraint(z[j] >= np.dot(w, B[j]) - gamma)
            prob.addConstraint(z[j] >= np.dot(-w, B[j]) + gamma) # swap label
            
        # set objective
        prob.setObjective(sum(y)+sum(z), sense = xp.minimize)

    return prob

# def create_problem_full(filename = None):

#     # Create a new optimization problem
#     prob = xp.problem()
#     # Disable presolve
#     prob.controls.xslp_presolve = 0
#     prob.presolve()

#     # Read the problem from a file
#     if filename != None:
#         # Read dat file to get M, N, K, a
#         M, N, K, a = read_dat_file(filename)
#         print(M, N, K)
        
#         # Create problem for k-hyperplane clustering without norm constraints
        
#         # Create variables
#         w = xp.vars(K, N, name='w', lb=0)
#         gamma = xp.vars(K, name="gamma", lb=0)
#         x = xp.vars(M, K, name='x', vartype=xp.binary)
#         y = xp.vars(M, name='y', lb=0)
#         prob.addVariable(w, gamma, x, y)
        
#         # Add constraints
#         BigM = 10e+4 
#         for i in range(M):
#             prob.addConstraint(sum(x[i,j] for j in range(K)) == 1)
#             for j in range(K):
#                 prob.addConstraint(y[i] >= np.dot(w[j], a[i]) - gamma[j] - BigM*(1-x[i,j]))
                
#         # Norm constraints
#         for j in range(K):
#             prob.addConstraint(sum(w[j,d]*w[j,d] for d in range(N)) >= 1)
        
#         return prob
#     else:
#         # Create a simple Linear classifier problem
#         # n = 3, m = 4, k = 5
#         global n 
#         n = 4 #dimension
#         m = 250 #number of A points
#         k = 250 #number of B points

#         # Random matrix A, B
#         np.random.seed(1012310)
#         A = np.random.random(m*n).reshape(m, n)
#         B = np.random.random(k*n).reshape(k, n)
        
#         # Create variables
#         w = xp.vars(n, name='w')
#         gamma = xp.var(name="gamma")

#         # A-point distance
#         y = xp.vars(m, name='y', lb=0)

#         # B-point distance
#         z = xp.vars(k, name='z', lb=0)

#         prob.addVariable(w, gamma, y, z)

#         # Add constraints
#         for i in range(m):
#             prob.addConstraint(y[i] >= np.dot(-w, A[i]) + gamma)

#         for j in range(k):
#             prob.addConstraint(z[j] >= np.dot(w, B[j]) - gamma)
        
#         prob.addConstraint(sum(w[i]*w[i] for i in range(n)) >= 1)

#         # set objective
#         prob.setObjective(sum(y)+sum(z))
    
#     return prob

def cbchecksol(prob, data, soltype, cutoff):
    """Callback function to reject the solution if it is not on the ball and accept otherwise."""
    if (prob.attributes.presolvestate & 128) == 0:
        return (1, 0)
    
    sol = []

    # Retrieve node solution
    try:
        prob.getlpsol(x=sol)
    except:
        return (1, cutoff)

    sol = np.array(sol)
    all_variables = prob.getVariable()
    w_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("w")]
    w_sol = sol[min(w_variables_idxs): max(w_variables_idxs)+1]

    # Check if the norm is 1
    # If so, we have a feasible solution
    # refuse = 0 if CheckOnBall(sol) else 1
    if CheckOnBall(w_sol):
        refuse = 0 
    else:
        refuse = 1
    # Return with refuse != 0 if solution is rejected, 0 otherwise;
    # and same cutoff
    return (refuse, cutoff)

def cbbranch(prob, data, branch):
    """Callback function to create new branching object and add the corresponding constraints."""
    # return new branching object
    sol = []

    if (prob.attributes.presolvestate & 128) == 0:
        return branch

    # Retrieve node solution
    try:
        prob.getlpsol(x=sol)
    except:
        return branch
    
    all_variables = prob.getVariable()
    w_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("w")]
    # print(w_variables_idxs)
    w_sol = sol[min(w_variables_idxs): max(w_variables_idxs)+1]

    if CheckOnBall(w_sol):
        return branch
    
    # Check if it is on the root node
    if all(element < 1e-8 for element in sol):
        # create the new object with n+1 empty branches
        bo = xp.branchobj(prob, isoriginal=True)
        bo.addbranches(n+1)
        initial_polytope = create_n_simplex(n)
        for i in range(n+1):
            # exclude point initial_polytope[i]
            submatrix = np.delete(initial_polytope, i, axis=0)  # Exclude row i
            # derive H-version of facet relaxation
            try:
                a_coeff = up_extension_constraint(submatrix)
            except:
                return branch

            for j in range(len(a_coeff)):
                if j == 0:
                    # add constraint aw >= 1
                    print(w_variables_idxs, a_coeff[j])
                    bo.addrows(i, ['G'], [1], [0, len(w_variables_idxs)], w_variables_idxs, a_coeff[j])
                    # bo.addrows(i, ['G'], [1], [0], w_variables_idxs, a_coeff[j])
                else:
                    dot_products = [np.dot(a_coeff[j], row) for row in submatrix]
                    if max(dot_products) < 1e-8:
                        # Negative case; switch 
                        a_coeff[j] = - a_coeff[j]
                    bo.addrows(i, ['G'], [0], [0, len(w_variables_idxs)], w_variables_idxs, a_coeff[j])
                    # bo.addrows(i, ['G'], [0], [0], w_variables_idxs, a_coeff[j])
        return bo
    else:
        pi_w = ProjectOnBall(w_sol)
        # extreme points of the current node
        initial_points = data[prob.attributes.currentnode]
        new_matrix = append_zeros_and_ones(initial_points)
        # This facet is not full rank (two points are too close)
        if np.linalg.matrix_rank(data[prob.attributes.currentnode], tol=1e-4) < n or np.linalg.matrix_rank(new_matrix, tol=1e-4) != np.linalg.matrix_rank(initial_points, tol=1e-4):
            return branch

        # create new object with n empty branches
        bo = xp.branchobj(prob, isoriginal=True)
        bo.addbranches(n)
        for i in range(n):
            submatrix = np.delete(initial_points, i, axis=0)
            extreme_points = np.concatenate((submatrix, [pi_w]), axis=0)
            try:
                a_coeff = up_extension_constraint(extreme_points)
            except:
                print('Cannot obtain coefficient for constraints')
                return branch
            for j in range(len(a_coeff)):
                if j == 0:
                    bo.addrows(i, ['G'], [1], [0, len(w_variables_idxs)], w_variables_idxs, a_coeff[j])
                    # bo.addrows(i, ['G'], [1], [0], w_variables_idxs, a_coeff[j])
                else:
                    dot_products = [np.dot(a_coeff[j], row) for row in extreme_points]
                    if max(dot_products) < 1e-6:
                        # Negative case; switch 
                        a_coeff[j] = - a_coeff[j]
                    bo.addrows(i, ['G'], [0], [0, len(w_variables_idxs)], w_variables_idxs, a_coeff[j])
                    # bo.addrows(i, ['G'], [0], [0], w_variables_idxs, a_coeff[j])
        return bo
    return branch

def cbnewnode(prob, data, parentnode, newnode, branch):
    """Callback function to add data of extreme points to each node."""
    sol = []

    if (prob.attributes.presolvestate & 128) == 0:
        return 0

    # Retrieve node solution
    try:
        prob.getlpsol(x=sol)
    except:
        return 0
    
    if int(newnode) <= n + 2:
        initial_polytope = create_n_simplex(n)
        data[newnode] = np.delete(initial_polytope, branch, axis=0)
    else:
        all_variables = prob.getVariable()
        w_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("w")]
        w_sol = sol[min(w_variables_idxs): max(w_variables_idxs)+1]
        initial_polytope = data[parentnode]
        submatrix = np.delete(initial_polytope, branch, axis=0)
        pi_w = ProjectOnBall(w_sol)
        data[newnode] = np.vstack((submatrix, pi_w))
    return 0

# def cbfindsol(prob, data):
#     prob_full = create_problem_full(filename = None)
#     prob_full.optimize('x')
#     num_nodes_full = prob_full.attributes.nodes
#     print("Number of nodes in the tree:", num_nodes_full)
#     sol_full = prob_full.getSolution()[:-1]
#     prob.addmipsol(sol_full, prob.getVariable(), "True Solution")
#     prob.removecboptnode(cbfindsol, data)
#     return 0

def cbprojectedsol(prob, data):
    # rand = np.random.rand()
    # if rand >= 0.1:
    #     return 0
    
    sol = []

    try:
        prob.getlpsol(x=sol)  # Retrieve node solution
    except:
        return 0

    all_variables = prob.getVariable()
    w_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("w")]
    w_sol = sol[min(w_variables_idxs): max(w_variables_idxs) + 1]

    # Check if solution is feasible
    if np.linalg.norm(w_sol) >= 1e-6:
        projected_sol = np.array(sol) / np.linalg.norm(w_sol)
        prob.addmipsol(projected_sol, all_variables, "Projected Solution")

    return 0



def solveprob(prob):
    """Function to solve the problem with registered callback functions."""

    data = {}
    # prob.setControl('feastol', 1e-6)
    prob.addcbpreintsol(cbchecksol, data, 1)
    prob.addcbchgbranchobject(cbbranch, data, 1)
    prob.addcbnewnode(cbnewnode, data, 1)
    prob.addcbnodelpsolved(cbprojectedsol, data, 1)
    # prob.addcboptnode(cbfindsol, data, 1)
    prob.controls.outputlog = 1
    prob.mipoptimize()
    
    # print("Solution status:", prob.getProbStatusString())
    # sol = prob.getSolution()
    # all_variables = prob.getVariable()
    # w_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("w")]
    # w_sol = sol[min(w_variables_idxs): max(w_variables_idxs)+1]
    # print("Optimal solution W:", w_sol, ' with norm = ', np.linalg.norm(w_sol))
    num_nodes = prob.attributes.nodes
    print("Number of nodes in the tree:", num_nodes)
    # print("Optimal objective value:", prob.getObjVal())
    # print("Solver Status:", prob.getProbStatus())
    
    
if __name__ == '__main__':
    tol = 1e-4
    # prob_full = create_problem_full(filename = 'srncr_20_2_3_2023_ins.dat')
    # prob_full.optimize('x')
    # print("Solution status:", prob_full.getProbStatusString())
    # num_nodes = prob_full.attributes.nodes
    # print("Number of nodes in the tree:", num_nodes)
    # sol = prob_full.getSolution()
    # all_variables = prob_full.getVariable()
    # y_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("y")]
    # y_sol = sol[min(y_variables_idxs): max(y_variables_idxs) + 1]
    # print(y_sol)
    prob = create_problem(filename = None)
    # prob = create_problem(filename = 'srncr_20_2_3_2023_ins.dat')
    solveprob(prob)
