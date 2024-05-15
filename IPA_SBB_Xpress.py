import xpress as xp
import numpy as np
from scipy.linalg import null_space
import line_profiler
np.set_printoptions(precision=3, suppress=True)
# xp.init('C:/Apps/Anaconda3/lib/site-packages/xpress/license/community-xpauth.xpr')

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
    
    # Compute the coefficient 'a' for the first constraint
    e = np.ones(len(E))
    a_coeff = np.empty((0, len(E[0])))  # Initialize an empty array to store coefficients
    
    a = np.linalg.pinv(E) @ e
    # a = np.linalg.solve(E, e)
    a_coeff = np.vstack((a_coeff, a))
    
    # Compute coefficients for additional constraints
    n = E.shape[0] - 1  # Number of extreme points
    for i in range(n):
        # Compute the null space basis of the subarray by removing the i-th row
        null_basis = null_space(np.delete(E, i, axis=0))
        
        # Append each null space vector as a constraint coefficient
        a_coeff = np.vstack((a_coeff, null_basis.T))
    
    return a_coeff



def CheckOnBall(w):
    """Check if the obtained solution norm is almost one or not."""
    # tol = 1e-6 if w close to pi(w) accept the 
    # tol_diff = 1e-2
    # pi_w = ProjectOnBall(w)
    # diff = np.abs(pi_w - w)
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
    
def create_problem(filename = None):
    """Read the given problem or create simple linear classifier problem."""

    # Create a new optimization problem
    prob = xp.problem()
    # Disable presolve
    prob.controls.xslp_presolve = 0
    prob.presolve()

    # Read the problem from a file
    if filename != None:
        prob.read(filename)
    else:
        # Create a simple Linear classifier problem
        # n = 3, m = 4, k = 5
        global n 
        n = 5 #dimension
        m = 20 #number of A points
        k = 20 #number of B points

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
            prob.addConstraint(y[i] >= np.dot(-w, A[i]) + gamma)

        for j in range(k):
            prob.addConstraint(z[j] >= np.dot(w, B[j]) - gamma)
            
        # set objective
        prob.setObjective(sum(y)+sum(z), sense = xp.minimize)
        
        # Disable presolved
        # prob.setControl({"presolve":0})
        # Silence output
        # prob.setControl ('outputlog', 0)
    
    return prob

def create_problem_full(filename = None):

    # Create a new optimization problem
    prob = xp.problem()
    # Disable presolve
    prob.controls.xslp_presolve = 0
    prob.presolve()

    # Read the problem from a file
    if filename != None:
        prob.read(filename)
    else:
        # Create a simple Linear classifier problem
        # n = 3, m = 4, k = 5
        global n 
        n = 5 #dimension
        m = 20 #number of A points
        k = 20 #number of B points

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
            prob.addConstraint(y[i] >= -sum(w[j] * A[i][j] for j in range(n)) + gamma)

        for j in range(k):
            prob.addConstraint(z[j] >= sum(w[i] * B[j][i] for i in range(n)) - gamma)
        
        prob.addConstraint(sum(w[i]*w[i] for i in range(n)) >= 1)

        # set objective
        prob.setObjective(sum(y)+sum(z))
    
    return prob

def cbchecksol(prob, data, soltype, cutoff):
    """Callback function to reject the solution if it is not on the ball and accept otherwise."""
    # print('We are in the preintsol callback.')
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
    
    # Add partial mipsol
    # prob.addmipsol(w_sol, w_variables_idxs, name="Partial Solution")

    # Check if the norm is 1
    # If so, we have a feasible solution
    # refuse = 0 if CheckOnBall(sol) else 1
    if CheckOnBall(w_sol):
        # print('Norm of the solution = ', np.linalg.norm(w_sol), "I am on the ball!")
        refuse = 0 
    else:
        #     print('Norm of the solution = ', np.linalg.norm(w_sol), "I am NOT the ball!")
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
    w_sol = sol[min(w_variables_idxs): max(w_variables_idxs)+1]

    if CheckOnBall(w_sol):
        return branch
    
    # Check if it is on the root node
    if all(element < 1e-8 for element in sol):
        # create the new object with n+1 empty branches
        # bo = xp.branchobj(prob, isoriginal=True)
        bo = xp.branchobj(prob, isoriginal=True)
        bo.addbranches(n+1)
        initial_polytope = create_n_simplex(n)
        for i in range(n+1):
            # exclude point initial_polytope[i]
            submatrix = np.delete(initial_polytope, i, axis=0)  # Exclude row i
            # derive H-version of facet relaxation
            a_coeff = up_extension_constraint(submatrix)
            for j in range(len(a_coeff)):
                if j == 0:
                    # check feasibility
                    # for row in submatrix:    
                        # print('The product >= 1', np.dot(a_coeff[j], row) >= 1-tol)
                    # add constraint aw >= 1
                    bo.addrows(i, ['G'], [1], [0, len(w_variables_idxs)], w_variables_idxs, a_coeff[j])
                else:
                    dot_products = [np.dot(a_coeff[j], row) for row in submatrix]
                    # print(dot_products, min(dot_products), max(dot_products))
                    if max(dot_products) < 1e-8:
                        # Negative case; switch 
                        a_coeff[j] = - a_coeff[j]
                    # for row in submatrix:    
                    #     dot_product = np.dot(a_coeff[j], row)
                    #     if dot_product < -1e-8:
                    #         a_coeff[j] = - a_coeff[j]
                            # print('dot product is = ', dot_product)
                        # print('The product >= 0', np.dot(a_coeff[j], row) >= -tol)
                    # add constraint aw >= 0
                    bo.addrows(i, ['G'], [0], [0, len(w_variables_idxs)], w_variables_idxs, a_coeff[j])
            # print('a_coeff = ', a_coeff)
        return bo
    else:
        pi_w = ProjectOnBall(w_sol)
        # extreme points of the current node
        initial_points = data[prob.attributes.currentnode]
        # print('We are in node: ', prob.attributes.currentnode)
        new_matrix = append_zeros_and_ones(initial_points)
        # This facet is not full rank (two points are too close)
        if np.linalg.matrix_rank(data[prob.attributes.currentnode]) < n or np.linalg.matrix_rank(new_matrix) != np.linalg.matrix_rank(initial_points): 
            # print('EXCLUDE THE NODE ', prob.attributes.currentnode)
            return branch
        # else:
            # print('Solution = ', np.array(w_sol), ', projected point = ', pi_w)
            # print('The extreme points at this node are ', data[prob.attributes.currentnode])
            # print('The corresponding rank = ', np.linalg.matrix_rank(data[prob.attributes.currentnode]))
            
        
        # create new object with n empty branches
        bo = xp.branchobj(prob, isoriginal=True)
        bo.addbranches(n)
        for i in range(n):
            submatrix = np.delete(initial_points, i, axis=0)
            extreme_points = np.concatenate((submatrix, [pi_w]), axis=0)
            a_coeff = up_extension_constraint(extreme_points)
            for j in range(len(a_coeff)):
                if j == 0:
                    # for row in extreme_points:    
                    #     print('The product >= 1', np.dot(a_coeff[j], row) >= 1-tol)
                    bo.addrows(i, ['G'], [1], [0, len(w_variables_idxs)], w_variables_idxs, a_coeff[j])
                else:
                    dot_products = [np.dot(a_coeff[j], row) for row in extreme_points]
                    if max(dot_products) < 1e-8:
                        # Negative case; switch 
                        a_coeff[j] = - a_coeff[j]
                    # print(dot_products, min(dot_products), max(dot_products))
                    # for row in extreme_points:    
                    #     dot_product = np.dot(a_coeff[j], row)
                    #     if dot_product < -1e-8:
                    #         a_coeff[j] = - a_coeff[j]
                            # print('dot product is = ', dot_product)
                    bo.addrows(i, ['G'], [0], [0, len(w_variables_idxs)], w_variables_idxs, a_coeff[j])
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

def cbfindsol(prob, data):
    prob_full = create_problem_full(filename = None)
    prob_full.optimize('x')
    sol_full = prob_full.getSolution()[:-1]
    prob.addmipsol(sol_full, prob.getVariable(), "True Solution")
    prob.removecboptnode(cbfindsol, data)
    return 0

def cbprojectedsol(prob, data):
    # return new branching object
    sol = []

    # Retrieve node solution
    try:
        prob.getlpsol(x=sol)
    except:
        return 0
    
    all_variables = prob.getVariable()
    w_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("w")]
    w_sol = sol[min(w_variables_idxs): max(w_variables_idxs)+1]
    pi_w = ProjectOnBall(w_sol)
    # print('pi_w = ', pi_w)
    if CheckOnBall(pi_w):
        print('pi_w = ', pi_w)
        prob.addmipsol(pi_w, w_variables_idxs, "Partial Solution")
        # prob.removecboptnode(cbprojectedsol, data)
        return 0
    else:
        return 0


def solveprob(prob):
    """Function to solve the problem with registered callback functions."""

    data = {}
    prob.addcbpreintsol(cbchecksol, data, 1)
    prob.addcbchgbranchobject(cbbranch, data, 1)
    prob.addcbnewnode(cbnewnode, data, 1)
    # prob.addcboptnode(cbfindsol, data, 3)
    # prob.addcboptnode(cbprojectedsol, data, 3)
    prob.mipoptimize()
    
    print("Solution status:", prob.getProbStatusString())
    sol = prob.getSolution()
    all_variables = prob.getVariable()
    w_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("w")]
    w_sol = sol[min(w_variables_idxs): max(w_variables_idxs)+1]
    print("Optimal solution W:", w_sol, ' with norm = ', np.linalg.norm(w_sol))
    # print("Optimal solution:", prob.getSolution())
    # print('Length of solutions = ', len(prob.getSolution()))
    print("Optimal objective value:", prob.getObjVal())
    print("Solver Status:", prob.getProbStatus())
    
    
    
if __name__ == '__main__':
    # profile = line_profiler.LineProfiler()
    # profile.add_function(cbchecksol)
    # profile.add_function(cbbranch)
    # profile.add_function(cbnewnode)
    # profile.add_function(solveprob)
    
    tol = 1e-4
    prob = create_problem(filename = None)
    
    
    # Run your script with the profiler
    # profile.run('solveprob(prob)')

    # Display the profiling results
    # profile.print_stats()
    
    solveprob(prob)
