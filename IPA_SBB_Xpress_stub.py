# from __future__ import print_function

import xpress as xp
import numpy as np
from scipy.linalg import null_space
np.set_printoptions(precision=3, suppress=True)

def gram_schmidt(A):
    """Orthogonalize a set of vectors stored as the columns of matrix A."""
    # Get the number of vectors.
    n = A.shape[1]
    for j in range(n):
        # To orthogonalize the vector in column j with respect to the
        # previous vectors, subtract from it its projection onto
        # each of the previous vectors.
        for k in range(j):
            A[:, j] -= np.dot(A[:, k], A[:, j]) * A[:, k]
        A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
    return A

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
    a_coeff = []
    # create up_extension constraints' coefficient from 
    E = np.array(vertices_matrix)
    # find normal vector aw >= 1 from Ea = e
    e = np.ones(len(E))
    a = np.linalg.pinv(E)@e
    a_coeff.append(a)
    
    # choose n-1 extreme points of the facet and the origin to create a hyperplane of the form aw >= 0
    # nonzero a can be found from null space of E_prime
    # Initialize an empty list to store subarrays
    subarrays = []

    # Loop through each row and create a subarray by deleting that row
    for i in range(E.shape[0]):
        subarray = np.delete(E, i, axis=0)
        subarrays.append(subarray)

    # Convert the list of subarrays to a numpy array
    subarrays = np.array(subarrays)
    
    for subarray in subarrays:
        null_basis = null_space(subarray)
        for a_null in null_basis.T:
            a_coeff.append(a_null)
    return a_coeff

def CheckOnBall(w):
    """Check if the obtained solution norm is almost one or not."""
    tol = 1e-6
    if np.abs(np.linalg.norm(w) - 1) < tol:
        return True
    else:
        return False

def ProjectOnBall(w):
    """Project the obtained solution onto the ball through the origin."""
    return w/np.linalg.norm(w)

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
        global n 
        n = 3 #dimension
        m = 10 #number of A points
        k = 12 #number of B points

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

        # set objective
        prob.setObjective(sum(y)+sum(z), sense = xp.minimize)
    
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
    # print(sol)

    # Check if the norm is 1
    # If so, we have a feasible solution

    # refuse = 0 if CheckOnBall(sol) else 1
    if CheckOnBall(w_sol):
        # print('Norm of the solution = ', np.linalg.norm(w_sol), "I am on the ball!")
        refuse = 0 
    else:
        # print('Norm of the solution = ', np.linalg.norm(w_sol), "I am NOT the ball!")
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
        bo = xp.branchobj(prob, isoriginal=True)
        bo.addbranches(n+1)
        initial_polytope = create_n_simplex(n)
        for i in range(n+1):
            # exclude point initial_polytope[i]
            submatrix = np.delete(initial_polytope, i, axis=0)  # Exclude row i
            # derive H-version of facet relaxation
            a_coeff = up_extension_constraint(submatrix)
            for j in range(len(a_coeff)):
                # Need to figure out index and start of variables w
                if j == 0:
                    # add constraint aw >= 1
                    bo.addrows(i, ['G'], [1], [0, len(w_variables_idxs)], w_variables_idxs, a_coeff[j])
                else:
                    # add constraint aw <= 0
                    bo.addrows(i, ['L'], [0], [0, len(w_variables_idxs)], w_variables_idxs, a_coeff[j])
                    
            maxrows = 100
            maxcoefs = 100
            rowtype = []
            rhs = []
            start = []
            colind = []
            rowcoef = []
            bo.getrows(i, maxrows, maxcoefs, rowtype, rhs, start, colind, rowcoef)
            print("Coefficient of branch " ,i," is " , np.array(rowcoef))
            print("RHS = ", rhs)
            print("Row type = ", rowtype)
        return bo
    else:
        pi_w = ProjectOnBall(w_sol)
        # extreme points of the current node
        initial_points = data[prob.attributes.currentnode]
        print('The current node is ', prob.attributes.currentnode)
        print('W_sol = ', w_sol, 'Projected Point Pi_w = ', pi_w)
        print('The extreme points at this node are ', data[prob.attributes.currentnode])
        
        # ---------------------------------------------------------------------------------
        # Branch only if the matrix of extreme points is full rank
        # ---------------------------------------------------------------------------------
        non_empty_i = []
        for i in range(n):
            submatrix = np.delete(initial_points, i, axis=0)
            extreme_points = np.concatenate((submatrix, [pi_w]), axis=0)
            # non_empty_i.append(i)
            new_matrix = append_zeros_and_ones(extreme_points)
            if np.linalg.matrix_rank(new_matrix) != np.linalg.matrix_rank(extreme_points):
                print('The matrix of extreme point of this facet do not contain 0 ', extreme_points)
                non_empty_i.append(i)
            # if n != np.linalg.matrix_rank(extreme_points):
            #     non_empty_i.append(i)
            # if pi_w not in submatrix:
            #     extreme_points = np.concatenate((submatrix, [pi_w]), axis=0)
            #     # Check Critical Case
            #     new_matrix = append_zeros_and_ones(extreme_points)
            #     if np.linalg.matrix_rank(new_matrix) != np.linalg.matrix_rank(extreme_points):
            #         print('The matrix of extreme point of this facet do not contain 0 ', extreme_points)
            #         non_empty_i.append(i)
        
        n_branch = len(non_empty_i)
        print('n_branch = ', n_branch)
        
        if n_branch == 0:
            return branch
        else:
            bo = xp.branchobj(prob, isoriginal=True)
            bo.addbranches(n_branch)
            for i in range(n_branch):
                submatrix = np.delete(initial_points, non_empty_i[i], axis=0)
                extreme_points = np.concatenate((submatrix, [pi_w]), axis=0)
                a_coeff = up_extension_constraint(extreme_points)
                for j in range(len(a_coeff)):
                    # create constraints from extreme points of the facet
                    if j == 0:
                        # add constraint aw >= 1
                        bo.addrows(i, ['G'], [1], [0, len(w_variables_idxs)], w_variables_idxs, a_coeff[j])
                    else:
                        # add constraint aw <= 0
                        bo.addrows(i, ['L'], [0], [0, len(w_variables_idxs)], w_variables_idxs, a_coeff[j])
            maxrows = 100
            maxcoefs = 100
            rowtype = []
            rhs = []
            start = []
            colind = []
            rowcoef = []
            bo.getrows(i, maxrows, maxcoefs, rowtype, rhs, start, colind, rowcoef)
            print("Coefficient of branch " ,i," is " , np.array(rowcoef))
            print("RHS = ", rhs)
            print("Row type = ", rowtype)
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

    print('Parent node = ', parentnode, 'new node = ', newnode, 'branch = ',branch)
    # coef, ind, beg = [], [], []
    # prob.getrows(beg, ind, coef, 1000000, 0, prob.attributes.rows - 1)
    # print("COEF = ", coef)
    # print(prob.attributes.rows)
    # print("IND = ", ind)
    # print("BEG = ", beg)
    
    if int(newnode) <= n+2: # these nodes are created from the root node
        initial_polytope = create_n_simplex(n)
        submatrix = np.delete(initial_polytope, branch, axis=0)
        data[newnode] = submatrix
    else:
        # get relaxation solution
        all_variables = prob.getVariable()
        w_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("w")]
        w_sol = sol[min(w_variables_idxs): max(w_variables_idxs)+1]
        # projection new point
        pi_w = np.array(ProjectOnBall(w_sol))
        # print('The projected point is ', pi_w)
        initial_polytope = data[parentnode]
        submatrix = np.delete(initial_polytope, branch, axis=0)
        # add point pi(w*)
        extreme_points = np.concatenate((submatrix, [pi_w]), axis=0)
        data[newnode] = extreme_points
    return 0

def solveprob(prob):
    """Function to solve the problem with registered callback functions."""

    data = {}
    prob.addcbpreintsol(cbchecksol, data, 1)
    # p.addcboptnode(cbaddcuts, data, 3)
    prob.addcbchgbranchobject(cbbranch, data, 1)
    prob.addcbnewnode(cbnewnode, data, 1)
    prob.mipoptimize()
    
    print("Solution status:", prob.getProbStatusString())
    sol = prob.getSolution()
    all_variables = prob.getVariable()
    w_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("w")]
    w_sol = sol[min(w_variables_idxs): max(w_variables_idxs)+1]
    print("Optimal solution W:", np.array(w_sol), ' with norm = ', np.linalg.norm(w_sol))
    print("Optimal solution:", np.array(prob.getSolution()))
    print("Optimal objective value:", np.array(prob.getObjVal()))
    print("Solver Status:", prob.getProbStatus())
    
    
    
if __name__ == '__main__':
    prob = create_problem(filename = None)
    solveprob(prob)
