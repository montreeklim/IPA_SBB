from __future__ import print_function

import xpress as xp
import numpy as np
from scipy.linalg import null_space

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

# This function should be used in a callback 
# H-version of facet relaxation
# verices_matrix = matrix of extreme points
def up_extension_constraint(vertices_matrix):
    a_coeff = []
    # create up_extension constraints' coefficient from 
    E = np.array(vertices_matrix)
    # find normal vector aw >= 1 from Ea = e
    e = np.ones(len(E))
    a = np.linalg.pinv(E)@e
    a_coeff.append(a)
    
    # add constraint
    # new_model.addConstr(gp.quicksum(a[i]*new_model.x[i] for i in range(len(a))) >= 1)
    
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
        a_null = np.squeeze(null_space(subarray))
        a_coeff.append(a_null)
    # add constraint
    # new_model.addConstr(gp.quicksum(a_null[i]*new_model.x[i] for i in range(len(a))) >= 0)
    return a_coeff

# def CheckOnBall(w):
#     tol = 1e-8
#     if np.linalg.norm(w) <= 1-tol:
#         return False
#     else:
#         return True
    
def CheckOnBall(w):
    tol = 1e-8
    if np.abs(np.linalg.norm(w) - 1) < tol:
        return True
    else:
        return False

def ProjectOnBall(w):
    return w/np.linalg.norm(w)

def CreateNewFacets(vertices_matrix, w):
    vertices_matrix = np.array(vertices_matrix)
    w = np.array(w)
    # facet is of the form Ax >= b where b = [1,0,0,...,0] E = matrix of the facet
    new_point = ProjectOnBall(w)
    # replace one row of E with new_point
    list_of_facets = []
    for i in range(len(vertices_matrix)):
        new_facet_matrix = vertices_matrix.copy()
        new_facet_matrix[i] = new_point
        list_of_facets.append(new_facet_matrix)
    return list_of_facets

def create_problem(filename = None):

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
        n = 2 #dimension
        m = 4 #number of A points
        k = 5 #number of B points

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
        prob.setObjective(sum(y)+sum(z))
        
        # Disable presolved
        prob.setControl({"presolve":0})
        # Silence output
        # prob.setControl ('outputlog', 0)
    
    return prob

def cbchecksol(prob, data, soltype, cutoff):
    # print('We are in the preintsol callback.')
    # if (prob.attributes.presolvestate & 128) == 0:
    #     return (1, 0)
    
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
        # print("I am on the ball!")
        refuse = 0 
    else:
        # print("I am NOT the ball!")
        refuse = 1

    # Return with refuse != 0 if solution is rejected, 0 otherwise;
    # and same cutoff
    return (refuse, cutoff)

def cbbranch(prob, data, branch):
    # return new branching object
    
    # print("We entered the Change Branch Object Callback")
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
    # print(w_sol)

    # Check if it is on the root node
    if all(element < 1e-8 for element in sol):
        # create the new object with n+1 empty branches
        # bo = xp.branchobj(prob, isoriginal=True)
        bo = xp.branchobj(prob, isoriginal=False)
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
                    # add constraint aw >= 0
                    bo.addrows(i, ['G'], [0], [0, len(w_variables_idxs)], w_variables_idxs, a_coeff[j])
        return bo
    else:
        print("IN THE ELSE BRANCH")
        print('Branch id = ', branch.getid())
        print('Number of branches = ', branch.getbranches())
        # create new object with n empty branches
        bo = xp.branchobj(prob, isoriginal=False)
        bo.addbranches(n)
        # get facets
        pi_w = ProjectOnBall(w_sol)
        # Need to find the data w.r.t. current branc
        list_of_facets = CreateNewFacets(data[4], pi_w) # ***Need to fix in general term***
        # Add constraints for each facet
        for i in range(n):
            facet = list_of_facets[i]
            for j in range(len(facet)):
                # create constraints from extreme points of the facet
                a_coeff = up_extension_constraint(facet)
                if j == 0:
                    # add constraint aw >= 1
                    bo.addrows(i, ['G'], [1], [0, len(w_variables_idxs)], w_variables_idxs, a_coeff[j])
                else:
                    # add constraint aw >= 0
                    bo.addrows(i, ['G'], [0], [0, len(w_variables_idxs)], w_variables_idxs, a_coeff[j])
        return bo
    return branch

def cbnewnode(prob, data, parentnode, newnode, branch):
    sol = []

    if (prob.attributes.presolvestate & 128) == 0:
        return 0

    # Retrieve node solution
    try:
        prob.getlpsol(x=sol)
    except:
        return 0
    
    if int(newnode) <= n+2: # these nodes are created from the root node
        initial_polytope = create_n_simplex(n)
        submatrix = np.delete(initial_polytope, branch, axis=0)
        data[newnode] = submatrix
    else:
        print('Parentnode = ', parentnode, ', Newnode = ', newnode, ', branch = ', branch)
        # get relaxation solution
        all_variables = prob.getVariable()
        w_variables_idxs = [ind for ind, var in enumerate(all_variables) if var.name.startswith("w")]
        w_sol = sol[min(w_variables_idxs): max(w_variables_idxs)+1]
        # projection new point
        pi_w = np.array(ProjectOnBall(w_sol))
        initial_polytope = data[parentnode]
        submatrix = np.delete(initial_polytope, branch, axis=0)
        # add point pi(w*)
        extreme_points = np.concatenate((submatrix, [pi_w]), axis=0)
        data[newnode] = extreme_points
    return 0

def solveprob(prob):

    data = {}
    prob.addcbpreintsol(cbchecksol, data, 1)
    # p.addcboptnode(cbaddcuts, data, 3)
    prob.addcbchgbranchobject(cbbranch, data, 1)
    # prob.addcbchgbranchobject(cbbranch_stub, data, 1)
    prob.addcbnewnode(cbnewnode, data, 1)
    prob.mipoptimize()
    
    print("Solution status:", prob.getProbStatusString())
    print("Optimal solution:", prob.getSolution())
    print("Optimal objective value:", prob.getObjVal())
    print("Solver Status:", prob.getProbStatus())
    
    
if __name__ == '__main__':
    prob = create_problem(filename = None)
    prob.write("simple_prob.lp")
    solveprob(prob)


# print("------")
# print("solving")
# print("------")

# SLP setting
# prob.setControl({'slpfeastol': 1e-4, 'slppresolve': 0})

# Solve the problem
# prob.mipoptimize('x')
# prob.optimize('x') # Use optimize('x') for global solver

