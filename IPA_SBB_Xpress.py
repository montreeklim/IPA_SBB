#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Test problem on a dot product between matrices of scalars and/or of
# variables. Note that the problem cannot be solved by the Optimizer
# as it is nonconvex.

from __future__ import print_function

import xpress as xp
import numpy as np
from scipy.linalg import null_space


# In[8]:


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

def up_extension_constraint(vertices_matrix):
    # new_model = model.copy()
    a_coeff = []
    # create up_extension constraints from 
    E = vertices_matrix
    # find normal vector aw >= 1 from Ea = e
    e = np.ones(len(E))
    a = np.linalg.pinv(E)@e
    a_coeff.append(a)
    
    # add constraint
    # new_model.addConstr(gp.quicksum(a[i]*new_model.x[i] for i in range(len(a))) >= 1)
    
    # choose n-1 extreme points of the facet and the origin to create a hyperplane of the form aw >= 0
    # nonzero a can be found from null space of E_prime
    for i, point in enumerate(E):
        E_prime = E.copy()
        E_prime[i] = np.zeros(len(E))
        a_null = null_space(E_prime)
        a_coeff.append(a_null.reshape(len(E),))
        # add constraint
        # new_model.addConstr(gp.quicksum(a_null[i]*new_model.x[i] for i in range(len(a))) <= 0)
    return a_coeff

def CheckOnBall(w):
    tol = 1e-10
    if np.linalg.norm(w) <= 1-tol:
        return False
    else:
        return True

def ProjectOnBall(w):
    return w/np.linalg.norm(w)

def CreateNewFacets(vertices_matrix, w):
    # facet is of the form Ax >= b where b = [1,0,0,...,0] E = matrix of the facet
    new_point = ProjectOnBall(w)
    # replace one row of E with new_point
    list_of_facets = []
    for i in range(len(vertices_matrix)):
        new_facet_matrix = vertices_matrix.copy()
        new_facet_matrix[i] = new_point
        list_of_facets.append(new_facet_matrix)
    return list_of_facets


# In[9]:


three_simplex = create_n_simplex(3)
vertices_matrix = three_simplex[:-1]
a_coeff = up_extension_constraint(vertices_matrix)
print(a_coeff)


# In[10]:


# Create a simple Linear classifier problem
# n = 3, m = 4, k = 5
n = 3 #dimension
m = 4 #number of A points
k = 5 #number of B points

# Random matrix A, B
np.random.seed(1012310)
A = np.random.random(m*n).reshape(m, n)
B = np.random.random(k*n).reshape(k, n)

print(A)
print(B)

print(n)


# In[11]:


w_hat = [0,0,0,0]
print(all(element == 0 for element in w_hat))


# In[12]:


# Create a new optimization problem
prob = xp.problem()
# Disable presolve
prob.controls.xslp_presolve = 0
prob.presolve()

# Read the problem from a file
# prob.read("problem.lp")

# Create variables
w = [xp.var(name="w{0}".format(i)) for i in range(n)]
gamma = xp.var(name = "gamma")

#A-point distance
y = [xp.var(name="y{0}".format(i), lb = 0) for i in range(m)]

#B-point distance
z = [xp.var(name="z{0}".format(i), lb = 0) for i in range(k)]

prob.addVariable(w, gamma, y, z) # figure out how to name variables
# so that we can see their name when we write the problem to file

# Add constraints
# prob.addConstraint(y[i] >= - xp.Dot(w, A[i]) + gamma for i in range(m))
# prob.addConstraint(z[j] >= xp.Dot(w, B[j]) + gamma for j in range(k))
prob.addConstraint(y[i] >= - sum(w[j]*A[i][j] for j in range(n)) + gamma for i in range(m))
prob.addConstraint(z[j] >= sum(w[i]*B[j][i] for i in range(n)) - gamma for j in range(k))

prob.addConstraint(sum(w[i]*w[i] for i in range(n)) >= 1)
#prob.addConstraint(sum(w[i] for i in range(n)) >= 0.1)

# set objective
prob.setObjective(sum(y)+sum(z))


# Define a callback function to be called when the relaxation has been solved
def Node_Callback(prob, vertices_facet):
    print("We entered a node callback")
    
    # Obtain node relaxation solution
    s = []
    prob.getlpsol(s, None, None, None)
    w_hat = s[0:n]
    print("Solution to the node relaxation w = ", w_hat)
    if CheckOnBall(w_hat):
        return 0
    else:
        # project the optimal point onto the 2-norm ball
        w_proj = ProjectOnBall(w_hat)
        # create new facets which each element contains vertices of facet
        list_of_facets = CreatNewFacets(vertices_facet, w_proj)
        return list_of_facets

def ChgBranchObj_Callback(prob, list_of_facets, w_hat):
    print("We entered the Change Branch Object Callback")
    # create branching obj
    # Check if it is on the root node
    if all(element == 0 for element in w_hat):
        # create the new object with n+1 empty branches
        bo = xp.branchobj(prob, isoriginal=True)
        bo.addbranches(n+1)
        # Add the constraint of the branching object
        for i in range(n+1):
            addrowzip(prob, bo, i, 'G', 1, [0,1,2], list_of_facets[i])
        print("BRANCHES AT THE ROOT NODE ARE CREATED")
    else:
        # create new object with n empty branches
        bo = xp.branchobj(prob, isoriginal=True)
        bo.addbranches(n)
        # Add constraints for each facet
        for i in range(n):
            addrowzip(prob, bo, i, 'G', 1, [0,1,2], list_of_facets[i])
        print("THE CONSTRAINTS ARE ADDED TO THE NODES")
        print("Number of branches = ", bo.getbranches())
    # set low priority value
    bo.setpriority(1)
    # store
    status = bo.store()
    return bo

# Disable presolved
prob.setControl({"presolve":0})
# Silence output
# prob.setControl ('outputlog', 0)
# Add the callback function for changing the branching object
prob.addcboptnode(Node_Callback, None, 2)
prob.addcbchgbranchobject(ChgBranchObj_Callback, None, 3)

print("------")

print("printing")
prob.write("myProblem", "lps")
# exit()

print("------")
print("solving")
print("------")


# Solve the problem
# prob.mipoptimize()
# prob.optimize()
prob.nlpoptimize()

# Print the solution status
print("Solution status:", prob.getProbStatusString())
# Print the optimal solution
print("Optimal solution:", prob.getSolution())
# Print the optimal objective value
print("Optimal objective value:", prob.getObjVal())

