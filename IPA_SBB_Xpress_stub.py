from __future__ import print_function

import xpress as xp
import numpy as np
from scipy.linalg import null_space

def CheckOnBall(w):
    tol = 1e-10
    if np.linalg.norm(w) <= 1-tol:
        return False
    else:
        return True

def ProjectOnBall(w):
    return w/np.linalg.norm(w)

def create_problem(filename = None):

    # Create a new optimization problem
    prob = xp.problem()
    # Disable presolve
    # prob.controls.xslp_presolve = 0
    # prob.presolve()

    # Read the problem from a file
    if filename != None:
        prob.read(filename)
    else:
        # Create a simple Linear classifier problem
        # n = 3, m = 4, k = 5
        global n 
        n = 3 #dimension
        m = 4 #number of A points
        k = 5 #number of B points

        # Random matrix A, B
        np.random.seed(1012310)
        A = np.random.random(m*n).reshape(m, n)
        B = np.random.random(k*n).reshape(k, n)
        
        # Create variables
        w = [xp.var(name="w{0}".format(i)) for i in range(n)]
        gamma = xp.var(name = "gamma")

        #A-point distance
        y = [xp.var(name="y{0}".format(i), lb = 0) for i in range(m)]

        #B-point distance
        z = [xp.var(name="z{0}".format(i), lb = 0) for i in range(k)]

        prob.addVariable(w, gamma, y, z) 

        # Add constraints
        prob.addConstraint(y[i] >= - sum(w[j]*A[i][j] for j in range(n)) + gamma for i in range(m))
        prob.addConstraint(z[j] >= sum(w[i]*B[j][i] for i in range(n)) - gamma for j in range(k))

        # prob.addConstraint(sum(w[i]*w[i] for i in range(n)) >= 1)
        # prob.addConstraint(sum(w[i] for i in range(n)) >= 0.1)

        # set objective
        prob.setObjective(sum(y)+sum(z))
        
        # Disable presolved
        # prob.setControl({"presolve":0})
        # Silence output
        # prob.setControl ('outputlog', 0)
    
    return prob

def cbchecksol(prob, aux, soltype, cutoff):
    print('We are in the preintsol callback.')
    # if (prob.attributes.presolvestate & 128) == 0:
    #     return (1, 0)
    
    sol = []

    # Retrieve node solution
    try:
        prob.getlpsol(x=sol)
    except:
        return (1, cutoff)

    sol = np.array(sol)
    print(sol)

    # Check if the norm is 1
    # If so, we have a feasible solution

    refuse = 0 if CheckOnBall(sol) else 1

    # Return with refuse != 0 if solution is rejected, 0 otherwise;
    # and same cutoff
    return (refuse, cutoff)

def solveprob(prob):

    aux = []

    prob.addcbpreintsol(cbchecksol, aux, 1)
    # p.addcboptnode(cbaddcuts, 3)
    # prob.addcbchgbranchobject(cbbranch, 1)

    print("aaaa")
    prob.mipoptimize()
    
    print("Solution status:", prob.getProbStatusString())
    print("Optimal solution:", prob.getSolution())
    print("Optimal objective value:", prob.getObjVal())
    print("Solver Status:", prob.getProbStatus())
    
    
if __name__ == '__main__':
    prob = create_problem(filename = None)
    prob.write("simple_prob.lp")
    solveprob(prob)


