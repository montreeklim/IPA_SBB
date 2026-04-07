import xpress as xp
import numpy as np

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
        
        prob.addConstraint(sum(w[i]*w[i] for i in range(n)) >= 1)

        # set objective
        prob.setObjective(sum(y)+sum(z))
        
        # Disable presolved
        prob.setControl({"presolve":0})
        # Silence output
        # prob.setControl ('outputlog', 0)
    
    return prob

if __name__ == '__main__':
    prob = create_problem()
    prob.optimize('x')
    print("Solution status:", prob.getProbStatusString())
    print("Optimal solution:", prob.getSolution())
    print("Optimal objective value:", prob.getObjVal())
    print("Solver Status:", prob.getProbStatus())
    