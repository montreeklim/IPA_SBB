import xpress as xp
import numpy as np
from scipy.linalg import null_space

np.set_printoptions(precision=3, suppress=True)
xp.init('C:/Apps/Anaconda3/lib/site-packages/xpress/license/community-xpauth.xpr')

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
    if filename == None:
        # Read dat file to get M, N, K, a
        global M, N, K
        # M = #points
        # N = #dimensions
        # K = #hyperplane
        M = 60
        N = 2
        K = 2
        a = np.random.rand(M,N)*10
        print(a)
        # M, N, K, a = read_dat_file(filename)
        # print(M, N, K)
        # print(a)
        
        # Create problem for k-hyperplane clustering without norm constraints
        
        # Create variables
        w = xp.vars(K, N, name='w', lb=0)
        gamma = xp.vars(K, name="gamma", lb=0)
        x = xp.vars(M, K, name='x', vartype=xp.binary)
        y = xp.vars(M, name='y', lb=0)
        prob.addVariable(w, gamma, x, y)
        
        # Add constraints
        BigM = 10e+6
        for i in range(M):
            prob.addConstraint(sum(x[i,j] for j in range(K)) == 1)
            for j in range(K):
                prob.addConstraint(y[i] >= np.dot(w[j], a[i]) - gamma[j] - BigM*(1-x[i,j]))
                prob.addConstraint(y[i] >= np.dot(-w[j], a[i]) + gamma[j] - BigM*(1-x[i,j]))
        # Add norm constraints
        for j in range(K):
            prob.addConstraint(sum(w[j, i] * w[j, i] for i in range(N)) == 1)
                
        # set objective
        prob.setObjective(sum(y[i]*y[i] for i in range(M)), sense = xp.minimize)
        return prob
    return prob

if __name__ == '__main__':
    np.random.seed(123)
    tol = 1e-4
    prob = create_problem()
    prob.optimize('x')
    num_nodes = prob.attributes.nodes
    print("Number of nodes in the tree:", num_nodes)
    # print("Solution status:", prob.getProbStatusString())
    sol = prob.getSolution()
    print("Optimal objective value:", prob.getObjVal())