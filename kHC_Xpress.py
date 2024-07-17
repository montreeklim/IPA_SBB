import xpress as xp
import numpy as np
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=3, suppress=True)
xp.init('C:/Apps/Anaconda3/lib/site-packages/xpress/license/community-xpauth.xpr')
# xp.init('C:/Users/montr/anaconda3/Lib/site-packages/xpress/license/community-xpauth.xpr') # License path for Laptop

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
    # prob.controls.xslp_presolve = 0
    # prob.presolve()

    # Read the problem from a file
    if filename == None:
        # Read dat file to get M, N, K, a
        global M, N, K
        # M = #points
        # N = #dimensions
        # K = #hyperplane
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
        # scaler = StandardScaler()
        # a = scaler.fit_transform(a)
        
        # Compute bounds
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
            prob.addConstraint(sum(x[i,j] for j in range(min(i+1,K))) >= 1)
            for j in range(K):
                if j <= i:
                    prob.addConstraint(y[i] >= np.dot(w[j], a[i]) - gamma[j] - BigM*(1-x[i,j]))
                    prob.addConstraint(y[i] >= np.dot(-w[j], a[i]) + gamma[j] - BigM*(1-x[i,j]))
        
        # Add norm constraints
        for j in range(K):
            prob.addConstraint(sum(w[j, i] * w[j, i] for i in range(N)) >= 1)
                
        # set objective
        prob.setObjective(sum(y[i]*y[i] for i in range(M)), sense = xp.minimize)
        
    
        # Enable scaling
        prob.controls.scaling = 1
    
        # Adjust tolerances
        # prob.controls.lpiterlimit = 100000
        prob.controls.miprelstop = 1e-4
        prob.controls.mipabsstop = 1e-6
    
        # Use barrier method with higher precision 
        prob.controls.defaultalg = 4 # slower the search but more numerical stable
        
        # Model Refinement (L2 regularization)
        # lambda_reg = 0.1  # Choose an appropriate regularization strength
        # prob.setObjective(sum(y[i]*y[i] for i in range(M)) + lambda_reg * sum(w[j,i]**2 for j in range(K) for i in range(N)), sense=xp.minimize)
        return prob
    return prob

if __name__ == '__main__':
    # np.random.seed(123)
    tol = 1e-4
    prob = create_problem()
    prob.controls.timelimit=60
    # prob.setControl('feastol', 1e-3)

    prob.optimize('x')

    num_nodes = prob.attributes.nodes
    print("Number of nodes in the tree:", num_nodes)
    # print("Solution status:", prob.getProbStatusString())
    sol = prob.getSolution()
    print("Optimal objective value:", prob.getObjVal())