import xpress as xp
import numpy as np
import time
import pickle
import pandas as pd

def create_problem_defaults(n_planes = 3, dataset = None):
    """Read the given problem or create simple linear classifier problem."""
    prob = xp.problem()
    global M, N, K, a, BigM

    M = dataset.shape[0]
    N = dataset.shape[1]
    K = n_planes
    a = dataset

    # Calculate the absolute maximum value in the dataset
    h_abs = np.max(np.abs(dataset))

    # Max possible value for the dot product w * a
    # Since w is in [-1, 1], max dot product is N * h_abs
    max_dot_product = N * h_abs

    # Gamma bound (assuming you keep your original logic)
    gamma_bound = N * h_abs + h_abs * np.sqrt(N)

    # BigM must be greater than or equal to the max possible value of |w*a - gamma|
    BigM = max_dot_product + gamma_bound

    # Create variables using addVariables
    w = prob.addVariables(K, N, lb=-1, ub=1, name='w')
    gamma = prob.addVariables(K, lb=-gamma_bound, ub=gamma_bound, name='gamma')
    x = prob.addVariables(M, K, vartype=xp.binary, name='x')
    y = prob.addVariables(M, lb=0, name='y')

    # ---------------------------------------------------------
    # VARIABLES FOR (multi, 1) FORMULATION
    # ---------------------------------------------------------
    # L1 variables 
    w_plus = prob.addVariables(K, N, lb=0, ub=1, name='w_plus')
    w_minus = prob.addVariables(K, N, lb=0, ub=1, name='w_minus')
    s = prob.addVariables(K, N, vartype=xp.binary, name='s')
    
    # L_inf variables 
    u = prob.addVariables(K, N, vartype=xp.binary, name='u')

    # Add symmetry breaking on x
    for m in range(M):
        for k in range(K):
            if k > m:
                x[m, k].ub = 0

    # Add constraints (y and assignment)
    for i in range(M):
        prob.addConstraint(xp.Sum(x[i,j] for j in range(min(i+1,K))) == 1)
        for j in range(K):
            if j <= i:
                prob.addConstraint(
                    y[i] >= xp.Dot(w[j], a[i]) - gamma[j] - BigM*(1 - x[i,j])
                )
                prob.addConstraint(
                    y[i] >= xp.Dot(-w[j], a[i]) + gamma[j] - BigM*(1 - x[i,j])
                )

    # ---------------------------------------------------------
    # STRENGTHENED NORM CONSTRAINTS: k-HC_(2,1),(multi, 1)
    # ---------------------------------------------------------
    threshold = 1.0 / np.sqrt(N)

    for j in range(K):
        # Base 2-norm constraint 
        prob.addConstraint(xp.Sum(w[j, i] * w[j, i] for i in range(N)) >= 1)
        
        # L1-norm MILP constraints
        for i in range(N):
            prob.addConstraint(w_plus[j, i] - w_minus[j, i] == w[j, i])
            prob.addConstraint(w_plus[j, i] <= s[j, i])
            prob.addConstraint(w_minus[j, i] <= 1 - s[j, i])
        
        prob.addConstraint(xp.Sum(w_plus[j, i] + w_minus[j, i] for i in range(N)) >= 1)

        # L_inf-norm MILP constraints
        for i in range(N):
            # Big-M constraint for restricted disjunction. 
            # If u[j,i] == 1, w >= threshold. If u[j,i] == 0, w >= threshold - 2.
            prob.addConstraint(w[j, i] >= threshold - 2 * (1 - u[j, i]))
        
        prob.addConstraint(xp.Sum(u[j, i] for i in range(N)) == 1)

    # set objective
    prob.setObjective(xp.Sum(y[i]*y[i] for i in range(M)), sense=xp.minimize)
    
    return prob

if __name__ == '__main__':
    TIMELIMIT = 3600
    # TARGET = (22, 2, 2)  # (m, n, k) to benchmark

    datasets_filenames = [
        'test_instances.pkl',
        # "LowDim_low_noise.pkl",
        # "LowDim_medium_noise.pkl",
        # "LowDim_high_noise.pkl",
        # "HighDim_low_noise.pkl",
        # "HighDim_medium_noise.pkl",
        # "HighDim_high_noise.pkl",
    ]

    for pkl_filename in datasets_filenames:
        print(f"Processing {pkl_filename} …")
        with open(pkl_filename, "rb") as f:
            datasets_dict = pickle.load(f)
        print(f"  Loaded {len(datasets_dict)} instances\n")

        results = []
        outname = f"results_{pkl_filename[:-4]}_ICLR.xlsx"

        for idx, ((m, n, k), data_array) in enumerate(datasets_dict.items()):
            # if (m, n, k) != TARGET:
            #     continue
            
            prob = create_problem_defaults(n_planes=k, dataset=data_array)
            prob.controls.timelimit = TIMELIMIT

            start_time = time.time()
            prob.optimize('x')
            duration = round(time.time() - start_time, 3)

            nodes       = prob.attributes.nodes
            lower_bound = prob.attributes.bestbound
            upper_bound = prob.attributes.objval
            mipobjval   = prob.attributes.mipobjval
            mip_status  = prob.attributes.mipstatus

            print(f"  Instance {idx} (m={m},n={n},k={k}):")
            print(f"    Nodes:       {nodes}")
            print(f"    Time:        {duration}s")
            print(f"    mipobjval:   {mipobjval:.6f}  mipbestobjval: {upper_bound:.6f}")
            print(f"    Lower Bound: {lower_bound:.6f}")
            print(f"    Upper Bound: {upper_bound:.6f}")
            print(f"    MIP status:  {mip_status}")

            if mip_status != xp.mip_optimal or upper_bound < 1e-9:
                print(f"    -> Skipped (status={mip_status}, UB={upper_bound:.6g})\n")
                continue

            results.append({
                "m": m, "n": n, "k": k,
                "Nodes": nodes,
                "Time": duration,
                "LowerBound": lower_bound,
                "UpperBound": upper_bound,
            })

            pd.DataFrame(results).to_excel(outname, index=False)
            print(f"    -> Saved {len(results)} result(s) to {outname}")

        print(f"\n  Done — {len(results)} instance(s) written to {outname}\n")