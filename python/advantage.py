import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

def simulate_minimum_resource_for_P0_range(P0_range, fixed_male_ratio=False):
    T = 70
    r_R = 0.4
    c_R = 0.1
    K_R = 200
    r_P = 0.1
    K_P = 200
    B_F = 10
    initial_R_values = np.arange(1, 150, 1)  
    results = []
    for P0 in P0_range:
        for initial_R in initial_R_values:
            R = np.zeros(T)
            R[0] = initial_R
            P = np.zeros(T)
            P[0] = P0
            M = np.zeros(T)
            M[0] = 0.6
            for t in range(1, T):
                R[t] = R[t-1] + r_R * R[t-1] * (1 - R[t-1] / K_R) - c_R * P[t-1]
                R[t] = max(R[t], 1)
                r_l = 1 - R[t] / K_R
                if fixed_male_ratio:
                    M[t] = 0.5
                else:
                    M[t] = math.sin(M[t-1] + (0.22 * r_l))
                E_B = r_P * (1 - M[t]) * B_F
                P[t] = P[t-1] + E_B * P[t-1] * (1 - P[t-1] / K_P) * (R[t] / K_R) - 20 * P[t] / R[t] - 2 * P[t-1] / R[t-1]
                P[t] = max(P[t], 0)
                if R[t] <= 0:
                    break
            if all(P > 0):
                results.append((P0, initial_R))
                break

    return pd.DataFrame(results, columns=['Initial_Population', 'Minimum_Initial_Resource'])

P0_range = range(1, 201)
results_dynamic = simulate_minimum_resource_for_P0_range(P0_range, fixed_male_ratio=False)
results_fixed = simulate_minimum_resource_for_P0_range(P0_range, fixed_male_ratio=True)
results = pd.merge(results_dynamic, results_fixed, on="Initial_Population", suffixes=('_Dynamic', '_Fixed'))

csv_path = "C:/Users/IK/Desktop/minimum_initial_resources.csv"
results.to_csv(csv_path, index=False)

csv_path
