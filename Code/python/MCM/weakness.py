import numpy as np
import matplotlib.pyplot as plt
import math

def simulate_population(T=70, fixed_male_ratio=False):
    R = np.zeros(T)
    R[0] = 150  
    r_R = 0.4 
    c_R = 0.1 
    K_R = 200  

    P = np.zeros(T)
    P[0] = 150  
    r_P = 0.1 
    K_P = 200  

    M = np.zeros(T)
    M[0] = 0.6 
    B_F = 10
    for t in range(1, T):
        R[t] = R[t-1] + r_R * R[t-1] * (1 - R[t-1] / K_R) - c_R * P[t-1]
        R[t] = max(R[t], 1)
        r_l = 1 - R[t] / K_R
        if fixed_male_ratio:
            M[t] = 0.5
        else:
            M[t] = math.sin(M[t-1] + (0.22 * r_l))
        E_B = r_P * (1 - M[t]) * B_F
        P[t] = P[t-1] + E_B * P[t-1] * (1 - P[t-1] / K_P) * (R[t] / K_R) - 20 * P[t] / R[t] - 2 * P[t-1]/R[t-1]
        P[t] = max(P[t], 0)
 
    return P
P_dynamic = simulate_population(fixed_male_ratio=False)
P_fixed = simulate_population(fixed_male_ratio=True)
plt.figure(figsize=(10, 6))
plt.plot(P_dynamic, label='Dynamic Male Ratio', linestyle='-', color='green')
plt.plot(P_fixed, label='Fixed Male Ratio (0.5)', linestyle='--', color='red')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Population Size', fontsize=14)
plt.title('Population Size Over Time for Different Male Ratios(P[0]=150,R[0]=150)')
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray')
plt.tight_layout()
plt.show()
