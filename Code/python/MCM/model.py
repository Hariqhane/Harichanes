
import numpy as np
import matplotlib.pyplot as plt
import math
T = 70

R = np.zeros(T)
R[0] = 100  
r_R = 0.4  
c_R = 0.1  
K_R = 200 

# 种群大小初始化及参数设置
P = np.zeros(T)
P[0] = 100  # 初始种群大小
r_P = 0.1  # 基础繁殖率
K_P = 200  # 种群的环境承载量

# 雄性比例初始化及调整策略c_R
M = np.zeros(T)
M[0] = 0.6  # 开始时较高的雄性比例
B_F = 10


# 根据资源量调整雄性比例，资源量越少，雄性比例越高
for t in range(1, T):
    # 缓和资源量的变化
    R[t] = R[t-1] + r_R * R[t-1] * (1 - R[t-1] / K_R) - c_R * P[t-1]
    R[t] = max(R[t], 2)  # 确保资源量不为负

    # 更新雄性比例，使其受到资源量的影响
    # 资源量越少，雄性比例越高
    r_l = 1 - R[t] / K_R

    M[t] =math.sin( M[t-1] +(0.22* r_l))
    print(M[t])
    # print(r_l)  # 根据资源缺口调整雄性比例

    # M[t] = np.clip(M[t], 0.56, 0.78)  # 保持雄性比例在56%到78%之间
    # M[t] = 0.5
    # 更新种群大小
    E_B = r_P * (1 - M[t]) * B_F
    P[t] = P[t-1] + E_B * P[t-1] * (1 - P[t-1] / K_P) * (R[t] / K_R)- 20  * P[t] / R[t] - 2 * P[t-1]/R[t-1]
    P[t] = max(P[t], 2)

fig, ax1 = plt.subplots(figsize=(10, 6))

# 设定颜色和线条样式
color_R = 'tab:blue'
color_P = 'tab:green'
color_M = 'tab:red'

ax1.set_xlabel('Time', fontsize=14)
ax1.set_ylabel('Resource Level and Population Size', fontsize=14,color=color_P)
line1, = ax1.plot(R[1:], label='Resource Level', color=color_R, linestyle='-', linewidth=2)
line3, = ax1.plot(P[1:], label='Population Size', color=color_P,linestyle='-', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color_R, labelsize=12)
ax1.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray')
ax1.legend(loc='lower right', fontsize=12)

# 第二个坐标轴
ax2 = ax1.twinx()
ax2.set_ylabel('Male Ratio', color=color_M, fontsize=14)
line2, = ax2.plot(M[1:], label='Male Ratio', color=color_M, linestyle='--', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color_M, labelsize=12)

# 合并图例
lines = [line1, line3, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower right', fontsize=12)

fig.tight_layout()  # 调整布局
plt.show()