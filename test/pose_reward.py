from math import tanh
import numpy as np
import matplotlib.pyplot as plt

# 定义距离范围 d ∈ [0,15] 米
d = np.linspace(0, 30, 100)

# 定义不同的衰减系数 k
k_values = [2, 0.5, 3, 5, 10, 20]

# 绘制曲线
plt.figure(figsize=(10, 6))
# 1 - torch.tanh(distance / std)

for k in k_values:
    # r = 2 * np.exp(-k * d**2)
    r = 1 - np.tanh(d/k)
    plt.plot(d, r, label=f'k={k}')



# 美化图形
plt.title('Exponential Reward Decay: $r = 2 \cdot e^{-k \cdot d}$')
plt.xlabel('Distance (m)')
plt.ylabel('Reward')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.ylim(0, 2.2)
plt.show()