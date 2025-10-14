import torch
import torch.nn as nn
from torch.distributions import Normal

# 一个简单的 DemoPolicy：只学动作均值 (mean)，标准差 (std) 是 nn.Parameter
class DemoPolicy(nn.Module):
    def __init__(self, num_actions, init_noise_std=1.0):
        super().__init__()
        # 这里 std 是一个可训练参数，初始值是 init_noise_std
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        # 简单起见：用一个线性层预测动作均值
        self.fc = nn.Linear(4, num_actions)  # 输入维度=4

    def forward(self, obs):
        mean = self.fc(obs)  # 根据输入计算动作均值
        # 扩展 std 的形状，让它和 mean 一样大
        std = self.std.expand_as(mean)
        dist = Normal(mean, std)  # 高斯分布
        return dist

# ========== 使用 DemoPolicy ==========
torch.manual_seed(0)
policy = DemoPolicy(num_actions=2, init_noise_std=0.5)

# 假设有一条观测
obs = torch.randn(1, 4)

# 前向，得到分布
dist = policy(obs)

print("均值 mean:", dist.mean.detach().numpy())
print("标准差 std:", dist.stddev.detach().numpy())

# 采样动作
for i in range(3):
    action = dist.sample()
    print(f"采样动作 {i+1}:", action.detach().numpy())

# 看看 std 是否真的是一个“可训练参数”
print("\nstd 参数:", list(policy.parameters())[0])
