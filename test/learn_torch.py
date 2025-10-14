import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, in_channels, block_channel, use_layernorm=True):
        super(SimpleMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.mlp(x)
        print("x ", x)
        # x shape: (B, N, C) -> apply max pooling along N
        x = torch.max(x, dim=1)[0]
        return x

# ==== Demo 开始 ====

# 假设输入特征维度是 6，MLP 中间维度设置如下：
in_channels = 3
block_channel = [16, 32, 5]
use_layernorm = True

# 创建模型
model = SimpleMLP(in_channels, block_channel, use_layernorm)

# 构造一个模拟输入：batch size = 2，N = 5个点，in_channels = 6
x_input = torch.randn(1, 2, 3)  # shape: (B=2, N=5, C=6)
print("x input ", x_input)
# 执行前向传播
output = model(x_input)

# 打印输出结果
print("输入 shape:", x_input.shape)
print("输出 shape:", output.shape)
print("输出内容:\n", output)
