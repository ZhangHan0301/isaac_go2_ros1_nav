import torch
from torch import nn

# ===== 直接用你给的 EmpiricalNormalization 类 =====
class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        if self.training:
            print("---------")
            self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count

        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


# ========== 使用 DEMO ==========

# 假设输入维度是 3
normalizer = EmpiricalNormalization(shape=(3,),until=1.0e8)

# 模拟几批观测数据
batches = [
    torch.tensor([[1.0, 2.0, 3.0],
                  [2.0, 4.0, 6.0]]),
    torch.tensor([[3.0, 6.0, 9.0],
                  [4.0, 8.0, 12.0]]),
    torch.tensor([[5.0, 10.0, 15.0]])
]

print("=== EmpiricalNormalization Demo ===")
for i, batch in enumerate(batches, 1):
    normed = normalizer(batch)  # 前向会更新统计量
    print(f"\n[批次 {i}] 原始数据:\n{batch}")
    print(f"[批次 {i}] 归一化结果:\n{normed}")
    print(f"[批次 {i}] 当前均值: {normalizer.mean}")
    print(f"[批次 {i}] 当前标准差: {normalizer.std}")

    # 验证 inverse 可以恢复
    restored = normalizer.inverse(normed)
    print(f"[批次 {i}] 恢复后的数据:\n{restored}")
