import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ========== 被测类 ==========
class _HeightScanPreproc1D(nn.Module):
    """将 (pose[4], height_scan[651]) -> (pose[4] + cnn_feat[elev_out])"""
    def __init__(self, pose_dim: int = 4, height_len: int = 651, elev_out: int = 128):
        super().__init__()
        self.pose_dim   = pose_dim
        self.height_len = height_len
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # -> (B,64,1)
            nn.Flatten(),             # -> (B,64)
        )
        self.fc = nn.Sequential(nn.Linear(64, elev_out), nn.ReLU(inplace=True))
        print(f"cnn MLP: {self.cnn}")
        
        print(f"fc MLP: {self.fc}")
        
        self.output_dim = pose_dim + elev_out

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        pose = obs_flat[:, :self.pose_dim]  # (B,4)
        scan = obs_flat[:, self.pose_dim:self.pose_dim + self.height_len]  # (B,651)

        # per-sample z-score
        mean = scan.mean(dim=1, keepdim=True)
        std  = scan.std(dim=1, keepdim=True) + 1e-6
        scan_n = (scan - mean) / std

        # CNN features
        h = self.cnn(scan_n.unsqueeze(1).float())
        h = self.fc(h)

        return torch.cat([pose, h], dim=1)  # (B, 4+elev_out)


# ========== 测试代码 ==========
if __name__ == "__main__":
    torch.manual_seed(0)

    BATCH = 1
    POSE_DIM = 4
    HEIGHT_LEN = 20
    ELEV_OUT = 12

    # 假数据
    pose = torch.randn(BATCH, POSE_DIM)
    x = torch.linspace(0, 2*np.pi, HEIGHT_LEN)
    base = torch.sin(2*x)[None, :] + 0.3*torch.cos(5*x)[None, :]
    noise = 0.1 * torch.randn(BATCH, HEIGHT_LEN)
    height_scan = base.repeat(BATCH, 1) + noise

    obs_flat = torch.cat([pose, height_scan], dim=1)

    # 实例化预处理器并前向
    pre = _HeightScanPreproc1D(pose_dim=POSE_DIM, height_len=HEIGHT_LEN, elev_out=ELEV_OUT)
    with torch.no_grad():
        fused = pre(obs_flat)

    print("输入 shape:", tuple(obs_flat.shape))       # (8, 655)
    print("输出 shape:", tuple(fused.shape))          # (8, 132)
    print("期望输出维度:", POSE_DIM + ELEV_OUT)

    # 检查归一化效果
    scan_mean = height_scan.mean(dim=1)
    scan_std  = height_scan.std(dim=1)
    scan_n = (height_scan - scan_mean[:, None]) / (scan_std[:, None] + 1e-6)
    print("\n原始 height_scan: mean≈", scan_mean[:3].numpy(), "std≈", scan_std[:3].numpy())
    print("归一化后: mean≈", scan_n.mean(dim=1)[:3].numpy(), "std≈", scan_n.std(dim=1)[:3].numpy())

    # 可视化一条样本的归一化效果
    i = 0
    plt.figure(figsize=(8,4))
    plt.plot(height_scan[i].numpy(), label="raw scan")
    plt.plot(scan_n[i].numpy(), label="normalized scan", linestyle="--")
    plt.title("Height scan (sample 0): raw vs normalized")
    plt.xlabel("index")
    plt.ylabel("value")
    plt.legend()
    plt.grid(True)
    plt.show()
