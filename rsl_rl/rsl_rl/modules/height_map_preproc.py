# rsl_rl/models/preproc_height_scan_1d.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class HeightScanPreproc1D(nn.Module):
    """
    输入:  obs_flat ∈ R^{B×(4+651)}，约定:
          obs[:, :pose_dim]           -> pose_command/state
          obs[:, pose_dim:pose_dim+L] -> height_scan (L=651)
    处理:  height_scan 先做 per-sample 标准化，再经过轻量 Conv1d 提特征；
          与 pose_command 拼接输出给 Actor 的 MLP。
    输出:  fused ∈ R^{B×(pose_dim + elev_out)}
    """
    def __init__(self, pose_dim: int = 4, height_len: int = 651, elev_out: int = 128):
        super().__init__()
        self.pose_dim   = pose_dim
        self.height_len = height_len

        # 1D-CNN：小而快
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # -> (B,64,1)
            nn.Flatten(),             # -> (B,64)
        )
        self.fc = nn.Sequential(nn.Linear(64, elev_out), nn.ReLU(inplace=True))

        # 输出维度给 ActorCritic 读取
        self.output_dim = pose_dim + elev_out

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        assert obs_flat.dim() == 2, "expect obs as (B, D)"
        B, D = obs_flat.shape
        assert D >= (self.pose_dim + self.height_len), \
            f"obs dim {D} < pose_dim+height_len ({self.pose_dim+self.height_len})"

        pose = obs_flat[:, :self.pose_dim]                         # (B, 4)
        scan = obs_flat[:, self.pose_dim:self.pose_dim + self.height_len]  # (B, 651)

        # --- 归一化：对每个样本沿长度维做 z-score（简单稳定） ---
        mean = scan.mean(dim=1, keepdim=True)
        std  = scan.std(dim=1, keepdim=True) + 1e-6
        scan_n = (scan - mean) / std                               # (B, 651)

        # --- 1D-CNN 特征 ---
        x = scan_n.unsqueeze(1).float()                            # (B,1,651)
        h = self.cnn(x)                                            # (B,64)
        h = self.fc(h)                                             # (B, elev_out)

        return torch.cat([pose, h], dim=1)                         # (B, pose_dim+elev_out)
