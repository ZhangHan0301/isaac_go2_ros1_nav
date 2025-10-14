import torch
import math
import time
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for old mpl

# ==================== 四元数工具（保持不变） ====================
def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    """四元数共轭/逆（单位四元数时相同），q: (...,4)=(w,x,y,z)"""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    用四元数 q 旋转向量 v（完全向量化）
    q: (...,4)=(w,x,y,z) 单位四元数（表示 从局部->世界 的旋转）
    v: (...,3)
    return: (...,3) 旋转后的向量
    """
    w, x, y, z = q.unbind(-1)
    qv  = torch.stack([x, y, z], dim=-1)           # (...,3)
    uv  = torch.cross(qv, v, dim=-1)               # (...,3)
    uuv = torch.cross(qv, uv, dim=-1)              # (...,3)
    return v + 2.0 * (w.unsqueeze(-1) * uv + uuv)

# ==================== 世界→局部（用通用四元数，保持不变） ====================
def world_to_robot_local(points_w: torch.Tensor,
                         robot_pos_w: torch.Tensor,
                         robot_quat_wxyz: torch.Tensor) -> torch.Tensor:
    """
    points_w: (N,3) 世界坐标
    robot_pos_w: (3,)
    robot_quat_wxyz: (4,)=(w,x,y,z)，从“机器人局部->世界”的旋转
    return: (N,3) 机器人局部坐标
    """
    rel_w = points_w - robot_pos_w[None, :]              # 平移
    q_inv = _quat_conj(robot_quat_wxyz)[None, :]         # (1,4)
    q_inv_exp = q_inv.expand(points_w.shape[0], 4)       # (N,4)
    pts_local = _quat_rotate(q_inv_exp, rel_w)           # (N,3)
    return pts_local

# ==================== 主测试（仅改“位姿生成”为 yaw-only） ====================
def test_lidar_rotation_yaw_only_gen():
    # 配置
    num_points       = 5
    ground_threshold = 0.1
    num_sectors      = 36
    sector_angle     = 2 * math.pi / num_sectors

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- 数据：世界系点云 ----------
    points_w = torch.rand((num_points, 3), dtype=torch.float32, device=device)

    # ---------- 机器人位置（世界系） ----------
    robot_pos_w = torch.rand((3,), dtype=torch.float32, device=device)
    robot_pos_w[2]=0.0
    # ---------- 方式 A：仅绕 z 轴的 yaw -> 构造单位四元数 (w,0,0,z) ----------
    yaw = torch.tensor(math.pi / 2, dtype=torch.float32)  # 标量 (-pi, pi]
    half = 0.5 * yaw
    robot_quat_wxyz = torch.stack([
        torch.cos(half),            # w
        torch.zeros_like(half),     # x = 0
        torch.zeros_like(half),     # y = 0
        torch.sin(half)             # z
    ]).cuda()  # (4,)

    # ---------- 世界 -> 机器人局部（仍用“通用四元数”函数） ----------
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    points_local = world_to_robot_local(points_w, robot_pos_w, robot_quat_wxyz)
    torch.cuda.synchronize() if device == "cuda" else None
    print(f"World->Local time (yaw-only quat, generic rotate): {time.time()-t0:.6f}s")

    # ---------- 地面过滤（在局部系：z > threshold） ----------
    valid = points_local[:, 2] > ground_threshold
    pts_local_valid = points_local[valid]  # (K,3)

    # ---------- 极坐标（机器人局部 XY） ----------
    xy = pts_local_valid[:, :2]                     # (K,2)
    r  = torch.linalg.norm(xy, dim=-1)              # (K,)
    theta = torch.atan2(xy[:, 1], xy[:, 0])         # (K,)
    theta = (theta + 2 * math.pi) % (2 * math.pi)   # [0, 2π)
    sectors = torch.floor(theta / sector_angle).clamp(max=num_sectors - 1).long()  # (K,)

    # ---------- 每个扇区最小距离 ----------
    # 用 scatter_reduce 更高效；老版本无此 API 时回退到循环
    r_min = torch.full((num_sectors,), float('inf'), dtype=r.dtype, device=r.device)
    if hasattr(r_min, "scatter_reduce"):
        r_min = r_min.scatter_reduce(0, sectors, r, reduce="amin", include_self=True)
    else:
        # fallback：简明循环
        for k in range(num_sectors):
            mask_k = (sectors == k)
            if mask_k.any():
                r_min[k] = torch.min(r[mask_k])

    # ==================== 可视化 ====================
    fig = plt.figure(figsize=(13, 6))

    # ===== 左图：世界点云 + 世界坐标轴 + 机器人坐标轴（由 yaw-only 四元数生成） =====
    ax = fig.add_subplot(121, projection='3d')

    pw = points_w.detach().cpu().numpy()
    ax.scatter(pw[:, 0], pw[:, 1], pw[:, 2], s=6, c='tab:blue', alpha=0.7, label='Points (World)')

    # 世界坐标轴（从原点）
    axis_len_world = 0.5
    ax.quiver(0,0,0, axis_len_world,0,0, color='r', linewidth=2, label='World X')
    ax.quiver(0,0,0, 0,axis_len_world,0, color='g', linewidth=2, label='World Y')
    ax.quiver(0,0,0, 0,0,axis_len_world, color='b', linewidth=2, label='World Z')

    # 机器人位置
    rp = robot_pos_w.detach().cpu().numpy()
    ax.scatter(rp[0], rp[1], rp[2], c='k', s=30, marker='o', label='Robot position')

    # 用 yaw-only 四元数得到机器人坐标轴在世界中的方向
    cy = float(torch.cos(yaw).item())
    sy = float(torch.sin(yaw).item())
    xw = np.array([ cy, sy, 0.0])  # X'
    yw = np.array([-sy, cy, 0.0])  # Y'
    zw = np.array([0.0, 0.0, 1.0]) # Z'

    axis_len_robot = 0.35
    ax.quiver(rp[0], rp[1], rp[2], xw[0], xw[1], xw[2], color='r', length=axis_len_robot, normalize=True, linewidth=2, label="Robot X'")
    ax.quiver(rp[0], rp[1], rp[2], yw[0], yw[1], yw[2], color='g', length=axis_len_robot, normalize=True, linewidth=2, label="Robot Y'")
    ax.quiver(rp[0], rp[1], rp[2], zw[0], zw[1], zw[2], color='b', length=axis_len_robot, normalize=True, linewidth=2, label="Robot Z'")

    ax.set_title("World frame: points + World axes + Robot axes (yaw only)")
    ax.set_xlabel('X (world)')
    ax.set_ylabel('Y (world)')
    ax.set_zlabel('Z (world)')
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=22, azim=45)

    # ===== 右图：机器人局部 XY 的 2D 极坐标（含最小距离线） =====
    ax2 = fig.add_subplot(122)
    pl = pts_local_valid.detach().cpu().numpy()
    ax2.scatter(pl[:, 0], pl[:, 1], s=14, c='tab:red', alpha=0.85, label='Points (Robot XY)')

    # 极坐标射线（每个点）
    for i in range(pl.shape[0]):
        ax2.plot([0.0, pl[i, 0]], [0.0, pl[i, 1]], linestyle='--', linewidth=0.6, color='gray', alpha=0.4)

    # 扇区边界
    r_np = r.detach().cpu().numpy()
    r_max = float(np.percentile(r_np, 95)) if pl.shape[0] > 0 else 1.0
    for k in range(num_sectors):
        ang = k * sector_angle
        ax2.plot([0, r_max * math.cos(ang)], [0, r_max * math.sin(ang)],
                 color='0.3', linewidth=1, alpha=0.5)

    # ✅ 每个扇形区域的“最小距离线”（画在扇区中心角上）
    rmin_np = r_min.detach().cpu().numpy()
    for k in range(num_sectors):
        if math.isfinite(rmin_np[k]):  # 该扇区存在有效点
            ang_c = (k + 0.5) * sector_angle
            x_end = rmin_np[k] * math.cos(ang_c)
            y_end = rmin_np[k] * math.sin(ang_c)
            ax2.plot([0, x_end], [0, y_end], color='tab:green', linewidth=2.0, alpha=0.9)
            # 也可标注数值（可选）
            ax2.text(x_end, y_end, f"{rmin_np[k]:.2f}", fontsize=7, color='tab:green')

    # 画出机器人 X' 轴（局部）
    ax2.plot([0, r_max], [0, 0], color='k', linewidth=1.2, alpha=0.7, label="Robot X'")

    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title(f'Robot local XY (z>{ground_threshold}) with {num_sectors} sectors\nand per-sector min range')
    ax2.set_xlabel('x (robot)')
    ax2.set_ylabel('y (robot)')
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()

# ============ 读取部分 ============

def load_frames(path):
    with open(path, "r") as f:
        data = json.load(f)

    # 判断是单帧还是多帧
    if isinstance(data, dict):
        data = [data]  # 转成 list，方便统一处理

    frames = []
    for d in data:
        frames.append({
            "points_w": torch.tensor(d["points_w"], dtype=torch.float32),
            "pos_w": torch.tensor(d["pos_w"], dtype=torch.float32),
            "quat_w": torch.tensor(d["quat_w"], dtype=torch.float32),
        })
    
    print("frame ", frames[0]["points_w"])
    return frames

# ==================== 运行 ====================
if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    frames = load_frames("data/lidar_data.json")
    
    test_lidar_rotation_yaw_only_gen()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"Total wall time: {time.time() - t0:.3f}s")
