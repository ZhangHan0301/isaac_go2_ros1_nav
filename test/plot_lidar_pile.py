import json
import math
from turtle import color
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ---------- 四元数工具（保持 w,x,y,z） ----------
def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    """四元数共轭（单位四元数时 = 逆）; q: (...,4)=(w,x,y,z)"""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """用四元数 q 旋转向量 v（完全向量化）"""
    w, x, y, z = q.unbind(-1)
    qv  = torch.stack([x, y, z], dim=-1)
    uv  = torch.cross(qv, v, dim=-1)
    uuv = torch.cross(qv, uv, dim=-1)
    return v + 2.0 * (w.unsqueeze(-1) * uv + uuv)

# ---------- 可视化主流程 ----------
def visualize_from_json(json_path: str, ground_threshold: float = 0.1, num_sectors: int = 13, env_ids: int=0):
    # 读取 JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # 转为 torch 张量（CPU 即可）
    points_w = torch.tensor(data["points_w"], dtype=torch.float32)  # (E,R,3)
    pos_w    = torch.tensor(data["pos_w"],   dtype=torch.float32)  # (E,3)
    quat_w   = torch.tensor(data["quat_w"],  dtype=torch.float32)  # (E,4)

    # 仅使用第 0 个环境（文件里就是单环境）
    points_w0 = points_w[env_ids]         # (R,3)
    robot_pos = pos_w[env_ids]            # (3,)
    robot_q   = quat_w[env_ids]           # (4,)

    # -------- 世界 -> 局部：先平移，再乘 q^{-1} --------
    rel = points_w0 - robot_pos[None, :]           # (R,3)
    q_inv = _quat_conj(robot_q)[None, :].expand(rel.shape[0], 4)
    points_local = _quat_rotate(q_inv, rel)        # (R,3)

    # 地面阈值过滤（在局部系）
    valid = points_local[:, 2] > ground_threshold
    pts_local_valid = points_local[valid]          # (K,3)

    # 极坐标（局部 XY）
    xy = pts_local_valid[:, :2]                    # (K,2)
    r  = torch.linalg.norm(xy, dim=-1)             # (K,)
    theta = torch.atan2(xy[:, 1], xy[:, 0])        # (K,)
    theta = (theta + 2*math.pi) % (2*math.pi)      # [0,2π)

    # 扇区划分 & 每扇区最小距离
    seg_angle = 2 * math.pi / num_sectors
    seg = torch.floor(theta / seg_angle).clamp(max=num_sectors-1).long()  # (K,)
    r_min = torch.full((num_sectors,), float("5.0"), dtype=r.dtype)
    # 用 scatter_reduce 更快（若版本不支持就回退循环）
    if hasattr(r_min, "scatter_reduce"):
        r_min = r_min.scatter_reduce(0, seg, r, reduce="amin", include_self=True)
        print("r min ", r_min)
    else:
        for k in range(num_sectors):
            mk = (seg == k)
            if mk.any():
                r_min[k] = torch.min(r[mk])

    # ========== 画图 ==========
    fig = plt.figure(figsize=(13, 6))

    # ---- 左图：世界点云 + 世界轴 + 机器人轴 ----
    ax = fig.add_subplot(121, projection='3d')
    pw = points_w0.numpy()
    ax.scatter(pw[:, 0], pw[:, 1], pw[:, 2], s=30, alpha=0.8, label='Points (World)')

    # 世界轴（原点）
    Lw = 0.5
    ax.quiver(0, 0, 0, Lw, 0, 0, color='r'); ax.text(Lw,0,0,'X', color='r')  # 所有箭头都使用红色
    ax.quiver(0, 0, 0, 0, Lw, 0, color='g'); ax.text(0,Lw,0,'Y', color='g') # Y 轴使用绿色
    ax.quiver(0, 0, 0, 0, 0, Lw, color='b'); ax.text(0,0,Lw,'Z', color='b') # Z 轴使用蓝色

    # 机器人位置
    rp = robot_pos.numpy()
    ax.scatter(rp[0], rp[1], rp[2], c='k', s=50, marker='o', label='Robot pos')

    # 机器人轴（把局部单位轴旋到世界方向，从机器人位置出发）
    axes_local = torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=torch.float32)  # (3,3)
    rq = robot_q[None, :].expand(3, 4)                                         # (3,4)
    axes_world = _quat_rotate(rq, axes_local)                                  # (3,3)

    Lr = 0.35
    ax.quiver(rp[0], rp[1], rp[2], axes_world[0,0].item(), axes_world[0,1].item(), axes_world[0,2].item(), length=Lr, normalize=True,color='r')  # X'
    ax.quiver(rp[0], rp[1], rp[2], axes_world[1,0].item(), axes_world[1,1].item(), axes_world[1,2].item(), length=Lr, normalize=True,color='g')  # Y'
    ax.quiver(rp[0], rp[1], rp[2], axes_world[2,0].item(), axes_world[2,1].item(), axes_world[2,2].item(), length=Lr, normalize=True,color='b')  # Z'

    ax.set_title("World frame: points + world axes + robot axes")
    ax.set_xlabel("X (world)"); ax.set_ylabel("Y (world)"); ax.set_zlabel("Z (world)")
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=22, azim=45)

    # ---- 右图：局部 XY + 扇区边界 + 最小距离线 ----
    ax2 = fig.add_subplot(122)
    pl = pts_local_valid.numpy()
    if pl.shape[0] > 0:
        ax2.scatter(pl[:, 0], pl[:, 1], s=35, alpha=0.9, label='Points (robot XY)')
        # 每个点的极坐标射线
        for i in range(pl.shape[0]):
            ax2.plot([0, pl[i,0]], [0, pl[i,1]], linestyle='--', linewidth=0.8, alpha=0.5)
        # 扇区边界
        r_np = r.numpy()
        r_max = float(np.percentile(r_np, 95)) if r_np.size > 0 else 1.0
        for k in range(num_sectors):
            ang = k * seg_angle
            ax2.plot([0, r_max*math.cos(ang)], [0, r_max*math.sin(ang)], linewidth=1, alpha=0.5)
        # 每扇区最小距离线（画在扇区中心角）
        rmin_np = r_min.numpy()
        for k in range(num_sectors):
            if math.isfinite(float(rmin_np[k])):
                ang_c = (k + 0.5) * seg_angle
                xe = rmin_np[k] * math.cos(ang_c)
                ye = rmin_np[k] * math.sin(ang_c)
                ax2.plot([0, xe], [0, ye], linewidth=2.2)
                ax2.text(xe, ye, f"{rmin_np[k]:.2f}", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No valid points (z <= threshold).", ha='center', va='center', transform=ax2.transAxes)

    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title(f"Robot local XY (z>{ground_threshold}), {num_sectors} sectors\nwith per-sector min range")
    ax2.set_xlabel("x (robot)"); ax2.set_ylabel("y (robot)")
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=8)

  
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 把你贴的 JSON 内容保存成文件：lidar_data.json
    # 然后运行本脚本：
    visualize_from_json("data/lidar_data.json", ground_threshold=0.0, num_sectors=13)
