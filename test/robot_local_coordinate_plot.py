import json
from turtle import color
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 读取 JSON 文件
with open('data/lidar_data.json', 'r') as json_file:
    data = json.load(json_file)

# 提取数据
points_w0 = torch.tensor(data['points_w'], dtype=torch.float32)  # [env_nums, N, 3]
robot_pos = torch.tensor(data['pos_w'], dtype=torch.float32)   # [env_nums, 3]
robot_q = torch.tensor(data['quat_w'], dtype=torch.float32)    # [env_nums, 4]

# 打印数据验证
print(f"Points_w0: {points_w0}")
print(f"Robot Position: {robot_pos}")
print(f"Robot Quaternion: {robot_q}")

# 批量处理并绘图
def plot_envs_batched(points_w0: torch.Tensor,
                      robot_pos: torch.Tensor,
                      robot_q: torch.Tensor,
                      env_names=None):
    """
    使用 torch 批量计算，一次性绘制多个环境的点与机器人局部坐标系（2D）。
    points_w0: [env_nums, N, 3]
    robot_pos: [env_nums, 3]
    robot_q:   [env_nums, 4] = (w,x,y,z)
    """
    assert points_w0.ndim == 3 and points_w0.shape[-1] == 3
    assert robot_pos.shape == (points_w0.shape[0], 3)
    assert robot_q.shape == (points_w0.shape[0], 4)

    env_nums, N, _ = points_w0.shape

    # 只用 x-y 平面绘图
    pts_xy = points_w0[..., :2]                                # [env_nums, N, 2]
    rpos_xy = robot_pos[..., :2]                               # [env_nums, 2]

    # 归一化四元数，避免非单位带来的数值偏差
    rq = robot_q / robot_q.norm(dim=-1, keepdim=True)          # [env_nums, 4]
    w, x, y, z = rq.unbind(dim=-1)                             # [env_nums]

    # 由四元数得到 3x3 旋转的前 2x2（绕任意轴的投影），直接写出批量公式
    Rxx = 1 - 2*(y*y + z*z)
    Rxy = 2*(x*y - w*z)
    Ryx = 2*(x*y + w*z)
    Ryy = 1 - 2*(x*x + z*z)
    R = torch.stack([torch.stack([Rxx, Rxy], dim=-1),
                     torch.stack([Ryx, Ryy], dim=-1)], dim=-2) # [env_nums, 2, 2]

    # 世界点 -> 以机器人为原点的坐标（先平移后旋转）
    pts_rel = pts_xy - rpos_xy.unsqueeze(1)                    # [env_nums, N, 2]
    pts_local = torch.matmul(pts_rel, R.transpose(-1, -2))     # [env_nums, N, 2]

    # —— 绘图（一次性） ——
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1) 一次 quiver 画出所有机器人的局部坐标轴（两组：X 轴、Y 轴）
    X0 = rpos_xy[:, 0]
    Y0 = rpos_xy[:, 1]
    Ux = R[:, 0, 0]   # 机器人坐标系 X 轴方向向量在世界坐标的 x 分量
    Uy = R[:, 1, 0]   # … y 分量
    Vx = R[:, 0, 1]   # 机器人坐标系 Y 轴方向向量在世界坐标的 x 分量
    Vy = R[:, 1, 1]

    ax.quiver(X0, Y0, Ux, Uy, angles='xy', scale_units='xy', scale=2,color="r", label='Robot X')
    colors = ['green', 'orange']  # Robot1 用绿，Robot2 用橙
    ax.quiver(X0, Y0, Vx, Vy, angles='xy', scale_units='xy', scale=2, color="g", label='Robot Y')

    # 2) 一次 scatter 画出所有环境的点（用环境 id 作为颜色编码）
    env_ids = torch.arange(env_nums).unsqueeze(1).expand(env_nums, N).reshape(-1)   # [env_nums*N]
    xy_flat = pts_xy.reshape(-1, 2)                                   # [env_nums*N, 2]
    colors = ['blue', 'orange']

    sc = ax.scatter(xy_flat[:, 0].numpy(), xy_flat[:, 1].numpy(),
                    c=[colors[i] for i in env_ids], cmap='tab10', s=24, alpha=0.9)

    # 3) 机器人位置（一次 scatter）
    # ax.scatter(X0.numpy(), Y0.numpy(), s=80, marker='o', edgecolor='k', zorder=5, label='Robot')
    ax.scatter(X0, Y0, s=200, marker='o', c=colors, edgecolor='k', zorder=5)
    ax.legend(['Robot 1', 'Robot 2'])

    # 坐标轴等设置
    ax.axhline(0, linewidth=0.5)
    ax.axvline(0, linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Batched Plot: Points & Robot Local Frames (2D)')

    # 图例：为每个环境做代理句柄（不需要 for 也可根据 env_ids 自动生成）
    if env_names is None:
        env_names = [f'Environment {i+1}' for i in range(env_nums)]
    # 代理点（只生成一次列表，不显式 for 循环也可，但写成推导式最清晰）
    proxies = [Line2D([0], [0], marker='o', linestyle='',
                      markerfacecolor=plt.get_cmap('tab10')(i),
                      markeredgecolor='k', label=env_names[i]) for i in range(env_nums)]
    legend_elems = proxies + [Line2D([0],[0], color='r', lw=2, label='Robot X'),
                              Line2D([0],[0], color='b', lw=2, label='Robot Y'),
                              Line2D([0],[0], marker='o', linestyle='', color='k', label='Robot')]
    ax.legend(handles=legend_elems, loc='best')

    plt.show()

# 批量绘制所有环境
plot_envs_batched(points_w0, robot_pos, robot_q,)
