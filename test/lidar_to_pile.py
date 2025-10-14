import torch
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 四元数共轭计算
def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

# 四元数旋转
def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    qv = torch.stack([x, y, z], dim=-1)           # (..., 3)
    uv = torch.cross(qv, v, dim=-1)               # (..., 3)
    uuv = torch.cross(qv, uv, dim=-1)             # (..., 3)
    return v + 2.0 * (w.unsqueeze(-1) * uv + uuv)  # (..., 3)

# 创建测试数据
def test_lidar_rotation():
    # 假设数据：单个环境，10个点
    num_points = 10
    num_environments = 1
    num_sectors = 13  # 扇形区域数量

    # 随机生成点云数据 (E, R, 3)
    ray_hits_world = torch.rand((num_environments, num_points, 3), dtype=torch.float32).cuda()

    # 随机生成传感器位置和四元数
    sensor_positions_world = torch.rand((num_environments, 3), dtype=torch.float32).cuda()
    sensor_quaternions_world = torch.rand((num_environments, 4), dtype=torch.float32).cuda()

    # 归一化四元数（确保它是单位四元数）
    sensor_quaternions_world = sensor_quaternions_world / sensor_quaternions_world.norm(dim=-1, keepdim=True)

    # 传感器数据字典
    sensor_data = {
        'ray_hits_w': ray_hits_world,
        'pos_w': sensor_positions_world,
        'quat_w': sensor_quaternions_world
    }

    # 计算开始时间
    start_time = time.time()
    
    # 先进行地面过滤再旋转
    filtered_points = ray_hits_world[ray_hits_world[..., 2] > 0.1]  # (K, 3)
    points_after_rotation = lidar_processing(sensor_data, num_sectors=num_sectors, ground_threshold=0.1).cpu().numpy()
    
    end_time = time.time()
    print(f"Time for ground filtering and then rotation: {end_time - start_time:.6f} seconds")
    
    # 计算开始时间
    start_time = time.time()

    # 先进行旋转再地面过滤
    points_after_rotation2 = lidar_processing(sensor_data, num_sectors=num_sectors, ground_threshold=0.1).cpu().numpy()
    filtered_points2 = ray_hits_world[ray_hits_world[..., 2] > 0.1]  # (K, 3)

    end_time = time.time()
    print(f"Time for rotation and then ground filtering: {end_time - start_time:.6f} seconds")
    
    # 可视化
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')

    # 将 filtered_points 移动到 CPU，然后转换为 numpy 数组
    filtered_points_cpu = filtered_points.cpu().numpy()

    # 绘制机器人的位置 (原点) 和坐标系
    ax.scatter(0, 0, 0, c='black', marker='o', label="Robot Position")  # 机器人位置 (原点)
    ax.quiver(0, 0, 0, 1, 0, 0, length=0.2, color='r', label="X axis")  # X轴
    ax.quiver(0, 0, 0, 0, 1, 0, length=0.2, color='g', label="Y axis")  # Y轴
    ax.quiver(0, 0, 0, 0, 0, 1, length=0.2, color='b', label="Z axis")  # Z轴

    # 绘制点云
    ax.scatter(filtered_points_cpu[:, 0], filtered_points_cpu[:, 1], filtered_points_cpu[:, 2], c='blue', label="Filtered & Rotated Points")
    ax.set_title('Filtered and Rotated Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 在这里添加极坐标标记线
    for point in filtered_points_cpu:
        ax.plot([0, point[0]], [0, point[1]], [0, point[2]], color='gray', linestyle='--')

    ax = fig.add_subplot(122, projection='3d')

    # 将 filtered_points2 移动到 CPU，然后转换为 numpy 数组
    filtered_points2_cpu = filtered_points2.cpu().numpy()

    # 绘制机器人的位置 (原点) 和坐标系
    ax.scatter(0, 0, 0, c='black', marker='o', label="Robot Position")  # 机器人位置 (原点)
    ax.quiver(0, 0, 0, 1, 0, 0, length=0.2, color='r', label="X axis")  # X轴
    ax.quiver(0, 0, 0, 0, 1, 0, length=0.2, color='g', label="Y axis")  # Y轴
    ax.quiver(0, 0, 0, 0, 0, 1, length=0.2, color='b', label="Z axis")  # Z轴

    # 绘制点云
    ax.scatter(filtered_points2_cpu[:, 0], filtered_points2_cpu[:, 1], filtered_points2_cpu[:, 2], c='red', label="Rotated & Filtered Points")
    ax.set_title('Rotated and Filtered Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 在这里添加极坐标标记线
    for point in filtered_points2_cpu:
        ax.plot([0, point[0]], [0, point[1]], [0, point[2]], color='gray', linestyle='--')

    plt.tight_layout()
    plt.show()

# 定义 lidar_processing 函数
def lidar_processing(sensor_data, num_sectors=13, ground_threshold=0.1, default_value=float("inf")):
    ray_hits_world = sensor_data['ray_hits_w']  # (E, R, 3)
    sensor_positions_world = sensor_data['pos_w']  # (E, 3)
    sensor_quaternions_world = sensor_data['quat_w']  # (E, 4)
    
    device, dtype = ray_hits_world.device, ray_hits_world.dtype
    num_environments, num_rays, _ = ray_hits_world.shape

    # --- 世界坐标系到局部坐标系的转换 ---
    # 平移
    relative_positions = ray_hits_world - sensor_positions_world[:, None, :]  # (E, R, 3)

    # 逆旋转（单位四元数逆 = 共轭）
    quaternion_conjugates = _quat_conj(sensor_quaternions_world)  # (E, 4)
    quaternion_conjugates_expanded = quaternion_conjugates[:, None, :].expand(num_environments, num_rays, 4)  # (E, R, 4)
    ray_hits_local = _quat_rotate(quaternion_conjugates_expanded, relative_positions)  # (E, R, 3)

    # --- 地面过滤（在局部坐标系） ---
    valid_points = ray_hits_local[..., 2] > ground_threshold  # (E, R)

    # --- 极坐标转换和分箱 ---
    xy_coordinates = ray_hits_local[..., :2]  # (E, R, 2)
    distances = torch.linalg.norm(xy_coordinates, dim=-1)  # (E, R) 计算每个点的距离
    angles = torch.atan2(xy_coordinates[..., 1], xy_coordinates[..., 0])  # (E, R) 计算每个点的角度
    angles = (angles + 2 * math.pi) % (2 * math.pi)  # [0, 2π) 确保角度在 [0, 2π) 范围内
    
    # 每个扇形区域的角度跨度
    sector_angle = 2 * math.pi / num_sectors
    sectors = (angles / sector_angle).floor().clamp_(max=num_sectors-1).long()  # (E, R) 计算每个点属于哪个区域

    # 只保留有效的点
    valid_indices = valid_points
    if not valid_indices.any():
        return torch.full((num_environments, num_sectors), default_value, dtype=dtype, device=device)

    valid_distances = distances[valid_indices]  # (K,) 过滤后的有效点的距离
    valid_sectors = sectors[valid_indices]  # (K,) 过滤后的有效点的区域编号

    # 将 (env, seg) 压平成 1D 索引：idx = env * N + seg
    environment_indices = torch.arange(num_environments, device=device).unsqueeze(1).expand(num_environments, num_rays)  # (E, R)
    environment_valid_indices = environment_indices[valid_indices]  # (K,) 获取有效点对应的环境索引
    flat_indices = environment_valid_indices * num_sectors + valid_sectors  # (K,) 合并环境和区域索引为 1D 索引

    # --- 聚合为每环境×扇区的最小 r ---
    min_distances_per_sector = torch.full((num_environments * num_sectors,), float("inf"), dtype=dtype, device=device)

    # 使用 scatter_reduce 更高效
    if hasattr(min_distances_per_sector, "scatter_reduce"):
        min_distances_per_sector = min_distances_per_sector.scatter_reduce(0, flat_indices, valid_distances, reduce="amin", include_self=True)

    min_distances_per_sector = min_distances_per_sector.view(num_environments, num_sectors)
    min_distances_per_sector = torch.where(torch.isinf(min_distances_per_sector),
                                            torch.tensor(default_value, dtype=dtype, device=device),
                                            min_distances_per_sector)
    return min_distances_per_sector

# 运行测试
test_lidar_rotation()
