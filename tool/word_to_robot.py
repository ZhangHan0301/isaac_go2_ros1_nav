import torch

# 已知数据
R = torch.tensor([[[0.265, -0.964],
                  [0.964, 0.265]]], device='cuda:0')

cloud_point_w = torch.tensor([[-2.206, 1.447, 0.660]], device='cuda:0')
robot_pos_w = torch.tensor([[-1.640, -0.264, 0.545]], device='cuda:0')

# 计算相对位置（世界坐标系）
rel_w = cloud_point_w - robot_pos_w  # tensor([[-0.566, 1.711, 0.115]])

# 提取xy部分应用旋转
rel_w_xy = rel_w[:, :2]
rel_w_z = rel_w[:, 2]

# 应用旋转矩阵（世界坐标系 -> 机器人坐标系）
rel_robot_xy = torch.matmul(rel_w_xy, R)



print("雷达点在机器人坐标系中的相对坐标:", rel_robot_xy)