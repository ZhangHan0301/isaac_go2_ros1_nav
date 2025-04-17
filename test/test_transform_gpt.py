import torch

# 四元数转旋转矩阵
def quaternion_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    quat_normalized = quat / (torch.norm(quat, dim=1, keepdim=True) + 1e-8)  # 归一化四元数
    
    w, x, y, z = quat_normalized[:, 0], quat_normalized[:, 1], quat_normalized[:, 2], quat_normalized[:, 3]
    
    # 计算旋转矩阵
    x2, y2, z2 = x**2, y**2, z**2
    xy, xz, yz = x*y, x*z, y*z
    xw, yw, zw = x*w, y*w, z*w
    
    r00 = 1 - 2*y2 - 2*z2
    r01 = 2*xy - 2*zw
    r02 = 2*xz + 2*yw
    
    r10 = 2*xy + 2*zw
    r11 = 1 - 2*x2 - 2*z2
    r12 = 2*yz - 2*xw
    
    r20 = 2*xz - 2*yw
    r21 = 2*yz + 2*xw
    r22 = 1 - 2*x2 - 2*y2
    
    R = torch.stack([
        torch.stack([r00, r01, r02], dim=1),
        torch.stack([r10, r11, r12], dim=1),
        torch.stack([r20, r21, r22], dim=1)
    ], dim=1)
    return R

# 从世界坐标系转到机器人坐标系
def convert_ray_hits_to_robot_frame(ray_hits_w, pos_w, quat_w):
    # 平移变换（将世界坐标系中的点移到机器人坐标系原点）
    translated_points = ray_hits_w - pos_w.unsqueeze(1)  # 形状 (N, B, 3)
    
    # 生成旋转矩阵的逆（转置）进行旋转变换
    R = quaternion_to_rotation_matrix(quat_w)  # 旋转矩阵 R，形状 (N, 3, 3)
    R_inv = R.transpose(1, 2)  # 旋转矩阵的逆，形状 (N, 3, 3)
    
    # 旋转变换，利用逆矩阵
    ray_hits_local = torch.bmm(translated_points, R_inv)  # 形状 (N, B, 3)
    return ray_hits_local

# 测试demo
def test_convert_ray_hits_to_robot_frame():
    # 假设有2个机器人（N=2）每个机器人3个点（B=3）
    N = 2
    B = 3

    # 假设机器人的位置 (pos_w) 在世界坐标系中的位置
    pos_w = torch.tensor([[1.0, 2.0, 3.0],  # 机器人1的位置
                          [4.0, 5.0, 6.0]])  # 机器人2的位置

    # 假设机器人的姿态 (quat_w) 以四元数表示，(w, x, y, z)
    quat_w = torch.tensor([[0.707, 0.0, 0.707, 0.0],  # 四元数1，绕Y轴旋转90度
                           [1.0, 0.0, 0.0, 0.0]])  # 四元数2，表示无旋转

    # 假设雷达射线命中的点(ray_hits_w)，形状为 (N, B, 3)
    ray_hits_w = torch.tensor([[[5.0, 6.0, 7.0],  # 第一组数据
                               [8.0, 9.0, 10.0],
                               [11.0, 12.0, 13.0]],
                              [[2.0, 3.0, 4.0],  # 第二组数据
                               [3.0, 4.0, 5.0],
                               [6.0, 7.0, 8.0]]], dtype=torch.float32)

    # 调用转换函数
    ray_hits_local = convert_ray_hits_to_robot_frame(ray_hits_w, pos_w, quat_w)

    # 输出转换后的数据
    print("Ray hits in robot frame:")
    print(ray_hits_local)

# 运行测试
test_convert_ray_hits_to_robot_frame()
