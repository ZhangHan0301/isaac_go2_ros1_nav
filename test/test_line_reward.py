import torch
import pytest

def calculate_velocity_penalty(
    x_velocity: torch.Tensor,
    speed_limit: float = 1.0,
    penalty_factor: float = 1.0
) -> torch.Tensor:

    # 超速惩罚计算
    overspeed = torch.clamp(x_velocity - speed_limit, min=0)
    overspeed_penalty = overspeed * penalty_factor

    # 反向运动惩罚
    reverse_penalty = torch.abs(torch.clamp(x_velocity, max=0)) * penalty_factor

    # 合成总惩罚值
    total_penalty = overspeed_penalty + reverse_penalty
    
    # 调整形状适配奖励系统
    print(total_penalty)
    return total_penalty.unsqueeze(-1)


def calculate_velocity_penalty2(
    line_x_vel: torch.Tensor,
    vel_threshold: float = 1.0,
    penalty_coef: float = 1.0
) -> torch.Tensor:
    # 构建三种情况的掩码
    negative_mask = line_x_vel < 0
    over_threshold_mask = line_x_vel > vel_threshold
    
    # 计算各区域的惩罚值
    negative_penalty = penalty_coef * line_x_vel.abs()  # 负速度线性惩罚
    over_threshold_penalty = penalty_coef * (line_x_vel - vel_threshold)**2  # 超阈值二次惩罚
    
    # 向量化条件选择
    reward = torch.where(
        negative_mask,
        negative_penalty,  # 情况1：负速度
        torch.where(
            over_threshold_mask,
            over_threshold_penalty,  # 情况2：超过正向阈值（注意负号表示惩罚）
            torch.zeros_like(line_x_vel)+1  # 情况3：安全范围不惩罚
        )
    )
    print(reward)
    return reward



def test_velocity_penalty():
    """测试速度惩罚函数的各种边界情况"""
    
    # 测试用例：4个环境的不同速度状态
    test_velocities = torch.tensor([2.0,   # 超速
                                    1.0,   # 阈值
                                    0.5,   # 正常
                                    -0.5]) # 反向
    speed_limit = 1.0
    penalty_factor = 0.5
    
    # 执行计算
    penalties = calculate_velocity_penalty(
        x_velocity=test_velocities,
        speed_limit=speed_limit,
        penalty_factor=penalty_factor
    )
    
    # 预期结果（手工计算验证）
    expected = torch.tensor([
        # 超速：(2.0-1.0)*0.5 = 0.5 → -0.5
        -0.5,  
        # 阈值：无惩罚
        0.0,   
        # 正常：无惩罚
        0.0,    
        # 反向：abs(-0.5)*0.5 = 0.25
        0.25   
    ], dtype=torch.float32)
    
    # 验证数值正确性（允许1e-6误差）
    assert torch.allclose(penalties.squeeze(), expected, atol=1e-6), \
        "惩罚值计算结果与预期不符"
    
    # 验证输出形状
    assert penalties.shape == (4, 1), \
        f"输出形状错误，预期(4,1)，实际得到{penalties.shape}"

    # 验证零速度情况
    zero_penalty = calculate_velocity_penalty(torch.zeros(3), 1.0, 0.5)
    assert torch.all(zero_penalty == 0), "零速度时应无惩罚"

if __name__ == "__main__":
    # pytest.main(["-v", __file__])
    test_velocities = torch.tensor([2.0,   # 超速
                                    1.0,   # 阈值
                                    0.5,   # 正常
                                    -0.5]) # 反向
    speed_limit = 1.0
    penalty_factor = -1.0
    
    # 执行计算
    print("raw speed ", test_velocities)
    penalties = calculate_velocity_penalty(
        x_velocity=test_velocities,
        speed_limit =speed_limit,
        penalty_factor =  penalty_factor
    )
    
    print("-----------------\n raw speed ", test_velocities)
    penalties = calculate_velocity_penalty2(
        line_x_vel=test_velocities,
        vel_threshold=speed_limit,
        penalty_coef= penalty_factor
    )