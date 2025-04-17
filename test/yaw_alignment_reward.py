import torch
import math

def yaw_alignment_reward(current_position: torch.Tensor, 
                        target_yaw: torch.Tensor, 
                        current_yaw: torch.Tensor, 
                        k: float = 1.0) -> torch.Tensor:

    
   
    
    # 计算角度差异 Δθ = target_yaw - current_yaw，并规范化到 [-π, π]
    delta_yaw = target_yaw - current_yaw
    delta_yaw = (delta_yaw + math.pi) % (2 * math.pi) - math.pi  # 规范化
    print(delta_yaw)
    # 使用余弦函数计算对齐奖励（Δθ=0时奖励最大为k，Δθ=±π时奖励最小为-0.5k）
    # reward = k * torch.cos(delta_yaw)
    
    # 可选：将奖励映射到 [0, 2k] 范围（若需要非负奖励）
    reward = k * (torch.cos(delta_yaw) - 0.5)  
    
    return reward


# 示例用法
if __name__ == "__main__":
    # 机器人当前位置 (x=0, y=0)，目标点 (x=1, y=1)
    current_pos = torch.tensor([0.0, 0.0])
    
    
    # 测试不同 yaw 角的奖励值
    for yaw_deg in [0, 45, 90, -135, 180, -176]:
        yaw_rad = torch.deg2rad(torch.tensor(yaw_deg))
        target_yaw = torch.deg2rad(torch.tensor(35))
        reward = yaw_alignment_reward(current_pos, target_yaw, yaw_rad)
        print(f"Yaw={yaw_deg}°, Reward={reward.item():.2f}")

# 输出示例:
# Yaw=0°, Reward=0.71   (目标方向角为45°, Δθ=45° → cos(45°)≈0.707)
# Yaw=45°, Reward=1.00  (Δθ=0°, 完美对齐)
# Yaw=90°, Reward=0.00  (Δθ=-45°, cos(-45°)≈0.707 → 但示例中因目标方向角为45°, 此处应为cos(-45°)=0.707)
# 注：需要根据实际计算调整预期结果




