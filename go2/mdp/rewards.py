
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import wrap_to_pi
import math
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    
def robot_pose_target(env:ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg)->torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    robot_pose = asset.data.root_pos_w
    # print("robot pose: ", robot_pose)
    return (robot_pose[:,0] - target).clone().detach()   # 使用 torch.tensor()

def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    print("des pose ", command[:,:3])
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)

def yaw_alignment_reward(env: ManagerBasedRLEnv, 
                         command_name: str,
                         robot_config: SceneEntityCfg = SceneEntityCfg("robot"),
                          threshold: float = 0.5):
    asset: Articulation = env.scene[robot_config.name]
    
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    # 计算目标方向角（使用 atan2 处理象限）
    target_yaw = torch.atan2(command[:,1],command[:,0])  # 形状与 dx/dy 相同
    
    # quat_yaw = asset.data.root_quat_w
    # qw = quat_yaw[:, 0]
    # qx = quat_yaw[:, 1]
    # qy = quat_yaw[:, 2]
    # qz = quat_yaw[:, 3]
    # yaw_base = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    
    # # 计算角度差异 Δθ = target_yaw - current_yaw，并规范化到 [-π, π]
    # print("-----------\n target_yaw ", target_yaw * 180/math.pi)
    # print("yaw base ", yaw_base*180/math.pi)
    # delta_yaw = target_yaw - yaw_base
    # delta_yaw = (delta_yaw + math.pi) % (2 * math.pi) - math.pi  # 规范化
    # print("delta_yaw", delta_yaw)
    # 可选：将奖励映射到 [0, 2k] 范围（若需要非负奖励）
    reward = (torch.cos(target_yaw) - 0.5)  
    reward = torch.where(
        distance > threshold,
        reward,
        torch.zeros_like(reward)
    )
    # print("yaw reward ",reward)
    return reward