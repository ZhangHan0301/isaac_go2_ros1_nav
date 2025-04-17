# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from turtle import distance

from cv2 import norm
from regex import R

import torch
import math
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, Imu, RayCaster
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    threshold: float = 0.2,
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.

    """
    # extract the used quantities (to enable type-hinting)
    command = env.command_manager.get_command(command_name)


    des_pos_b = command[:, :3]
    # diff = current_robot_pose - des_pos_b
    distances = torch.norm(des_pos_b, dim=1)
    # 创建奖励张量（批量处理）
    base_reward = 100.0  # 成功到达的奖励值
    rewards = torch.where(
        distances < threshold,
        torch.full_like(distances, base_reward),
        torch.zeros_like(distances)
    )
    # print("reach goal reward", rewards)
    return rewards


def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)    
    des_pos_b = command[:, :3]
    
    distance = torch.norm(des_pos_b, dim=1)
    #print("position reward ", 1-torch.tanh(distance/std))
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str, threshold:float = 0.8) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    
    # print("des pose ", command)
    # print("heading_b.abs(): ",heading_b.abs())
    rewards = torch.where(
        distance < threshold,
        torch.abs(heading_b),
        torch.zeros_like(heading_b)
    )
    #print("heading reward ", rewards)
    return rewards


def base_lin_vel_penalize(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    speed_limit: float = 1.0,  # 新增速度阈值参数
    penalty_coef: float = 1.0    # 新增惩罚系数
) -> torch.Tensor:
    """Root linear velocity penalty with threshold handling.
    
    功能说明：
    - 当X轴速度 < 0 时：按原值施加线性惩罚
    - 当0 <= X轴速度 <= speed_limit 时：无惩罚（奖励=0)
    - 当X轴速度 > speed_limit 时：施加二次惩罚
    
    Args:
        speed_limit (float): 允许的最大正向速度（超过会惩罚）
        penalty_coef (float): 惩罚强度系数
    """
    # 获取机器人实体
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 提取X轴线速度 [num_envs, ]
    line_x_vel = asset.data.root_lin_vel_b[:, 0]
  
    # 构建三种情况的掩码
    negative_mask = line_x_vel < 0
    over_threshold_mask = line_x_vel > speed_limit
    
    # 计算各区域的惩罚值
    negative_penalty = penalty_coef * line_x_vel  # 负速度线性惩罚
    over_threshold_penalty = penalty_coef * (line_x_vel - speed_limit) # 超阈值二次惩罚
    
    # 向量化条件选择
    reward = torch.where(
        negative_mask,
        negative_penalty,  # 情况1：负速度
        torch.where(
            over_threshold_mask,
            -over_threshold_penalty,  # 情况2：超过正向阈值（注意负号表示惩罚）
            torch.zeros_like(line_x_vel) + 0.2  # 情况3：安全范围不惩罚
        )
    )
    # 获取所有环境实例的y轴速度绝对值
    line_y_vel_abs = torch.abs(asset.data.root_lin_vel_b[:, 1]) 
    
    vel_y_penalty = line_y_vel_abs*-2.5
    
    #print("line vel reward ", reward)
    return reward + vel_y_penalty


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
    

    # 可选：将奖励映射到 [0, 2k] 范围（若需要非负奖励）
    reward = (torch.cos(target_yaw) - 0.9)  
    reward = torch.where(
        distance > threshold,
        reward,
        torch.zeros_like(reward)
    )
    #print("yaw reward ",reward)
    return reward

def obstacle_reward(env: ManagerBasedRLEnv, 
                    sensor_cfg: SceneEntityCfg= SceneEntityCfg("height_scanner2"), 
                    z_threshold: float = 0.3,
                    d_safe : float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset 
    ray_data_w = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    
        # 提取第三个元素并生成掩码
    third_elements = ray_data_w[:, :, 2]
    mask = third_elements > z_threshold

    # 计算前两个元素的二范数
 
    obstacle_point_xy = ray_data_w[...,:2]
    
    obstacle_distance = torch.norm(obstacle_point_xy, p=2, dim=2)  # 形状 (2, 3)

    # 将范数赋值给符合条件的第三个元素，其余置零
    # result = ray_data_w.clone()
    ray_data_w[:, :, 2] = torch.where(mask, obstacle_distance, torch.zeros_like(third_elements))
    last_elements = ray_data_w[:, :, -1]
    reward =  torch.where(
        last_elements[:,:]<d_safe,
        last_elements.cos()+1.0,
        torch.zeros_like(last_elements)
    )
    reward *= last_elements 
    reward = -reward.sum(dim=1)
    #print("obstacle reward ", reward)
    #print("-----------------------------\n")
    return reward

