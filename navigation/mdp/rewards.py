# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from turtle import distance

from warp import zeros_like

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


def goal_distance_change_reward(env: ManagerBasedRLEnv, command_name: str,) -> torch.Tensor:
    """
    Gives a penalty if the rover is too far from the target.
    """

    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]
    pre_target = env.command_manager.last_command

    distance = torch.norm(target_position, p=2, dim=-1).to(device=env.device)
    last_distance = torch.norm(pre_target[:,:2], p=2, dim=-1).to(device=env.device)
    return torch.where(last_distance<distance, -1.0, 0.5)



def reached_target(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """
    Determine whether the target has been reached with an optimized reward function.

    This function checks if the rover is within a certain threshold distance from the target.
    The reward is scaled by both the distance to target and the angle of approach, and also by remaining time steps.
    """

    # Accessing the target's position w.r.t. the robot frame
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    # Get angle to target
    angle = env.command_manager.get_command(command_name)[:, 3]

    # Calculating the distance and determining if the target is reached
    distance = torch.norm(target_position, p=2, dim=-1)
    time_steps_to_goal = env.max_episode_length - env.episode_length_buf

    
    reward_scale = time_steps_to_goal / env.max_episode_length

    # Gradual reward based on distance to target (not just a binary threshold)
    distance_reward = torch.exp(-distance / threshold)  # Exponentially decaying reward with distance

    # Gradual reward based on angle (penalizing larger angles)
    angle_reward = torch.exp(-torch.abs(angle))  # Exponentially decaying reward with angle

    # Combined reward (distance, angle, time)
    combined_reward = distance_reward * angle_reward * reward_scale

    return torch.where(distance < threshold, combined_reward, 0.0)


def progressive_reward(env: ManagerBasedRLEnv, command_name: str, max_distance: float) -> torch.Tensor:
    """
    Gradually reward the rover as it gets closer to the target.

    This function gives a smooth, progressive reward based on the distance to the target,
    encouraging the rover to move toward the goal.
    """

    # Accessing the target's position w.r.t. the robot frame
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    # Calculating the distance to the target
    distance = torch.norm(target_position, p=2, dim=-1)

    # Normalize distance to [0, 1] range based on max_distance
    normalized_distance = torch.clamp(distance / max_distance, min=0.0, max=1.0)

    # Gradually increasing reward as the rover gets closer to the target
    # The closer it gets, the higher the reward.
    reward = torch.exp(-normalized_distance * 5)  # Exponential decay to gradually reward closer distances

    return reward


def oscillation_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    惩罚动作震荡（只考虑纵向速度 vx 和角速度 wz，忽略横向 vy）。
    """
    action = env.action_manager.action         # shape: [N, 3]
    prev_action = env.action_manager.prev_action

    vx_diff = action[:, 0] - prev_action[:, 0]  # 纵向速度差
    wz_diff = action[:, 2] - prev_action[:, 2]  # 角速度差

    penalty_x = torch.where(abs(vx_diff)>0.5, -0.5, 0.2)
    penalty_theta = torch.where(abs(wz_diff)>0.5, -0.5, 0.2)
    
    return penalty_x + penalty_theta 



def far_from_target_reward(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """
    Gives a penalty if the rover is too far from the target.
    """

    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    distance = torch.norm(target_position, p=2, dim=-1)

    return torch.where(distance > threshold, 1.0, 0.0)

def angle_to_target_penalty(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    Calculate the penalty for the angle between the rover and the target.

    This function computes the angle between the rover's heading direction and the direction
    towards the target. A penalty is applied if this angle exceeds a certain threshold.
    """

    # Get vector(x,y) from rover to target, in base frame of the rover.
    target_vector_b = env.command_manager.get_command(command_name)[:, :2]

    # Calculate the angle between the rover's heading [1, 0] and the vector to the target.
    angle = torch.atan2(target_vector_b[:, 1], target_vector_b[:, 0])

    # Return the absolute value of the angle, normalized by the maximum episode length.
    return torch.where(torch.abs(angle) > 2.0, torch.abs(angle) / env.max_episode_length, 0.0)

def wheel_robot_excessive_speed_penalty(env:ManagerBasedRLEnv, asset_cfg:SceneEntityCfg, speed_limit:float=1.0)->torch.Tensor:
    vel_x_penalty = torch.where(abs(env.action_manager.action[:, 0])> speed_limit, 2, 0.0)
    vel_theta_penalty = torch.where(abs(env.action_manager.action[:, 1])> speed_limit, 2, 0.0)
    return vel_x_penalty+vel_theta_penalty

def excessive_speed_penalty(env:ManagerBasedRLEnv, asset_cfg:SceneEntityCfg, speed_limit:float=1.0)->torch.Tensor:
    vel_x_penalty = torch.where(abs(env.action_manager.action[:, 0])> speed_limit, 3, 0.0)
    vel_y_penalty = torch.where(abs(env.action_manager.action[:, 1])> speed_limit, 3, 0.0)
    vel_theta_penalty = torch.where(abs(env.action_manager.action[:, 2])> speed_limit+1.0, 3, 0.0)
    vel_x_less_penalty = torch.where(abs(env.action_manager.action[:,0])<0.1, 3, 0.0)
    return vel_x_penalty+vel_y_penalty+vel_theta_penalty + vel_x_less_penalty

def heading_soft_contraint(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Calculate a penalty for driving backwards.
    后退惩罚

    This function applies a penalty when the rover's action indicates reverse movement.
    The penalty is normalized by the maximum episode length.
    """
    return torch.where(env.action_manager.action[:, 0] < 0.0, (1.0 / env.max_episode_length), 0.0)

def angle_to_goal_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    Calculate the angle to the goal.

    This function computes the angle between the rover's heading direction and the direction
    towards the goal. A reward is given based on the cosine of this angle.
    """
    # Get vector(x,y) from rover to target, in base frame of the rover.
    target_vector_b = env.command_manager.get_command(command_name)[:, :2]
    distance = torch.norm(target_vector_b, p=2, dim=-1)
    angle_b = env.command_manager.get_command(command_name)[:, 3]

    angle_reward = (1 / (1 + distance)) * 1 / (1 + torch.abs(angle_b))

    # Return the cosine of the angle, normalized by the maximum episode length.
    return angle_reward / env.max_episode_length

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

    des_pos_b = command[:, :2]
    # diff = current_robot_pose - des_pos_b
    distances = torch.norm(des_pos_b, dim=1)
    # 创建奖励张量（批量处理）
    base_reward = 5.0  # 成功到达的奖励值
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
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    #print("position reward ", 1-torch.tanh(distance/std))
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str, threshold:float = 0.8) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    distance = torch.norm(command[:,:2])
    if(distance>0.5):
        heading_reward = 0.0
    else:
        heading_reward = heading_b.abs()
    return heading_reward

def base_line_vel_y_penalize (env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    speed_limit: float = 1.0,  # 新增速度阈值参数
    penalty_coef: float = 1.0    # 新增惩罚系数
)-> torch.Tensor:
        # 获取机器人实体
        asset: RigidObject = env.scene[asset_cfg.name]
        
    # 获取所有环境实例的y轴速度绝对值
        line_y_vel_abs = torch.abs(asset.data.root_lin_vel_b[:, 1]) 
        # print("vel y penalty ", penalty_coef*line_y_vel_abs)
        return penalty_coef* line_y_vel_abs


def velocity_command_error(env: ManagerBasedRLEnv, command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # 获取目标命令
    command = env.command_manager.get_command(command_name)
    pose_goal = command[:, :2]

    # 计算目标方向的归一化向量
    norm_direction = torch.norm(pose_goal, dim=1, keepdim=True)


    # 防止除零错误，避免norm_direction为零
    norm_direction = torch.clamp(norm_direction, min=1e-6)  # 防止为零

    unit_direction = pose_goal / norm_direction

    # 获取当前物体的速度
    asset: RigidObject = env.scene[asset_cfg.name]
    velocity = asset.data.root_ang_vel_b[:, :2]

    # 计算速度奖励
    velocity_reward = torch.sum(unit_direction * velocity, dim=1)

    # 使用clamp确保奖励值在合理范围内，防止nan值
    velocity_reward = torch.clamp(velocity_reward, min=1e-6, max=1.0)


    return velocity_reward


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

    #print("line vel reward ", reward)
    return reward 


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

    # 将高度大于阈值的赋值obstacle_distance，其余置零
    # result = ray_data_w.clone()
    ray_data_w[:, :, 2] = torch.where(mask, obstacle_distance, torch.zeros_like(third_elements)+3.0)
    obs_distance = ray_data_w[:, :, -1]
    reward =  torch.where(
        obs_distance[:,:]<d_safe,
        obstacle_distance*12.5-12,
        torch.zeros_like(obs_distance)
    )
    reward, indices = torch.min(reward, dim=-1)
    # print(reward.size())
    # print("obstacle reward ", reward)
    #print("-----------------------------\n")
    return reward

