# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm
    

def bady_balance(env: ManagerBasedRLEnv, 
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                 threshold: float=-0.8) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity = asset.data.projected_gravity_b
    # print("gravity ",gravity)
    return gravity[:,2] > threshold

def object_reached_goal(
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
    return distances< threshold

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

def collision_with_obstacles(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """
    Checks for collision with obstacles.
    """
    # Accessing the contact sensor and its data
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Reshape as follows (num_envs, num_bodies, 3)
    force_matrix = contact_sensor.data.force_matrix_w.view(env.num_envs, -1, 3)

    # Calculating the force and returning true if it is above the threshold
    normalized_forces = torch.norm(force_matrix, dim=1)
    forces_active = torch.sum(normalized_forces, dim=-1) > threshold

    return torch.where(forces_active, True, False)