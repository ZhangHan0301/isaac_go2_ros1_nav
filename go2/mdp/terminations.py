# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.managers.command_manager import CommandTerm

from omni.isaac.lab.sensors import ContactSensor


    

def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    threshold: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    current_robot_pose = robot.data.root_pos_w
    command = env.command_manager.get_command(command_name)
    # print("\ncommand: ", command)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    # diff = current_robot_pose - des_pos_b
    distance = torch.linalg.norm(des_pos_b, dim=1)
    # print("distance :", distance)

    # rewarded if the object is lifted above the threshold
    return distance < threshold

def action_limitations(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_tool = ee_frame.data.target_pos_w[..., 0, :]
    x = ee_tool[:,0] - robot.data.root_state_w[:, 0]
    y = ee_tool[:,1] - robot.data.root_state_w[:, 1]

    return (ee_tool[:,2] < 0.2) | (ee_tool[:,2] > 2.0) | (y > 0.5) | (y < -0.2) | (x < 0.1)

def illegal_contact2(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w #四维数据 [环境数、历史轨迹、身体部位、xyz方向上的力]
    # print("net contact forces: ", net_contact_forces.shape)
    # print("net contact sum : ", net_contact_forces)

    print("\n ---------- \n  sensor name: ", sensor_cfg.body_names)
    print("net contact forces: ",net_contact_forces[:,sensor_cfg.body_ids])
    
    max_forces = torch.max(torch.norm(net_contact_forces[:, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    print("max forces: ", max_forces)
    
    
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=0
    )
    
def detec_collision(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces_history = contact_sensor.data.net_forces_w_history[:,:,sensor_cfg.body_ids] #四维数据 [环境数、历史轨迹、身体部位、xyz方向上的力]
    force_magnitude = torch.norm(net_contact_forces_history,dim=-1)
    print("force magnitude ", force_magnitude)
    
    collision_detected = (force_magnitude > threshold).any(dim=2)
    print("collision detected ", collision_detected)
    done = collision_detected.all(dim=1)
    # check if any contact force exceeds the threshold
    # done* (env.episode_length_buf>10)
    return done* (env.episode_length_buf>10)