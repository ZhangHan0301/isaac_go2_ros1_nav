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
from isaaclab.sensors import ContactSensor

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