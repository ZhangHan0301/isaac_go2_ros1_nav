# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from isaaclab.managers import CommandTerm
from isaaclab.envs.mdp.commands.pose_2d_command import UniformPose2dCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformPose2dCommandCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)

def modify_command_pose(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str,  num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """    
    # 定义课程阶段配置
    curriculum_stages = [
        {"name": "stage1", "steps": 0, "pos_x": [1.0, 1.0], "pos_y": [6.0, 6.0]},
        {"name": "stage2", "steps": 2, "pos_x": [-2.0, -1.0], "pos_y": [7.0, 8.0]},
        {"name": "stage3", "steps": 30, "pos_x": [9.5, 10.5], "pos_y": [4.0, 5.0]},
        {"name": "stage4", "steps": 50, "pos_x": [7.0, 7.0], "pos_y": [-3.0, -3.0]},
        {"name": "stage5", "steps": 80, "pos_x": [-9.0, -8.0], "pos_y": [-9.0, -8.0]},
        # {"name": "stage6", "steps": 50000, "pos_x": [5.0, 5.0], "pos_y": [3.0, 3.0]},
        # {"name": "stage7_random", "steps": 80000, "pos_x": [1.0, 5.0], "pos_y": [1.0, 3.0]}
    ]

    if env.common_step_counter > num_steps:
        # 找到当前阶段
        current_stage = None
        for stage in curriculum_stages:
            if env.common_step_counter >= stage["steps"]:
                current_stage = stage
        
        # 更新目标点设置
        term: CommandTerm = env.command_manager.get_term(term_name)
        term_cfg: UniformPose2dCommandCfg = term.cfg
        term_cfg.ranges.pos_x = current_stage["pos_x"]
        term_cfg.ranges.pos_y = current_stage["pos_y"]