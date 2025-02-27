
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    
def robot_pose_target(env:ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg)->torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    robot_pose = asset.data.root_pos_w
    print("robot pose: ", robot_pose)
    return (robot_pose[:,0] - target).clone().detach()   # 使用 torch.tensor()
    