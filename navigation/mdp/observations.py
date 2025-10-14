from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import math

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    
def height_scan_nav(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.0) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    result = sensor.data.ray_hits_w[..., 2] - sensor.data.pos_w[:, 2].unsqueeze(1) - offset
    return  result





def lidar_distance_scan_horizontal(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """返回360维水平距离信息（忽略高度差）"""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    sensor_pos = sensor.data.pos_w  # [num_envs, 3]
    hit_points = sensor.data.ray_hits_w  # [num_envs, num_rays, 3]
    print("sensor pos ", sensor_pos)
    print("hit points ", hit_points)
    # 只计算水平距离：sqrt((x2-x1)^2 + (y2-y1)^2)
    horizontal_diff = hit_points[..., :2] - sensor_pos[..., :2].unsqueeze(1)
    distances = torch.norm(horizontal_diff, dim=2)
    
    # 限制最大距离为3m
    distances = torch.clamp(distances,min=0.0, max=3.0)
    # print(distances)
    
    return distances  # 形状: [num_envs, 360]