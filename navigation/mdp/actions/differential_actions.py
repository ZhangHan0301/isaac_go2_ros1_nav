from __future__ import annotations

from typing import TYPE_CHECKING

import carb
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv  # noqa: F811

    from . import actions_cfg


class DifferentialDriveAction(ActionTerm):

    cfg: actions_cfg.DifferentialDriveActionCfg

    _asset: Articulation

    _wheel_distance: float

    _drive_joint_names: list[str]

    _scale: torch.Tensor

    _offset: torch.Tensor

    def __init__(self, cfg: actions_cfg.DifferentialDriveActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # self._drive_joint_ids, self._drive_joint_names = self._asset.find_joints(self.cfg.drive_joint_names)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        # self._joint_vel = torch.zeros(self.num_envs, len(self._drive_joint_ids), device=self.device)


    @property
    def action_dim(self) -> int:
        return 2  # Assuming a 2D action vector (linear velocity, angular velocity)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions = self.raw_actions*5.0

    def apply_actions(self):

        left_wheel_vel, right_wheel_vel = differential_drive(
            self._processed_actions[:, 0], self._processed_actions[:, 1], 0.12, self.device)
        
        # print("action ", self._processed_actions[:, 0], self._processed_actions[:, 1])
        # print("wheel vel ", left_wheel_vel, right_wheel_vel)
        self._asset.set_joint_velocity_target( torch.stack([left_wheel_vel,right_wheel_vel],dim=1))


def differential_drive(v, omega, wheel_separation, device):
    """
    将差速机器人的线速度和角速度转换为左右轮的线速度
    :param v: 机器人中心的线速度 (m/s)
    :param omega: 机器人绕中心的角速度 (rad/s)
    :param wheel_separation: 左右轮间距 (m)
    :return: (左轮线速度, 右轮线速度) 元组 (单位均为 m/s)
    """
    left_wheel_vel = v - (omega * wheel_separation) / 2
    right_wheel_vel = v + (omega * wheel_separation) / 2
    return left_wheel_vel, right_wheel_vel

