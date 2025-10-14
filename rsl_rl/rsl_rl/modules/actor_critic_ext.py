# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


# <<< NEW: 极简的预处理器（内置到本文件，避免额外导入） -----------------
class _HeightScanPreproc1D(nn.Module):
    """将 (pose_command[4], height_scan[651]) -> (pose[4] + cnn_feat[elev_out])"""
    def __init__(self, pose_dim: int = 4, height_len: int = 651, elev_out: int = 128):
        super().__init__()
        self.pose_dim = pose_dim
        self.height_len = height_len
        # 轻量 1D-CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # -> (B,64,1)
            nn.Flatten(),             # -> (B,64)
        )
        self.fc = nn.Sequential(nn.Linear(64, elev_out), nn.ReLU(inplace=True))
        self.output_dim = pose_dim + elev_out
        print(f"fc {self.fc}")

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        # 约定：obs = [pose(4), height(651), ...]（这里只取前 655 维）
        pose = obs_flat[:, :self.pose_dim]                                           # (B,4)
        scan = obs_flat[:, self.pose_dim:self.pose_dim + self.height_len]            # (B,651)
        # per-sample z-score（数值稳定、实现简单）
        mean = scan.mean(dim=1, keepdim=True)
        std = scan.std(dim=1, keepdim=True) + 1e-6
        scan_n = (scan - mean) / std                                                 # (B,651)
        h = self.cnn(scan_n.unsqueeze(1).float())                                    # (B,64)
        h = self.fc(h)                                                               # (B,elev_out)
        return torch.cat([pose, h], dim=1)                                           # (B,4+elev_out)
# -----------------------------------------------------------------------------


class ActorCriticCNN(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        # <<< NEW: 自动决定是否启用 height_scan 预处理
        self._actor_preproc = None
        if num_actor_obs >= 655:  # 4 + 651
            # 仅当前 655 维符合 (4,651) 约定时启用；否则保持 None
            self._actor_preproc = _HeightScanPreproc1D(pose_dim=4, height_len=651, elev_out=128)

        # <<< NEW: 如启用预处理，则 actor MLP 的输入维度改为预处理输出维度
        mlp_input_dim_a = getattr(self._actor_preproc, "output_dim", num_actor_obs)
        mlp_input_dim_c = getattr(self._actor_preproc, "output_dim", num_actor_obs)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function（保持不变）
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        Normal.set_default_validate_args(False)

    # <<< NEW: actor 观测预处理（若无预处理器，则原样返回）
    def _preprocess_actor_obs(self, observations):
        if self._actor_preproc is None:
            return observations
        # 兼容：若外部把 extra 维度拼在后面，这里只取前 655 维做预处理，再拼回剩余维度
        B, D = observations.shape
        main = observations[:, :655]                       # (pose4 + height651)
        rest = observations[:, 655:] if D > 655 else None
        fused = self._actor_preproc(main)                  # (B, 4+elev_out)
        return torch.cat([fused, rest], dim=1) if rest is not None else fused

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # <<< NEW: 先做 actor 侧的高程图预处理（若已启用）
        observations = self._preprocess_actor_obs(observations)
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        observations = self._preprocess_actor_obs(observations)  # <<< NEW
        actions_mean = self.actor(observations)
        # print(f"act inference {actions_mean}")
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        critic_observations = self._preprocess_actor_obs(critic_observations)
        value = self.critic(critic_observations)  # critic 路径不变
        return value
