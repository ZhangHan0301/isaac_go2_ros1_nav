# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
import sys
print(sys.path)
from legged_gym.envs import *
from legged_gym.utils.helpers import  get_args, export_policy_as_jit_actor, export_policy_as_jit_encoder,class_to_dict
from legged_gym.utils.logger import Logger
from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.base.history_wrapper import HistoryWrapper
import numpy as np
import torch
import pickle
import logging
import time

from rsl_rl.modules.actor_critic_wrapper import ActorCritic_Wrapper


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    class_to_dict(env_cfg)
    class_to_dict(train_cfg)
    
    with open('parameters.pkl', 'wb') as f:
        pickle.dump(class_to_dict(env_cfg), f)
    with open('train_cfg.pkl', 'wb') as f:
        pickle.dump(train_cfg, f)
    # override some parameters for testing
    env_cfg.terrain.border_size = 10
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.center_robots = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.test = True
    
    env_cfg.x_command = 1.0
    env_cfg.y_command = 0.0
    env_cfg.yaw_command = 0.0
    
    # env_cfg.sim_params = "cuda:0"

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env = HistoryWrapper(env)
    # load policy
    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    # to jit
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    export_policy_as_jit_actor(runner.alg.actor_critic, path=path)
    export_policy_as_jit_encoder(runner.alg.actor_critic, path=path)

    # to onnx
    model = runner.alg.actor_critic.to("cpu")
    wrapped_model = ActorCritic_Wrapper(model)

    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    pt_model_path = os.path.join(path, 'entire_actor_critic_model.pt')
    onnx_model_path = os.path.join(path, 'actor_critic.onnx')

    torch.save(wrapped_model, pt_model_path)

    dummy_obs = torch.zeros((1, 45 + 8 + 4 + 4))
    dummy_hist_obs = torch.zeros((1, (45 + 8 + 4 + 4)*10))
    input_names = ["observations", "observation_history"]
    output_names = ["action"]
    torch.onnx.export(
        wrapped_model,
        (dummy_obs, dummy_hist_obs),
        onnx_model_path,
        opset_version=11,
        input_names=input_names,
        output_names=output_names,
    )
    print("Model exported to ONNX.")


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)

