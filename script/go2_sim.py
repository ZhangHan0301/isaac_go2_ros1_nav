# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

import keyboard

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import sys
from pathlib import Path

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import navigation
import isaaclab_tasks  # noqa: F401
import velocity.anymal_c
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
import go2_ctrl

import threading
from pynput import keyboard

vel_command = torch.zeros(3)
# 控制函数，用来根据键盘输入来改变 vel_command 中的值
def on_press(key):
    global vel_command
    try:
        if key.char == 'w':  # 按下 'w' 键
            vel_command[0] += 0.1  # 增加前进速度
            print("Moving forward!")
        elif key.char == 's':  # 按下 's' 键
            vel_command[0] -= 0.1  # 减少前进速度（后退）
            print("Moving backward!")
        elif key.char == 'a':  # 按下 'a' 键
            vel_command[1] += 0.1  # 增加左转角度
            print("Turning left!")
        elif key.char == 'd':  # 按下 'd' 键
            vel_command[1] -= 0.1  # 减少右转角度
            print("Turning right!")
        elif key.char == 'q':  # 按下 'q' 键
            vel_command[2] += 0.1  # 增加第三个控制维度（例如升高等动作）
            print("Performing action Q!")
        elif key.char == 'e':  # 按下 'e' 键
            vel_command[2] -= 0.1  # 减少第三个控制维度（例如下降等动作）
            print("Performing action E!")
        elif key.char == 'z':  # 按下 'z' 键
            vel_command[:] = 0  # 重置所有控制值
            print("Resetting all controls!")
            # return False  # 停止监听
    except AttributeError:
        pass  # 忽略其他特殊键（例如按键盘上的功能键）

# 创建监听器
def control_robot():
    print("Press 'W' to move forward, 'S' to move backward, 'A' to turn left, 'D' to turn right.")
    print("Press 'Q' for action Q, 'E' for action E, 'Z' to reset all controls.")
    print("Press 'Z' to quit.")
    
    # 启动键盘监听
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

   
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.physics_dt

    # reset environment
    obs, _ = env.get_observations()
    
    timestep = 0
    
    
    # 启动键盘监听线程
    control_thread = threading.Thread(target=control_robot, daemon=True)
    control_thread.start()
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        print("=================================================")
        print(f"Current vel_command: {vel_command}")
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            obs[0,9] = vel_command[0]
            obs[0,10] = vel_command[1]
            obs[0,11] = vel_command[2]
         
            actions = policy(obs)
            # env stepping
            obs, _, _, extras = env.step(actions)
            print("obs ", extras["observations"]["test"])
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
