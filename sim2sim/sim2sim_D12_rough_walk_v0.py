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
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch
import os
from joystick import JoyStickController
import time
import pygame
from pygame.locals import *
import queue
import threading
import glob
from datetime import datetime


plot_flag = False


def get_latest_policy_path(base_dir):
    """获取最新的策略模型路径

    Args:
        base_dir: 策略模型的基础目录，例如 "/home/lenovo/rl_project/IsaacLab_perception/logs/rsl_rl/h1_flat"

    Returns:
        最新策略模型的完整路径，如果没有找到则返回None
    """
    # 查找所有策略目录，格式为 "日期时间_随机字符串"
    policy_dirs = glob.glob(os.path.join(base_dir, "*_*"))

    if not policy_dirs:
        print(f"警告: 在 {base_dir} 下未找到策略目录")
        return None

    # 按修改时间排序，获取最新的目录
    latest_dir = max(policy_dirs, key=os.path.getmtime)
    print(f"找到最新策略目录: {latest_dir}")

    # 构建策略模型路径
    policy_model_path = os.path.join(latest_dir, "exported", "policy.pt")
    encoder_module_path = os.path.join(latest_dir, "exported", "encoder_module.pt")
    cv_mu_path = os.path.join(latest_dir, "exported", "cv_mu.pt")

    if not os.path.exists(policy_model_path):
        print(f"警告: 在 {latest_dir} 下未找到策略模型")
        return None
    if not os.path.exists(encoder_module_path):
        print(f"警告: 在 {latest_dir} 下未找到策略模型")
        return None
    if not os.path.exists(cv_mu_path):
        print(f"警告: 在 {latest_dir} 下未找到策略模型")
        return None
    
    return policy_model_path, encoder_module_path, cv_mu_path

# 配置打印选项（全局生效）
np.set_printoptions(
    suppress=True,    # 禁用科学计数法
    precision=6,      # 设置小数位数（可根据需要调整）
    floatmode='fixed' # 固定小数格式
)

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

use_joy =  True

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    '''Extracts an observation from the mujoco data structure

    Args:
        data: MuJoCo数据结构，包含关节位置、速度等状态信息

    Returns:
        元组包含以下观测值:
            q: 关节位置向量
            dq: 关节速度向量
            quat: 四元数表示的朝向 [x, y, z, w]
            v: 基坐标系下的线速度
            omega: 角速度向量
            gvec: 基坐标系下的重力向量投影
    '''
    # 提取关节位置并转换为双精度浮点数
    q = data.qpos.astype(np.double)
    # 提取关节速度并转换为双精度浮点数
    dq = data.qvel.astype(np.double)
    # 提取传感器的四元数朝向数据，并重排为[x, y, z, w]格式
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    # 使用四元数创建旋转矩阵对象
    r = R.from_quat(quat)
    # 将全局坐标系下的线速度转换到基坐标系下
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    # 提取传感器的角速度数据
    omega = data.sensor('angular_velocity').data.astype(np.double)
    # 计算基坐标系下的重力向量投影（全局重力向量为[0, 0, -1]）
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def load_policy(logdir, cfg):
    # 自动获取最新的策略模型路径
    policy_model_path, encoder_module_path, cv_mu_path = get_latest_policy_path(policy_base_dir)
    if policy_model_path is None:
        print("错误: 无法加载策略模型，程序退出")
        exit(1)
    if encoder_module_path is None:
        print("错误: 无法加载策略模型，程序退出")
        exit(1)
    if cv_mu_path is None:
        print("错误: 无法加载策略模型，程序退出")
        exit(1)
    
    # 加载预训练的策略模型
    actor = torch.jit.load(policy_model_path)
    encoder_module = torch.jit.load(encoder_module_path)
    cv_mu = torch.jit.load(cv_mu_path)
    
    def policy(observation):
        h =encoder_module(observation)
        mu = cv_mu(h)
        latent = mu
        single_obs = observation[:,-(cfg.robot_config.num_single_obs+187):]
        actions_mean = actor( torch.cat((latent, single_obs), dim = -1) )
        return actions_mean
    
    
    return policy

def run_mujoco(policy, cfg):
    """
    使用提供的策略和配置运行Mujoco仿真

    参数:
        policy: 用于控制仿真的策略模型
        cfg: 包含仿真设置的配置对象

    返回:
        None
    """
    # 加载Mujoco模型并配置仿真参数
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt  # 设置仿真时间步长
    data = mujoco.MjData(model)             # 创建仿真数据对象

    #-----------------------
    # 获取当前yaw角（忽略pitch/roll）
    quat = data.qpos[3:7]
    yaw = np.arctan2(2*(quat[3]*quat[0] + quat[1]*quat[2]),
                     1 - 2*(quat[0]**2 + quat[1]**2))

    # 锁定激光平台仅跟随yaw
    data.qpos[data.joint("laser_yaw_joint").id] = yaw
    # #------------------------

    #初始化数据
    data.qpos[2] = 1.08
    data.qpos[7:19] = cfg.robot_config.default_joint_pos
    mujoco.mj_step(model, data)            # 执行一步仿真以初始化
    viewer = mujoco_viewer.MujocoViewer(model, data) # 创建可视化查看器
    viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False  # 关闭测距仪可视化

    # 初始化控制变量
    target_q = np.zeros(cfg.robot_config.num_actions, dtype=np.double)  # 目标关节位置
    action = np.zeros(cfg.robot_config.num_actions, dtype=np.double)   # 策略输出的动作

    # 初始化观察历史队列（用于帧堆叠）
    hist_obs = deque()
    for _ in range(cfg.robot_config.frame_stack): # 帧堆叠数量
        hist_obs.append(np.zeros([1, cfg.robot_config.num_single_obs], dtype=np.double))

    count_lowlevel = 0 # 低级控制计数器（用于降采样）
    gait_cycle = cfg.robot_config.gait_cycle # 表示机器人完成一个完整步态循环（左右脚各迈一步）需要 0.7 秒，对应步态频率为 1/0.7 ≈ 1.43 Hz。


    # 创建一个临时变量存储重新排序后的action
    action_reordered = np.zeros_like(action)
    leg_joints = cfg.robot_config.num_actions//2  # 每条腿的关节数，使用整数除法确保leg_joints是整数
    # 定义自定义缩放因子列表
    scale_factors = np.array([
        0.25, 0.25, 0.25, 0.25, 0.1, 0.1,
        0.25, 0.25, 0.25, 0.25, 0.1, 0.1
    ], dtype=np.double)

    gait_phase = np.zeros(2)
    phase_ratio = np.array([cfg.robot_config.gait_air_ratio_l,cfg.robot_config.gait_air_ratio_r])
    phase_offset = np.array([cfg.robot_config.gait_phase_offset_l, cfg.robot_config.gait_phase_offset_r])


    
    # 初始化手柄控制器
    joystick = None
    if use_joy:
        joystick = JoyStickController()
        joystick.start()  # 启动手柄线程



    # 主仿真循环
    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        #是否使用手柄控制
        if use_joy:
            joystick._update_cmd_vel()
            cmd_vel = joystick.get_cmd_vel()
            cmd.vx = cmd_vel["vx"]
            cmd.vy = cmd_vel["vy"]
            cmd.dyaw = cmd_vel["wz"]

        # 从仿真环境获取当前状态
        q, dq, quat, v, omega, gvec = get_obs(data)  # 获取关节位置、速度、四元数、基座线速度、基座的角速度、重力向量在机器人基座坐标系下的投影等
        q = q[-cfg.robot_config.num_actions:]      # 提取感兴趣的关节位置
        dq = dq[-cfg.robot_config.num_actions:]     # 提取感兴趣的关节速度

        ## 控制频率降采样（例如从1000Hz降至100Hz） 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            # 构建当前观察向量
            obs = np.zeros([1, cfg.robot_config.num_single_obs], dtype=np.float32)
            # 四元数转欧拉角
            eu_ang = quaternion_to_euler_array(quat)
            # 将角度归一化到[-π, π]
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            # 计算归一化相位（0-1范围）
            phase = (count_lowlevel * cfg.sim_config.dt / gait_cycle)
            t = np.array([phase])  # 仅包含相位值的数组
            gait_phase[0] = (t + phase_offset[0]) % 1.0
            gait_phase[1] = (t + phase_offset[1]) % 1.0


            # 示例：计算单个机器人的重力投影 # [roll, pitch, yaw] (弧度)
            # euler_angles = torch.tensor([ [eu_ang[0],eu_ang[1], eu_ang[2]] ])
            # projected_gravity = calculate_gravity_projection_euler(euler_angles)
            # print("投影重力projected_gravity = ", projected_gravity)
            # print("投影重力gvec = ", gvec)


            # 填充观察向量
            # 1. 线速度 v [vx, vy, vz]
            # obs[0, 0:3] = v
            # 2. 角速度 omega [ωx, ωy, ωz]
            obs[0, 0:3] = omega
            # 3. 重力向量投影 gvec [gx, gy, gz]
            obs[0, 3:6] = gvec
            # 4. 速度命令 [vx_cmd, vy_cmd, dyaw_cmd]
            obs[0, 6:9] = [cmd.vx, cmd.vy, cmd.dyaw]
            # 5. 关节位置 q (12个元素) - 调整为左腿-右腿交替
            for i in range(leg_joints):
                obs[0, 9 + i*2] = q[i] - cfg.robot_config.default_joint_pos[i]  # 左腿关节i
                obs[0, 10 + i*2] = q[i + leg_joints] - cfg.robot_config.default_joint_pos[i + leg_joints]  # 右腿关节i
            # 6. 关节速度 dq (12个元素) - 调整为左腿-右腿交替
            for i in range(leg_joints):
                obs[0, 21 + i*2] = dq[i]  # 左腿关节i的速度
                obs[0, 22 + i*2] = dq[i + leg_joints]  # 右腿关节i的速度
            # 7. 上一步动作 action (12个元素)
            obs[0, 33:45] = action
            # Gait parameters
            obs[0, 45:47] = np.sin(2 * np.pi * gait_phase) #2
            obs[0, 47:49] = np.cos(2 * np.pi * gait_phase) #2
            obs[0, 49:51] = phase_ratio
		
	    #------------------------------------------------------------------------
	    #  地形高度点 (17 * 11 = 187 个元素)
            height_measurements = []
            for i in range(17):  # x方向17个点
                for j in range(11):  # y方向11个点
                    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"rangefinder_{i}_{j}")
                    distance = data.sensordata[sensor_id]
                    # 计算绝对高度（假设激光从z=0.5m高度向下发射）
                    height = distance  # 1.07是发射点高度
                    height_measurements.append(height)

            # 转换为17x11网格格式
            height_grid = np.array(height_measurements).reshape(17, 11)
            # 将 height_grid 展平
            height_grid_flat = height_grid.flatten()
	        # 计算需要赋值的元素数量
            num_height_grid = height_grid_flat.size
            # 限制高度值在（0.6 ～ 1.4）- 0.5米范围内（1.07），       
            height_grid_flat_clamped = np.clip(height_grid_flat, 0.7, 1.4) - 0.5   #1.07 - 0.5
            
	    #------------------------------------------------------------------------


            # 裁剪观察值以防止数值过大
            obs = np.clip(obs, -18, 18)

            # 更新观察历史队列（滑动窗口）
            hist_obs.append(obs)
            hist_obs.popleft() # 移除最早的观察

            # 将历史观察拼接为策略输入
            policy_input = np.zeros([1, cfg.robot_config.num_observations + num_height_grid], dtype=np.float32)
            for i in range(cfg.robot_config.frame_stack):
                # 将每帧观察按顺序拼接
                policy_input[0, i * cfg.robot_config.num_single_obs : (i + 1) * cfg.robot_config.num_single_obs] = hist_obs[i][0, :]
            # 拼接高程信息
            policy_input[0, cfg.robot_config.num_observations:cfg.robot_config.num_observations+num_height_grid] = height_grid_flat_clamped
	    
	    # 通过策略网络获取动作
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            # 裁剪动作值以防止超出范围
            action = np.clip(action, -18, 18)


            # 将左右交替的action重新排列为先左后右
            for i in range(leg_joints):
                action_reordered[i] = action[i*2]        # 从交替顺序中提取左腿关节
                action_reordered[i+leg_joints] = action[i*2+1]  # 从交替顺序中提取右腿关节

            # 计算目标关节位置（使用重新排序后的action）
            target_q = cfg.robot_config.default_joint_pos + action_reordered * scale_factors


        # 计算目标关节速度（当前设置为零，即无速度控制）
        target_dq = np.zeros(cfg.robot_config.num_actions, dtype=np.double)
        # 使用PD控制器计算关节扭矩
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                         target_dq, dq, cfg.robot_config.kds)  # Calc torques
        # 裁剪扭矩值以防止过大
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        # 应用计算得到的扭矩到仿真环境
        data.ctrl = tau
        

        # 执行一步仿真
        mujoco.mj_step(model, data)
        # 更新可视化
        viewer.render()
        # 增加低级控制计数器
        count_lowlevel += 1


    joystick.stop()  # 停止手柄线程
    viewer.close()


if __name__ == '__main__':
    """
    模拟环境部署脚本：加载预训练模型并在指定地形运行
    """
    current_directory = os.getcwd()
    print("路径：", current_directory)

    class Sim2simCfg():

        class sim_config:
            mujoco_model_path = f'{current_directory}/D12/mjcf/D12_12dof_rough.xml'
            # mujoco_model_path = f'{current_directory}/D12/D12/mjcf/D12_NoArm.xml'
            sim_duration = 60.0
            dt = 0.001
            decimation = 20

        class robot_config:
            num_actions = 12
            num_single_obs = 51
            frame_stack = 10
            num_observations = num_single_obs * frame_stack
            default_joint_pos = np.array([0, 0, -0.1916, 0.4643, -0.2727, 0,
                                          0, 0, -0.1916, 0.4643, -0.2727, 0])
            kps = np.array([150, 100, 150, 170, 40, 25, 150, 100, 150, 170, 40, 25], dtype=np.double)
            kds = np.array([1.5, 1, 1.5, 1.5, 1, 1, 1.5, 1, 1.5, 1.5, 1, 1], dtype=np.double)
            tau_limit = 300. * np.ones(12, dtype=np.double)
            
            gait_air_ratio_l: float = 0.5
            gait_air_ratio_r: float = 0.5
            gait_phase_offset_l: float = 0.5
            gait_phase_offset_r: float = 1.0
            gait_cycle: float = 0.85

            # gait_air_ratio_l = 0.6
            # gait_air_ratio_r = 0.6
            # gait_phase_offset_l = 0.6
            # gait_phase_offset_r = 0.1
            # gait_cycle = 0.5

    # 直接指定策略模型路径（替换为实际路径）
    policy_base_dir = "/home/lenovo/rl_project/Tiangong_LIB/D12-Lab/logs/v0_walk_rough"  # 例如: "/home/user/legged_gym/logs/XBot/model_10000.jit"
    policy = load_policy(policy_base_dir, Sim2simCfg())
    
    run_mujoco(policy, Sim2simCfg())
