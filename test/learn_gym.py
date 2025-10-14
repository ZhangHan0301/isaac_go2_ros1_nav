import gym
from gym import spaces
import numpy as np
import torch
from abc import ABC, abstractmethod


# ===================== 底层环境 =====================
class CounterEnv(gym.Env):
    """简单计数环境"""
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(2)  # 0:+1, 1:-1
        self.observation_space = spaces.Box(low=-100, high=100, shape=(1,), dtype=np.int32)
        self.state = 0
        self.episode_length_buf = 0  # 底层 buffer

    def reset(self):
        self.state = 0
        self.episode_length_buf = 0
        return np.array([self.state], dtype=np.int32)

    def step(self, action):
        self.state += 1 if action == 0 else -1
        self.episode_length_buf += 1
        reward = -1
        done = self.state == 10
        if done:
            reward = 10
        return np.array([self.state], dtype=np.int32), reward, done, {}

    def render(self, mode="human"):
        print(f"State: {self.state}, Step buffer: {self.episode_length_buf}")


# ===================== VecEnv 抽象基类 =====================
class VecEnv(ABC):
    def __init__(self):
        self.num_envs = 0

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_async(self, actions):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


# ===================== RSL-RL 风格 VecEnv 封装 =====================
class RslRlVecEnvWrapper(VecEnv):
    """封装多个 CounterEnv，支持 episode_length_buf"""
    def __init__(self, env_fns):
        super().__init__()
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self._actions = None

    @property
    def episode_length_buf(self):
        # 返回所有 env 的 buffer
        return np.array([env.episode_length_buf for env in self.envs])

    @episode_length_buf.setter
    def episode_length_buf(self, values: torch.Tensor):
        # 支持批量写入
        for env, val in zip(self.envs, values):
            env.episode_length_buf = val.item()

    def reset(self):
        return np.array([env.reset() for env in self.envs])

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, self._actions):
            o, r, d, i = env.step(action)
            obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(i)
        return np.array(obs), np.array(rewards), np.array(dones), infos


# ===================== 测试 =====================
if __name__ == "__main__":
    # 创建 3 个环境
    env_fns = [CounterEnv for _ in range(3)]
    env = RslRlVecEnvWrapper(env_fns)

    obs = env.reset()
    print("初始状态:", obs, "buffer:", env.episode_length_buf)

    # 修改 buffer
    env.episode_length_buf = torch.tensor([5, 10, 15])
    print("修改后的 buffer:", env.episode_length_buf)

    max_steps = 10  # 最多运行 10 步
    for step in range(max_steps):
        actions = [e.action_space.sample() for e in env.envs]
        obs, rewards, done, infos = env.step(actions)
        print("Obs:", obs, "Reward:", rewards, "Done:", done, "Buffer:", env.episode_length_buf)
