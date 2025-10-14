from cv2 import line
import torch

import numpy as np

# policy_path = "ckpts/unitree_go2/go2/model_9000.pt"
policy_path = "/home/nav/isaac_new/RLRoverLab/rover_envs/envs/navigation/robots/aau_rover/policies/best_agent2.pt"
policy = torch.load(policy_path, weights_only=True)

print("policy ",policy)


goal_command = np.zeros(4)
goal_command[0] = 9.9
goal_command[1] = -0.96
goal_command[3] = -0.88

line_vel_command = np.zeros(3)
lidar_input = np.zeros(961) -0.6

# 拼接成一个一维数组
combined_array = np.concatenate([line_vel_command,goal_command, lidar_input])
input_tensor = torch.from_numpy(combined_array).float()
with torch.inference_mode():
    action = policy(input_tensor)
    print("action ", action)
    
    # 截断虑波
    if(abs(action[0])>1.0):
        action[0] = np.sign(action[0])*1
    if abs(action[1])>1.0: 
        action[1] = np.sign(action[1])*1
    if abs(action[2])>1.0:
        action[2] = np.sign(action[2])*1