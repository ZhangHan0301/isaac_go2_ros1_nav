from isaacsim import SimulationApp

# launch omniverse app
simulation_app = SimulationApp({"headless": False, "anti_aliasing": 0,
                                "width": 1280, "height": 720, 
                                "hide_ui": True})
# import rclpy
import torch
import omni
import carb
import go2.go2_ctrl as go2_ctrl
# import ros2.go2_ros2_bridge as go2_ros2_bridge
from go2.go2_env import Go2RSLEnvCfg, camera_follow
import env.sim_env as sim_env
import go2.go2_sensors as go2_sensors
import time
import os
import hydra
from omni.isaac.lab.sensors import RayCasterCfg, patterns,RayCaster
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
import numpy as np

from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
torch.set_printoptions(precision=4, sci_mode=False) 

def quaternion_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    # quat = quat / torch.norm(quat, dim=1, keepdim=True)  # 归一化
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # 计算旋转矩阵元素
    x2, y2, z2 = x**2, y**2, z**2
    xy, xz, yz = x*y, x*z, y*z
    xw, yw, zw = x*w, y*w, z*w
    
    r00 = 1 - 2*y2 - 2*z2
    r01 = 2*xy - 2*zw
    r02 = 2*xz + 2*yw
    
    r10 = 2*xy + 2*zw
    r11 = 1 - 2*x2 - 2*z2
    r12 = 2*yz - 2*xw
    
    r20 = 2*xz - 2*yw
    r21 = 2*yz + 2*xw
    r22 = 1 - 2*x2 - 2*y2
    
    R = torch.stack([
        torch.stack([r00, r01, r02], dim=1),
        torch.stack([r10, r11, r12], dim=1),
        torch.stack([r20, r21, r22], dim=1)
    ], dim=1)
    return R



    # 创建发布器
lidar_pub = rospy.Publisher('/lidar_points', PointCloud2, queue_size=10)

FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")
print(FILE_PATH)
@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg):
    # Go2 Environment setup
    go2_env_cfg = Go2RSLEnvCfg()
    go2_env_cfg.scene.num_envs = cfg.num_envs
    go2_ctrl.init_base_vel_cmd(cfg.num_envs)
    # env, policy = go2_ctrl.get_rsl_flat_policy(go2_env_cfg)
    env, policy = go2_ctrl.get_rsl_rough_policy(go2_env_cfg)

    # # Simulation environment
    # if (cfg.env_name == "obstacle-dense"):
    #     sim_env.create_obstacle_dense_env() # obstacles dense
    # elif (cfg.env_name == "obstacle-medium"):
    #     sim_env.create_obstacle_medium_env() # obstacles medium
    # elif (cfg.env_name == "obstacle-sparse"):
    #     sim_env.create_obstacle_sparse_env() # obstacles sparse
    # elif (cfg.env_name == "warehouse"):
    #     sim_env.create_warehouse_env() # warehouse
    # elif (cfg.env_name == "warehouse-forklifts"):
    #     sim_env.create_warehouse_forklifts_env() # warehouse forklifts
    # elif (cfg.env_name == "warehouse-shelves"):
    #     sim_env.create_warehouse_shelves_env() # warehouse shelves
    # elif (cfg.env_name == "full-warehouse"):
    #     sim_env.create_full_warehouse_env() # full warehouse

    # Sensor setup
    sm = go2_sensors.SensorManager(cfg.num_envs)
    lidar_annotators = sm.add_rtx_lidar()
    cameras = sm.add_camera()

    # Keyboard control
    system_input = carb.input.acquire_input_interface()
    system_input.subscribe_to_keyboard_events(
        omni.appwindow.get_default_app_window().get_keyboard(), go2_ctrl.sub_keyboard_event)
    

    # Run simulation
    sim_step_dt = float(go2_env_cfg.sim.dt * go2_env_cfg.decimation)
    obs, _ = env.reset()
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():            
            # control joints
            lidar_data = lidar_annotators[0].get_data()["data"].reshape(-1, 3)
            # print("lidar data size: ", lidar_data.shape)
            # print("lidar_annotators: ",lidar_annotators[0].get_data()["data"].reshape(-1, 3))
            # 将点云数据转换为 ROS PointCloud2 格式
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "base_link"  # 设定坐标系名称

            # 将点云转换为 PointCloud2 格式
            pc_data = pc2.create_cloud_xyz32(header, lidar_data)
            lidar_pub.publish(pc_data)
            
            actions = policy(obs)

            # step the environment
            obs, reward, done, _ = env.step(actions)
            # print("join pose ", obs.flatten()[12:24])
            sensor: RayCaster = env.env.scene["height_scanner2"]
            ray_pos_w = sensor.data.pos_w
            robot_quat = sensor.data.quat_w
            qw = robot_quat[:, 0]
            qx = robot_quat[:, 1]
            qy = robot_quat[:, 2]
            qz = robot_quat[:, 3]
            yaw_base = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            # print("yaw_ base ", yaw_base)
            # print("pos_w ", ray_pos_w)
            # print("quat ", robot_quat)
            # print("ray hit ", sensor.data.ray_hits_w)
            ray_hit_w = sensor.data.ray_hits_w - ray_pos_w.unsqueeze(1)
            third_elements = ray_hit_w[:, :, 2]
            mask = third_elements > -0.3
            
            
            
            obstacle_point_xy = ray_hit_w[...,:2]
            obstacle_distance = torch.norm(obstacle_point_xy, p=2, dim=2)  # 形状 (2, 3)
            print("raw ray hit ", ray_hit_w)
            ray_hit_w[:, :, 2] = torch.where(mask, obstacle_distance, torch.zeros_like(third_elements))
            print("ray hit w ", ray_hit_w)
            # print(obstacle_distance)
            # print(obstacle_point_xy)
            # ray_hit_np = obstacle_point_xy.cpu().numpy()
            # with open("ray_hit.txt", "w") as f:
            #     f.write("ray hit \n")
            #     np.savetxt(f, ray_hit_np, fmt="%.3f")
            # print("translated points ", translated_points)
            R = quaternion_to_rotation_matrix(robot_quat)
            
            R_transposed = R.transpose(1, 2)
            ray_hits_local = torch.bmm(ray_hit_w, R)
            # print("p robot ", ray_hits_local)
            
            
            # Camera follow
            if (cfg.camera_follow):
                camera_follow(env)
                
            if(done):
                print("\n\n robot is done\n\n\n")
                env.reset()
                # env.env.scene["contact_forces"].reset()
                # env.env.scene.update(0.1)
                
                
            
            

            # limit loop time
            elapsed_time = time.time() - start_time
            if elapsed_time < sim_step_dt:
                sleep_duration = sim_step_dt - elapsed_time
                time.sleep(sleep_duration)
        actual_loop_time = time.time() - start_time
        rtf = min(1.0, sim_step_dt/elapsed_time)
        # print(f"\rStep time: {actual_loop_time*1000:.2g}ms, Real Time Factor: {rtf:.2g}\n", end='', flush=True)
    
    # dm.destroy_node()
    # rclpy.shutdown()

if __name__ == "__main__":
    rospy.init_node('lidar_point_cloud_publisher', anonymous=True)
    
    run_simulator()
    simulation_app.close()
