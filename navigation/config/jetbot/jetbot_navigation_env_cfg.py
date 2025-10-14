# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math

from sympy import true
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim import SimulationCfg as SimCfg
from isaaclab.sim import PhysxCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.sim as sim_utils

from .terrain_cfg import HfUniformDiscreteObstaclesTerrainCfg
from navigation import mdp
from isaaclab.actuators import ImplicitActuatorCfg


JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Jetbot/jetbot.usd"),
    
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-1.0, -0.0, 0.0),
    ),
    
    actuators={"wheel_acts": ImplicitActuatorCfg(
        joint_names_expr=[".*"], 
        effort_limit_sim=100.0,
        velocity_limit_sim=2.0,
        damping=None, 
        stiffness=None)},
    
)

# 回合重置配置
@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, -1.0), 
                "y": (-0.0, 0.0),
                "z": (0.0, 0.0),  # 降低初始高度
                "yaw": (0, 0)
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

# 动作空间配置（差速驱动）
@configclass
class ActionsCfg:
    differential_action = mdp.actions.actions_cfg.DifferentialDriveActionCfg(
        asset_name="robot",
        # wheel_radius=0.03,        # 车轮半径
        wheel_distance=0.12,      # 轮距
        # drive_joint_names=["left_wheel_joint", "right_wheel_joint"],
    )

# 观测空间配置
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        height_scan = ObsTerm(
            func=mdp.height_scan_nav,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-0.1, 0.5)  # 调整高度扫描范围
        )
    
    policy: PolicyCfg = PolicyCfg()

# 奖励函数配置
@configclass
class RewardsCfg:
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=25.0,
        params={"std": 1.5, "command_name": "pose_command"},
    )
    distance_to_target = RewTerm(
        func=mdp.distance_to_target_reward,
        weight=50.0,
        params={"command_name": "pose_command"},
    )
    
    reached_target = RewTerm(
        func=mdp.reached_target,
        weight=5.0,
        params={"command_name": "pose_command", "threshold": 0.18},
    )
    
    obstacle_reward = RewTerm(
        func=mdp.obstacle_reward,
        weight= 0.1,
        params={ "sensor_cfg":SceneEntityCfg("height_scanner"),"z_threshold": -0.3, 
                "d_safe":0.2},
    )
    
    excessive_speed_penalty = RewTerm(
        func=mdp.wheel_robot_excessive_speed_penalty,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg(name="robot"),"speed_limit":1.1},
    )
    
    yaw_alignment_reward = RewTerm(
        func=mdp.yaw_alignment_reward,
        weight=0.3, 
        params={"command_name": "pose_command"},
    )

# 目标生成配置
@configclass
class CommandsCfg:
    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(10.0, 15.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-10.0, -10.0), 
            pos_y=(-5.0, 5.0), 
            heading=(-math.pi, math.pi)  # 限制转向范围
        ),
    )

# 终止条件配置
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # tip_over = DoneTerm(
    #     func=mdp.body_orientation_violation,
    #     params={"threshold": 0.8, "asset_cfg": SceneEntityCfg("robot")}
    # )
    # collision = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 1.0}
    # )

# 场景配置
@configclass
class JetbotSceneCfg(InteractiveSceneCfg):

    terrain2 = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            # num_cols=2,
            # num_rows= 2,
            size=(30, 30),
            color_scheme="height",
            sub_terrains={"t1": HfUniformDiscreteObstaclesTerrainCfg(
                seed=0,
                size=(20, 20),
                obstacle_width_range=(0.5, 1.0),
                obstacle_height_range=(0.5, 1.5),
                num_obstacles=50,
                obstacles_distance=2.5,
                border_width=3,
                avoid_positions=[[0, 0.0]]
            )},
        ),
        visual_material=None,     
    )
    
    # terrain = TerrainImporterCfg(prim_path="/World/ground", 
    #             terrain_type="usd", 
    #             usd_path="/home/nav/isaac_new/asset/Assets/Isaac/4.5/Isaac/Environments/my_scence/narrow_corrier/World1/World0.usd")

    robot: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 5.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    # contact_sensor = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/.*",
    #     filter_prim_paths_expr=["/World/ground"]
    # )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0)
    )

# 主环境配置
@configclass
class JetbotNavigationEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneEntityCfg = JetbotSceneCfg(num_envs=128, env_spacing=2.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

      # Setup PhysX Settings
    sim: SimCfg = SimCfg(
        physx=PhysxCfg(
            enable_stabilization=True,
            gpu_max_rigid_contact_count=8388608,
            gpu_max_rigid_patch_count=262144,
            gpu_found_lost_pairs_capacity=2**21,
            gpu_found_lost_aggregate_pairs_capacity=2**25,  # 2**21,
            gpu_total_aggregate_pairs_capacity=2**21,   # 2**13,
            gpu_max_soft_body_contacts=1048576,
            gpu_max_particle_contacts=1048576,
            gpu_heap_capacity=67108864,
            gpu_temp_buffer_capacity=16777216,
            gpu_max_num_partitions=8,
            gpu_collision_stack_size=2**28,
            friction_correlation_distance=0.025,
            friction_offset_threshold=0.04,
            bounce_threshold_velocity=2.0,
        )
    )
    
    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = 0.005
        self.sim.render_interval = 4
        self.decimation = 4
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]*10

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.sim.dt*self.decimation

        # if self.scene.contact_sensor is not None:
        #     self.scene.contact_sensor.update_period = self.sim.dt*self.decimation
