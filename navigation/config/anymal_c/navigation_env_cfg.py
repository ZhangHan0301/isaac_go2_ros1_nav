# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from sympy import false, true
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.sim as sim_utils

from navigation.mdp.rewards import progressive_reward
from navigation.terrain_cfg.terrain_cfg import HfUniformDiscreteObstaclesTerrainCfg
# from .terrain import *
from navigation import mdp 

# import isaaclab_tasks.manager_based.navigation.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

LOW_LEVEL_ENV_CFG = AnymalCFlatEnvCfg()

# 回合重置
@configclass
class EventCfg:
    """Configuration for events."""
     # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),  #静态摩擦力
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0), #弹性
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-0.0, 0.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.5, 1.5), "y": (2.5, 3.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )

    
# 网络输出
@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        # policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt",
        policy_path=f"{ISAACLAB_NUCLEUS_DIR}/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Anymal-C-v0/exported/policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )

# 网络输入
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pose_command = ObsTerm(func=mdp.generated_commands, 
                               params={"command_name": "pose_command"})
        # height_scan = ObsTerm(func=mdp.height_scan_nav,
        #         params={"sensor_cfg": SceneEntityCfg("height_scanner2")},
        #         noise=Unoise(n_min=-0.1, n_max=0.1),
        #         clip=(-1.0, 1.5))
        lidar_scan = ObsTerm(func=mdp.lidar_distance_scan_horizontal,
                            params={"sensor_cfg": SceneEntityCfg("lidar_scanner")})
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    # class TestCfg(ObsGroup):
    #     """Observations for policy group."""

    #     # pose_command2 = ObsTerm(func=mdp.generated_commands, 
    #     #                        params={"command_name": "pose_command"})
    #     height_scan2 = ObsTerm(func=mdp.height_scan_to_polar,
    #             params={"sensor_cfg": SceneEntityCfg("height_scanner3"), "N":6},
    #             # noise=Unoise(n_min=-0.1, n_max=0.1),
    #             clip=(-1.0, 1.5))

            
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    # test:TestCfg = TestCfg()
    



#  奖励值设置
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 30.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 0.3, "command_name": "pose_command"},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.05,
        params={"command_name": "pose_command"},
    )
    
    reach_goal = RewTerm(func=mdp.object_reached_goal_reward,
        weight = 10.0,
        params={"command_name": "pose_command", "threshold": 0.1},
    )
    
    # vel_command_reward = RewTerm(func=mdp.velocity_command_error,
    #                              weight= 0.5,
    #                              params={"command_name": "pose_command"})

    
    # distance_to_target = RewTerm(
    #     func=mdp.distance_to_target_reward,
    #     weight=50.0,
    #     params={"command_name": "pose_command"},
    # )
    reached_target = RewTerm(
        func=mdp.reached_target,
        weight=2.0,
        params={"command_name": "pose_command", "threshold": 0.18},
    )
    
    progressive_reward = RewTerm(
        func=mdp.progressive_reward,
        weight=2.0,
        params={"command_name": "pose_command", "max_distance": 15.0},
    )
    # oscillation = RewTerm(
    #     func=mdp.oscillation_penalty,
    #     weight=-0.01,
    #     params={},
    # )
    # angle_to_target = RewTerm(
    #     func=mdp.angle_to_target_penalty,
    #     weight=-1.5,
    #     params={"command_name": "pose_command"},
    # )
    
    # yaw_alignment_reward = RewTerm(
    #     func=mdp.yaw_alignment_reward,
    #     weight=2.0, 
    #     params={"command_name": "pose_command"},
    # )
    #后退惩罚
    # heading_soft_contraint = RewTerm(
    #     func=mdp.heading_soft_contraint,
    #     weight=-0.5,
    #     params={"asset_cfg": SceneEntityCfg(name="robot")},
    # )

    # far_from_target = RewTerm(
    #     func=mdp.far_from_target_reward,
    #     weight=-2.0,
    #     params={"command_name": "pose_command", "threshold": 11.0},
    # )
    # angle_diff = RewTerm(
    #     func=mdp.angle_to_goal_reward,
    #     weight=5.0,
    #     params={"command_name": "pose_command"},
    # )
    # excessive_speed_penalty = RewTerm(
    #     func=mdp.excessive_speed_penalty,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg(name="robot"),"speed_limit":1.5},
    # )
    
    # obstacle_reward = RewTerm(
    #     func=mdp.obstacle_reward,
    #     weight= 0.8,
    #     params={ "z_threshold": -0.3, 
    #             "d_safe":0.50},
    # )
    

# 随机生成目标点
@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(5.0, 15.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-2.0,2.0), pos_y=(3.5, 6.0), heading=(-math.pi, math.pi)),
        # ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(3.0, 8.0), pos_y=(0.0, 0.0), heading=(0.0, 0.0)),
    )

# 终止条件
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )



@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""
    
    terrain2 = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(30,30),
            color_scheme="height",
            sub_terrains={"t1": HfUniformDiscreteObstaclesTerrainCfg(
                seed=0,
                size=(30, 30),
                obstacle_width_range=(0.5, 2.0),
                obstacle_height_range=(1.0, 1.5),
                num_obstacles=50,
                obstacles_distance=2.0,
                border_width=0.1,
                platform_width=2.0,
                avoid_positions=[[13, 15.5],[0.0,-0.2],[0.5,1.0],[5.0, 30],[10.0,29]]
            )},
        ),
        visual_material=None,     
    )
    
    # robots
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    
    # height_scanner2 = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[3.0, 2.0]),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )
    
    lidar_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),  # 安装高度0.2m
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,  # 单线激光雷达
            vertical_fov_range=(0, 0),  # 垂直角度为0，水平扫描
            horizontal_fov_range=(-180, 180),  # 360度全覆盖
            horizontal_res=10.0,  # 1度分辨率
        ),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],  # 空列表，检测所有障碍物
        max_distance=3.0,  # 最大检测距离3m
    )
    
    # height_scanner3 = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.5, size=[3.0, 2.0]),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )
        
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_command_pose, params={"term_name": "pose_command", "num_steps": 100}
    )

@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # environment settings
    scene: SceneEntityCfg = MySceneCfg(num_envs=1, env_spacing=2.5)
    # scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 5
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]*3
        # self.scene.terrain2.terrain_type = "plane"
        # self.scene.terrain2.terrain_generator = None
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


class NavigationEnvCfg_PLAY(NavigationEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
