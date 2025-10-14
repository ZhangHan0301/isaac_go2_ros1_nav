# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
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
from isaaclab.sim import SimulationCfg as SimCfg
from isaaclab.sim import PhysxCfg

from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.sim as sim_utils

from .terrain_cfg import HfUniformDiscreteObstaclesTerrainCfg
from .terrain import *
from navigation import mdp 

# import isaaclab_tasks.manager_based.navigation.mdp as mdp
from robots.aau_rover_simple import AAU_ROVER_SIMPLE_CFG


# 回合重置
@configclass
class EventCfg:
    """Configuration for events."""
    # reset_state = EventTerm(
    #     func=mdp.reset_root_state_rover,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg(name="robot"),
    #     },
    # )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0),"z":(0.5,0.5), "yaw": (0, 0)},
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

# 网络输出
@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action = mdp.actions.actions_cfg.AckermannActionCfg(
            asset_name="robot",
            wheelbase_length=0.849,
            middle_wheel_distance=0.894,
            rear_and_front_wheel_distance=0.77,
            wheel_radius=0.1,
            min_steering_radius=0.8,
            steering_joint_names=[".*Steer_Revolute"],
            drive_joint_names=[".*Drive_Continuous"],
            offset=-0.0135
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
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        height_scan = ObsTerm(func=mdp.height_scan_nav,
                params={"sensor_cfg": SceneEntityCfg("height_scanner2")},
                clip=(-1.0, 10.0))
    # observation groups
    policy: PolicyCfg = PolicyCfg()

#  奖励值设置
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # distance_to_target = RewTerm(
    #     func=mdp.distance_to_target_reward,
    #     weight=50.0,
    #     params={"command_name": "pose_command"},
    # )
    # reached_target = RewTerm(
    #     func=mdp.reached_target,
    #     weight=5.0,
    #     params={"command_name": "pose_command", "threshold": 0.18},
    # )
    
    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    #速度平滑话性奖励设置
    # oscillation = RewTerm(
    #     func=mdp.oscillation_penalty,
    #     weight=-0.05,
    #     params={},
    # )
    # angle_to_target = RewTerm(
    #     func=mdp.angle_to_target_penalty,
    #     weight=-1.5,
    #     params={"command_name": "pose_command"},
    # )
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
    #     func=mdp.wheel_robot_excessive_speed_penalty,
    #     weight=-0.2,
    #     params={"asset_cfg": SceneEntityCfg(name="robot"),"speed_limit":1.1},
    # )
    
    # obstacle_reward = RewTerm(
    #     func=mdp.obstacle_reward,
    #     weight= 0.2,
    #     params={ "z_threshold": -0.3, 
    #             "d_safe":1.0},
    # )
    


# 随机生成目标点
@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(15.0, 20.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-10.0,10.0), pos_y=(-10.0, 10.0), heading=(-math.pi, math.pi)),
        # ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(3.0, 8.0), pos_y=(0.0, 0.0), heading=(0.0, 0.0)),
    )

# 终止条件
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="Body"), "threshold": 0.2},
    )
    # collision = DoneTerm(
    #     func=mdp.collision_with_obstacles,
    #     params={"sensor_cfg": SceneEntityCfg(
    #         "contact_sensor"), "threshold": 0.1},
    # )




@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""
    
    terrain2 = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(50, 50),
            color_scheme="height",
            sub_terrains={"t1": HfUniformDiscreteObstaclesTerrainCfg(
                seed=0,
                size=(50, 50),
                obstacle_width_range=(0.5, 1.0),
                obstacle_height_range=(0.3, 2.0),
                num_obstacles=80,
                obstacles_distance=2.5,
                border_width=5,
                avoid_positions=[[0, 0]]
            )},
        ),
        visual_material=None,     
    )
    
    # robots
    robot: ArticulationCfg = AAU_ROVER_SIMPLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    
    height_scanner2 = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Body",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[3.0, 2.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        filter_prim_paths_expr=["/World/ground"],
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class AauRoverNavigationEnvCfg(ManagerBasedRLEnvCfg):
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

        self.sim.dt = 1/30
        self.sim.render_interval = 6
        self.decimation = 6
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]*5

        if self.scene.height_scanner2 is not None:
            self.scene.height_scanner2.update_period = self.sim.dt*self.decimation

        if self.scene.contact_sensor is not None:
            self.scene.contact_sensor.update_period = self.sim.dt*self.decimation


