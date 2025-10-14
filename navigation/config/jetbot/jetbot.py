
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

JETBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # 假设USD文件路径（根据实际路径修改）
        usd_path=f"/home/nav/isaac_new/asset/Assets/Isaac/4.5/Isaac/Robots/Jetbot/jetbot.usd",
        activate_contact_sensors=True,
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,   # 缩小接触检测范围
            rest_offset=0.005      # 减小静止间隙
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_linear_velocity=2.0,      # 降低最大线速度
            max_angular_velocity=20.0,    # 减小角速度限制
            max_depenetration_velocity=0.5,  # 降低穿透修正速度
            disable_gravity=False,        # 启用重力
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # 禁用自碰撞
            solver_position_iteration_count=16,  # 物理求解器迭代次数
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),  # 初始高度5cm防止下沉
        joint_pos={
            "left_wheel_joint": 0.0,    # 左轮初始位置
            "right_wheel_joint": 0.0    # 右轮初始位置
        },
        joint_vel={
            "left_wheel_joint": 0.0,    # 左轮初始速度
            "right_wheel_joint": 0.0    # 右轮初始速度
        },
    ),
    actuators={
        "wheel_drive": ImplicitActuatorCfg(  # 差速驱动配置
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],  # 匹配关节名称
            velocity_limit=12.57,  # 最大角速度(120 RPM)
            effort_limit=2.0,      # 扭矩限制(N·m)
            stiffness=400.0,       # 刚度系数
            damping=50.0           # 阻尼系数
        )
    },
    sensors={
        "contact_forces": ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_wheel"),  # 轮子接触检测
        "imu": ImuSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base_link")  # IMU传感器
    }
)