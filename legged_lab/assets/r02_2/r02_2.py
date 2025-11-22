"""Configuration for R02_2 quadruped robot.

Motor specs (defined in URDF):
- Effort limit: 12 N.m
- Velocity limit: 42 rad/s
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

R02_2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/tjz/Ro2_dog/urdf/Ro2_dog.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),  # 初始高度35cm
        joint_pos={
             # ========== 验证过的默认站立姿态 ==========

            # Yaw关节（髋关节外展）：全部保持0（自然站立）
            ".*_joint_yaw": 0.0,

            # 左前腿 (lf: left front)
            "lf2_joint_pitch": -0.3,
            "lf3_joint_pitch": 0.9,

            # 左后腿 (lb: left back)
            "lb2_joint_pitch": -0.3,
            "lb3_joint_pitch": 0.9,

            # 右前腿 (rf: right front)
            "rf2_joint_pitch": 0.3,
            "rf3_joint_pitch": -0.9,

            # 右后腿 (rb: right back)
            "rb2_joint_pitch": 0.3,
            "rb3_joint_pitch": -0.9,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint_yaw", ".*_joint_pitch"],
            # PD控制器参数 - 参考Go1/ANYmal标准配置
            # 电机specs: effort=12 N.m, velocity=42 rad/s
            effort_limit=17.0,  # 提高力矩限制以支持后空翻
            stiffness=20.0,   # 提高刚度，足以支撑1.77kg躯体
            damping=0.5,      # 适中阻尼，平衡响应速度和稳定性
        ),
    },
)
