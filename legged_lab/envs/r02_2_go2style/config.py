"""
RO2后空翻训练配置 - 完全复现go2backflip
除了机器人用RO2，其他参数完全照搬go2backflip
"""

import math
from legged_lab.envs.base.base_env_config import BaseEnvCfg, RewardCfg, BaseAgentCfg
from legged_lab.assets.r02_2 import R02_2_CFG
from legged_lab.envs.r02_2_go2style import rewards as go2_rewards
import legged_lab.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg, RslRlPpoActorCriticCfg


class R02Go2StyleEnvCfg(BaseEnvCfg):
    """RO2后空翻环境配置 - go2backflip风格"""

    def __post_init__(self):
        super().__post_init__()

        # 设置机器人
        self.scene.robot = R02_2_CFG

        # 场景配置 - 完全照搬go2backflip
        self.scene.num_envs = 4096
        self.scene.env_spacing = 4.0
        self.scene.max_episode_length_s = 2.0  # go2backflip是2秒

        # 平坦地形
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None

        # PD参数 - 根据RO2电机力矩调整
        self.robot.stiffness = 35.0      # 按力矩比例：17/23.5 × 25 = 18
        self.robot.damping = 0.5         # go2backflip是0.5
        self.robot.action_scale = 0.4    # 降低到0.4，避免动作过大

        # 终止条件：仅超时（无接触终止）
        self.robot.terminate_contacts_body_names = []  # 无接触终止
        self.robot.feet_body_names = [".*3_.*"]

        # 观测配置 - 简化为单帧（go2backflip没有历史）
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1

        # 命令配置：后空翻不需要速度命令
        self.commands.debug_vis = False
        self.commands.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.ranges.ang_vel_z = (0.0, 0.0)

        # 域随机化 - 完全照搬go2backflip
        self.domain_rand.events.reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-3.14, 3.14)},
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

        # 关节初始化随机化 - 降低随机范围，避免初始姿态违反关节限位
        self.domain_rand.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.15, 0.15),  # 降低到±0.15（±8.6°），避免缠绕
                "velocity_range": (0.0, 0.0),
            }
        )

        # 禁用外力扰动（go2backflip没有）
        self.domain_rand.events.push_robot = None

        # 设置质量随机化的body_names（基础配置必需）
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["body"]
        self.domain_rand.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 0.8)  # RO2体重小


class R02Go2StyleRewardCfg(RewardCfg):
    """
    RO2后空翻奖励配置 - 完全复现go2backflip
    权重完全照搬，只改坐标系
    """

    # 1. 终止惩罚 (go2: -5000)
    is_terminated = RewTerm(
        func=mdp.is_terminated,
        weight=-5000.0,
    )

    # 2. 姿态控制 (go2: -2.0)
    orientation_control = RewTerm(
        func=go2_rewards.orientation_control_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # 3. 侧翻惩罚 - Y轴角速度 (go2的ang_vel_x: -1.0)
    ang_vel_y = RewTerm(
        func=go2_rewards.base_ang_vel_y_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # 4. 后空翻奖励 - X轴角速度 (go2的ang_vel_y: 25.0)
    ang_vel_x = RewTerm(
        func=go2_rewards.base_ang_vel_x,
        weight=35.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # 5. 偏航惩罚 (go2: -1.0)
    ang_vel_z = RewTerm(
        func=go2_rewards.base_ang_vel_z_l1,
        weight=-7.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # 6. 起跳奖励 (go2: 35.0)
    lin_vel_z = RewTerm(
        func=go2_rewards.base_lin_vel_z,
        weight=36.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # 7. 高度控制 (go2: -20.0)
    height_control = RewTerm(
        func=go2_rewards.height_control_l2,
        weight=-15.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # 8. 动作对称 (go2: -2.0) -> 增强到-10.0
    action_sym = RewTerm(
        func=go2_rewards.actions_symmetry_l2,
        weight=-12.0,
    )

    # 9. 侧向重力 - X轴 (go2的gravity_y: -10.0)
    gravity_x = RewTerm(
        func=go2_rewards.gravity_x_l2,
        weight=-12.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # 10. 足端站距 (go2: 8.0)
    feet_distance = RewTerm(
        func=go2_rewards.feet_distance_y_exp,
        weight=1.0,
        params={
            "std": math.sqrt(0.25),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*3_.*"]),
            "stance_width": 0.25,  # RO2稍窄，go2是0.3
        },
    )

    # 11. 翻转前足部离地惩罚 (go2: -30.0)
    feet_height_before = RewTerm(
        func=go2_rewards.feet_height_before_backflip_l1,
        weight=-35.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*3_.*"]),
            "height_threshold": 0.1,
        },
    )

    # 12. 腿部接触惩罚 (go2: -2.0)
    undesired_contacts_leg = RewTerm(
        func=go2_rewards.undesired_contacts_foot,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor",
                body_names=[".*1_.*", ".*2_.*"]),  # RO2的髋和大腿
            "threshold": 1.0
        },
    )

    # 13. 躯干接触惩罚 (go2: -10.0)
    undesired_contacts_base = RewTerm(
        func=go2_rewards.undesired_contacts_base,
        weight=-8.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["body"]),
            "threshold": 1.0
        },
    )

    # 14. 动作平滑 (go2: -0.001)
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.001,
    )

    # ========== 新增：膝盖控制奖励 ==========

    # 15. 膝盖触地惩罚（全程）-> 增强到-15.0，扩大检测范围
    knee_contact = RewTerm(
        func=go2_rewards.knee_contact_penalty,
        weight=-10.0,  # 增强惩罚
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*2_.*"]),  # 大腿
            "threshold": 1.0
        },
    )

    # 16. 膝盖高度惩罚（全程）-> 改为惩罚，权重-10.0
    knee_height = RewTerm(
        func=go2_rewards.knee_height_reward,
        weight=-2.0,  # 改为负权重（惩罚低于阈值）
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*2_.*"]),  # 大腿
            "min_height": 0.05,  # 7cm阈值
        },
    )

    # 17. 髋关节角度惩罚（站立+着陆阶段）
    hip_joint_angle = RewTerm(
        func=go2_rewards.hip_joint_angle_penalty,
        weight=-5.0,  # 强惩罚
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 0.3,  # ~17度偏离阈值
        },
    )


@configclass
class R02Go2StyleTerminationsCfg:
    """终止条件 - 仅超时（go2backflip风格）"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


class R02Go2StyleFlatEnvCfg(R02Go2StyleEnvCfg):
    """RO2后空翻平地训练配置"""

    def __post_init__(self):
        super().__post_init__()
        self.reward = R02Go2StyleRewardCfg()
        self.terminations = R02Go2StyleTerminationsCfg()


class R02Go2StyleAgentCfg(BaseAgentCfg):
    """训练Agent配置 - 完全照搬go2backflip"""

    seed = 42
    device = "cuda:0"

    num_steps_per_env = 16  # go2backflip
    max_iterations = 10000
    save_interval = 50  # go2backflip

    experiment_name = "r02_go2style_backflip"

    resume = False  # 从头开始训练
    load_run = None  # 不加载预训练权重
    load_checkpoint = None  # 不加载检查点

    # 网络架构配置（完全照搬go2backflip）
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # PPO配置 - 完全照搬go2backflip
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # go2backflip是0.005
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,  # go2backflip是1e-3
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,  # go2backflip是0.01
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )


class R02Go2StyleFlatAgentCfg(R02Go2StyleAgentCfg):
    """RO2后空翻平地训练Agent配置"""
    pass
