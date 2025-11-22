"""
R02_2后空翻专用奖励函数
基于Stage-Wise-CMORL方法，结合legged_gym和parkour的最佳实践

Stage划分:
  Stage 0: Stand  - 站立准备（随机时间触发）
  Stage 1: Down   - 下蹲蓄力（COM降低，保持接触）
  Stage 2: Jump   - 起跳腾空（后足发力，开始pitch旋转）
  Stage 3: Turn   - 空中翻转（持续旋转，腾空）
  Stage 4: Land   - 着陆稳定（恢复直立，四足着地）

核心设计原则:
1. 每个Stage有不同的奖励权重组合
2. Stage自动切换基于物理条件（不依赖时间步）
3. 使用one-hot编码的stage_buf进行阶段式奖励
4. 引入半圈和全圈检测防止过度旋转
"""

import torch
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_from_euler_xyz
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv


# ==================== RSI - Reference State Initialization ====================

def reset_root_state_backflip_rsi(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    后空翻专用RSI初始化：随机初始化到不同阶段状态
    - 30% 正常站立
    - 30% 空中状态（高度0.35-0.5m，z速度1-2m/s）
    - 40% 翻转中途（pitch 45-90°，角速度2-4 rad/s）
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取默认root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # 随机决定每个环境的初始化模式
    rand_vals = torch.rand(len(env_ids), device=env.device)

    # 模式掩码
    normal_mask = rand_vals < 0.3  # 30% 正常
    airborne_mask = (rand_vals >= 0.3) & (rand_vals < 0.6)  # 30% 空中
    flip_mask = rand_vals >= 0.6  # 40% 翻转中

    # --- 正常站立模式 ---
    if normal_mask.any():
        normal_ids = normal_mask.nonzero(as_tuple=False).squeeze(-1)
        # 标准位置随机
        root_states[normal_ids, 0] += torch.empty(len(normal_ids), device=env.device).uniform_(*pose_range.get("x", (-0.5, 0.5)))
        root_states[normal_ids, 1] += torch.empty(len(normal_ids), device=env.device).uniform_(*pose_range.get("y", (-0.5, 0.5)))
        # yaw随机
        yaw = torch.empty(len(normal_ids), device=env.device).uniform_(*pose_range.get("yaw", (-3.14, 3.14)))
        quat = quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
        )
        root_states[normal_ids, 3:7] = quat

    # --- 空中状态模式 ---
    if airborne_mask.any():
        air_ids = airborne_mask.nonzero(as_tuple=False).squeeze(-1)
        # 高度 0.35-0.5m
        root_states[air_ids, 2] = torch.empty(len(air_ids), device=env.device).uniform_(0.35, 0.5)
        # XY随机
        root_states[air_ids, 0] += torch.empty(len(air_ids), device=env.device).uniform_(*pose_range.get("x", (-0.5, 0.5)))
        root_states[air_ids, 1] += torch.empty(len(air_ids), device=env.device).uniform_(*pose_range.get("y", (-0.5, 0.5)))
        # Z速度 1-2 m/s (向上)
        root_states[air_ids, 9] = torch.empty(len(air_ids), device=env.device).uniform_(1.0, 2.0)
        # yaw随机
        yaw = torch.empty(len(air_ids), device=env.device).uniform_(*pose_range.get("yaw", (-3.14, 3.14)))
        quat = quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
        )
        root_states[air_ids, 3:7] = quat

    # --- 翻转中途模式 ---
    if flip_mask.any():
        flip_ids = flip_mask.nonzero(as_tuple=False).squeeze(-1)
        # 高度 0.4-0.55m (翻转时更高)
        root_states[flip_ids, 2] = torch.empty(len(flip_ids), device=env.device).uniform_(0.4, 0.55)
        # XY随机
        root_states[flip_ids, 0] += torch.empty(len(flip_ids), device=env.device).uniform_(*pose_range.get("x", (-0.5, 0.5)))
        root_states[flip_ids, 1] += torch.empty(len(flip_ids), device=env.device).uniform_(*pose_range.get("y", (-0.5, 0.5)))
        # pitch 45-90° (0.785-1.57 rad) - RO2是Y在前，后空翻绕X轴
        roll = torch.empty(len(flip_ids), device=env.device).uniform_(0.785, 1.57)  # 翻转角度
        yaw = torch.empty(len(flip_ids), device=env.device).uniform_(*pose_range.get("yaw", (-3.14, 3.14)))
        quat = quat_from_euler_xyz(
            roll, torch.zeros_like(roll), yaw
        )
        root_states[flip_ids, 3:7] = quat
        # 角速度 2-4 rad/s (绕X轴)
        root_states[flip_ids, 10] = torch.empty(len(flip_ids), device=env.device).uniform_(2.0, 4.0)
        # Z速度 0.5-1.5 m/s
        root_states[flip_ids, 9] = torch.empty(len(flip_ids), device=env.device).uniform_(0.5, 1.5)

    # 设置root state
    asset.write_root_pose_to_sim(root_states[:, :7], env_ids)
    asset.write_root_velocity_to_sim(root_states[:, 7:13], env_ids)


# ==================== 阶段切换逻辑 ====================

def update_backflip_stages(env: "BaseEnv") -> None:
    """
    更新后空翻阶段状态（时间驱动版本）

    3秒Episode时间分配（按go2backflip的2秒比例缩放）：
    - Stage 0 (Stand):  0 - 0.45s   (15%)
    - Stage 1 (Down):   0.45 - 0.75s (10%)
    - Stage 2 (Jump):   0.75 - 1.05s (10%)
    - Stage 3 (Turn):   1.05 - 1.8s  (25%)
    - Stage 4 (Land):   1.8 - 3.0s   (40%)
    """
    asset: Articulation = env.scene["robot"]
    current_time = env.episode_length_buf * env.step_dt

    # 获取body_z用于翻转检测
    body_quat = asset.data.root_quat_w
    body_z = math_utils.quat_rotate_inverse(body_quat, torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1))

    # 检测翻转状态（RO2坐标系：Y轴在前，绕X轴旋转）
    # 半圈: body_z的Y和Z分量都为负（倒立后继续旋转）
    is_half_turn_now = torch.logical_and(body_z[:, 1] < 0, body_z[:, 2] < 0)
    env.is_half_turn_buf = torch.logical_or(env.is_half_turn_buf, is_half_turn_now.long())

    # 全圈: 在半圈之后，body_z的Y和Z分量都恢复为正
    is_one_turn_now = torch.logical_and(
        env.is_half_turn_buf.bool(),
        torch.logical_and(body_z[:, 1] >= 0, body_z[:, 2] >= 0)
    )
    env.is_one_turn_buf = torch.logical_or(env.is_one_turn_buf, is_one_turn_now.long())

    # 时间驱动的Stage分配（清零后重新设置）
    env.stage_buf.zero_()

    # Stage 0: Stand (0 - 0.45s)
    stage_0_mask = (current_time < 0.45).float()
    env.stage_buf[:, 0] = stage_0_mask

    # Stage 1: Down (0.45 - 0.75s)
    stage_1_mask = ((current_time >= 0.45) & (current_time < 0.75)).float()
    env.stage_buf[:, 1] = stage_1_mask

    # Stage 2: Jump (0.75 - 1.05s)
    stage_2_mask = ((current_time >= 0.75) & (current_time < 1.05)).float()
    env.stage_buf[:, 2] = stage_2_mask

    # Stage 3: Turn (1.05 - 1.8s)
    stage_3_mask = ((current_time >= 1.05) & (current_time < 1.8)).float()
    env.stage_buf[:, 3] = stage_3_mask

    # Stage 4: Land (1.8 - 3.0s)
    stage_4_mask = (current_time >= 1.8).float()
    env.stage_buf[:, 4] = stage_4_mask


def stage0_joint_deviation_penalty(
    env: "BaseEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Stage 0阶段关节偏离惩罚：防止机器人在站立准备阶段乱动
    只在Stage 0时惩罚关节角度偏离初始值
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取当前关节位置与默认位置的偏差
    joint_pos = asset.data.joint_pos
    default_joint_pos = asset.data.default_joint_pos
    deviation = torch.sum(torch.square(joint_pos - default_joint_pos), dim=-1)

    # 只在Stage 0时惩罚
    in_stage_0 = env.stage_buf[:, 0]

    return deviation * in_stage_0


# ==================== Stage-aware 奖励函数 ====================

def backflip_com_height_reward(
    env: "BaseEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    COM高度奖励（分阶段）
    Stage 0: 目标0.28m（站立）
    Stage 1: 目标0.18m（下蹲）
    Stage 2/3: 越高越好，目标0.45m以上（起跳/翻转）
    Stage 4: 目标0.28m（着陆）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    com_height = asset.data.root_pos_w[:, 2]

    reward = torch.zeros(env.num_envs, device=env.device)
    reward += env.stage_buf[:, 0] * (-torch.abs(com_height - 0.28))
    reward += env.stage_buf[:, 1] * (-torch.abs(com_height - 0.18))
    reward += env.stage_buf[:, 2] * torch.clamp(com_height, max=0.60)  # 提高上限到60cm
    reward += env.stage_buf[:, 3] * torch.clamp(com_height, max=0.60)
    reward += env.stage_buf[:, 4] * (-torch.abs(com_height - 0.28))

    return reward


def backflip_body_orientation_reward(
    env: "BaseEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    身体姿态奖励（分阶段）- RO2坐标系修正
    Stage 0/1/4: body_z垂直向上（body_z[2]=1），保持直立
    Stage 2/3: 不约束body_z方向，让orientation_control_l2处理动态旋转

    注：RO2前方是+Y，后空翻绕X轴旋转，body_z会在YZ平面内旋转
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_z = math_utils.quat_rotate_inverse(
        asset.data.root_quat_w,
        torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    )

    # Stage 0/1/4: 直立（body_z[2] ~ 1）
    upright_reward = -torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0))

    reward = torch.zeros(env.num_envs, device=env.device)
    reward += env.stage_buf[:, 0] * upright_reward  # Stand: 直立
    reward += env.stage_buf[:, 1] * upright_reward  # Crouch: 直立
    # Stage 2/3: 不约束，让orientation_control_l2处理旋转
    reward += env.stage_buf[:, 4] * upright_reward  # Land: 恢复直立

    return reward


def backflip_velocity_reward(
    env: "BaseEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    速度奖励（分阶段）- RO2坐标系修正
    Stage 0/1/4: 最小化所有速度（保持静止）
    Stage 2/3: 鼓励负X轴角速度（后空翻旋转），完成一圈后停止奖励

    RO2前方是+Y，后空翻绕X轴旋转（而不是标准的Y轴）
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 在body frame下的速度
    lin_vel_b = asset.data.root_lin_vel_b
    ang_vel_b = asset.data.root_ang_vel_b

    # 静止惩罚（x, y线速度 + yaw角速度）
    vel_penalty = (
        torch.square(lin_vel_b[:, 0]) +
        torch.square(lin_vel_b[:, 1]) +
        torch.square(ang_vel_b[:, 2])
    )

    # X轴角速度奖励（RO2坐标系修正：后空翻绕X轴）
    # 负值表示后仰旋转，仅在未完成一圈翻转时奖励
    roll_reward = -ang_vel_b[:, 0] * (1.0 - env.is_one_turn_buf.float())

    reward = torch.zeros(env.num_envs, device=env.device)
    reward += env.stage_buf[:, 0] * (-vel_penalty)
    reward += env.stage_buf[:, 1] * (-vel_penalty)
    reward += env.stage_buf[:, 2] * roll_reward  # 改用X轴角速度
    reward += env.stage_buf[:, 3] * roll_reward  # 改用X轴角速度
    reward += env.stage_buf[:, 4] * (-vel_penalty)

    return reward


def backflip_foot_contact_reward(
    env: "BaseEnv",
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    足端接触奖励（分阶段）
    Stage 0/1/4: 四足都应接触地面
    Stage 2: 后足接触（前足可腾空）
    Stage 3: 所有足腾空
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, :]

    # 检测接触（阈值10N）
    is_contact = (torch.norm(net_contact_forces, dim=-1) > 10.0).float()
    contact_ratio = is_contact.mean(dim=1)

    # 默认threshold（用于stage 0/1/4）
    foot_contact_threshold = 0.25

    reward = torch.zeros(env.num_envs, device=env.device)
    reward += env.stage_buf[:, 0] * foot_contact_threshold  # Stand: 期望接触
    reward += env.stage_buf[:, 1] * foot_contact_threshold  # Down: 期望接触
    # Stage 2: 期望后足接触
    # RO2后足索引：lb3(0), rb3(2)
    if is_contact.shape[1] >= 4:
        back_feet_contact = (is_contact[:, 0] + is_contact[:, 2]) / 2.0
        reward += env.stage_buf[:, 2] * (1.0 - back_feet_contact)
    else:
        reward += env.stage_buf[:, 2] * contact_ratio
    reward += env.stage_buf[:, 3] * foot_contact_threshold  # Turn: 期望腾空（反向）
    reward += env.stage_buf[:, 4] * foot_contact_threshold  # Land: 期望接触

    return reward


# ==================== 约束项（Cost函数）====================

def backflip_energy_cost(env: "BaseEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """能量消耗惩罚（所有阶段）"""
    asset: Articulation = env.scene[asset_cfg.name]
    # 使用力矩平方和作为能量指标
    torques = asset.data.applied_torque
    return torch.sum(torch.square(torques), dim=-1)


def backflip_style_cost(env: "BaseEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """姿态风格惩罚：关节位置偏离默认值（所有阶段）"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    default_joint_pos = asset.data.default_joint_pos
    deviation = torch.square(joint_pos - default_joint_pos)
    return torch.mean(deviation, dim=-1)


def backflip_body_contact_cost(
    env: "BaseEnv",
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    不期望的身体接触惩罚（分阶段）
    Stage 0/1/2: 躯干和大腿接触都会终止
    Stage 3/4: 仅大腿接触惩罚（允许翻转时躯干短暂接触）
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, :]

    # 检测任意body part接触（降低阈值到0.2N，更敏感地检测膝盖轻触）
    has_undesired_contact = torch.any(torch.norm(net_contact_forces, dim=-1) > 0.2, dim=-1).float()

    cost = torch.zeros(env.num_envs, device=env.device)
    cost += env.stage_buf[:, 0] * has_undesired_contact
    cost += env.stage_buf[:, 1] * has_undesired_contact
    cost += env.stage_buf[:, 2] * has_undesired_contact
    cost += env.stage_buf[:, 3] * has_undesired_contact * 0.5  # 翻转时放宽
    cost += env.stage_buf[:, 4] * has_undesired_contact * 0.5  # 着陆时放宽

    return cost


# ==================== 综合设计：混合Stage + 时间窗口 + 动态姿态 ====================

def phase_encoding(env: "BaseEnv") -> torch.Tensor:
    """
    Phase编码：sin/cos多频编码（6维）
    替代Stage one-hot，提供更平滑的时间信息
    参考go2backflip实现
    """
    current_time = env.episode_length_buf * env.step_dt
    max_time = 3.0  # 匹配新的Episode长度

    phase = torch.pi * current_time / max_time

    encoding = torch.stack([
        torch.sin(phase),
        torch.cos(phase),
        torch.sin(phase / 2.0),
        torch.cos(phase / 2.0),
        torch.sin(phase / 4.0),
        torch.cos(phase / 4.0),
    ], dim=-1)

    return encoding


def orientation_control_l2(env: "BaseEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    动态姿态追踪奖励：结合Stage + 连续时间（RO2坐标系修正）
    - Stage 0/1: 保持直立
    - Stage 2/3: 动态旋转目标（0.5秒内旋转2圈，4π）
    - Stage 4: 恢复直立

    RO2前方是+Y，后空翻绕X轴旋转，所以计算roll角（而不是pitch）
    参考go2backflip的orientation_control_l2，但改用roll公式
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt

    # 计算动态旋转目标（绕X轴）
    # Stage 2通常在~0.5s开始，持续到~1.0s
    phase = torch.clamp((current_time - 0.5) / 0.5, 0.0, 1.0)  # 0.5-1.0s映射到0-1
    flip_target_roll = 4.0 * torch.pi * phase  # 4π = 2圈

    # Stage-wise目标
    target_roll = torch.zeros(env.num_envs, device=env.device)
    target_roll += 0.0 * env.stage_buf[:, 0]  # Stage 0: 直立
    target_roll += 0.0 * env.stage_buf[:, 1]  # Stage 1: 直立
    target_roll += flip_target_roll * env.stage_buf[:, 2]  # Stage 2: 动态旋转
    target_roll += flip_target_roll * env.stage_buf[:, 3]  # Stage 3: 继续旋转
    target_roll += 0.0 * env.stage_buf[:, 4]  # Stage 4: 恢复直立

    # 从四元数计算当前roll（绕X轴旋转）- RO2坐标系修正
    body_quat = asset.data.root_quat_w
    # roll = arctan2(2(qw*qx + qy*qz), 1 - 2(qx^2 + qy^2))
    qw, qx, qy, qz = body_quat[:, 0], body_quat[:, 1], body_quat[:, 2], body_quat[:, 3]
    current_roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx**2 + qy**2))

    # 计算误差
    roll_error = torch.abs(current_roll - target_roll)

    return roll_error


def base_ang_vel_y(env: "BaseEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    X轴角速度奖励：后空翻旋转的核心驱动（RO2坐标系修正）
    RO2前方是+Y，所以后空翻绕X轴旋转（而不是标准的Y轴）
    只在Stage 2/3给奖励，移除时间窗口和最小旋转阈值
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 身体坐标系的角速度 - RO2坐标系修正：使用X轴角速度
    ang_vel_b = asset.data.root_ang_vel_b
    ang_vel_x = ang_vel_b[:, 0]  # X轴角速度（正值=后空翻旋转）
    ang_vel_x = torch.clamp(ang_vel_x, -7.2, 7.2)

    # 侧翻软惩罚：Y轴旋转越大，奖励衰减越多（指数衰减）
    purity = torch.exp(-torch.abs(ang_vel_b[:, 1]) / 2.0)

    # Stage窗口：Stage 2或3（翻转阶段）
    in_flip_stage = env.stage_buf[:, 2] + env.stage_buf[:, 3]

    # 奖励 = 旋转速度 × 纯度系数 × Stage掩码
    reward = ang_vel_x * purity * in_flip_stage

    return reward


def base_lin_vel_z(env: "BaseEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    z轴线速度奖励：起跳的核心驱动
    在Stage 1和Stage 2给奖励（让机器人在Stage 1就开始准备起跳）
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 世界坐标系的线速度
    lin_vel_w = asset.data.root_lin_vel_w
    lin_vel_z = lin_vel_w[:, 2]
    lin_vel_z = torch.clamp(lin_vel_z, max=3.0)

    # Stage窗口：Stage 1和2（下蹲+起跳）
    in_jump_stage = env.stage_buf[:, 1] + env.stage_buf[:, 2]

    reward = lin_vel_z * in_jump_stage

    return reward


def height_control_l2(env: "BaseEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    高度控制惩罚：非翻转时期保持目标高度
    在翻转外的时间（<0.6s或>2.1s）惩罚高度偏差
    参考go2backflip，时间窗口从2s缩放到3s
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt

    target_height = 0.25  # 放宽目标高度到25cm，更灵活
    current_height = asset.data.root_pos_w[:, 2]
    height_diff = torch.square(target_height - current_height)

    # 只在非翻转时期惩罚（修复：从2s缩放到3s）
    non_flip_phase = (current_time < 0.6) + (current_time > 2.1)

    return height_diff * non_flip_phase.float()


def feet_height_before_backflip_l1(
    env: "BaseEnv",
    height_threshold: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    翻转前足部离地惩罚（时间窗口版）
    在t<0.75s时，足部不能离地
    参考go2backflip，时间窗口从2s缩放到3s
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt

    # 获取足部高度（假设foot bodies在body_ids中）
    # 这里简化：用COM高度代替（实际应该用足部位置）
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] if hasattr(asset_cfg, 'body_ids') and len(asset_cfg.body_ids) > 0 else asset.data.root_pos_w[:, 2:3]

    if foot_height.dim() == 2 and foot_height.shape[1] > 1:
        foot_height_penalty = (foot_height - height_threshold).clamp(min=0.0).sum(dim=1)
    else:
        foot_height_penalty = (foot_height.squeeze() - height_threshold).clamp(min=0.0)

    # 只在翻转前惩罚（修复：从0.5s缩放到0.75s）
    before_backflip = (current_time < 0.75).float()

    return foot_height_penalty * before_backflip


def feet_height_after_backflip_l1(
    env: "BaseEnv",
    height_threshold: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    翻转后足部离地惩罚（时间窗口版）
    在t>2.25s时，足部不能离地
    参考go2backflip，时间窗口从2s缩放到3s
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt

    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] if hasattr(asset_cfg, 'body_ids') and len(asset_cfg.body_ids) > 0 else asset.data.root_pos_w[:, 2:3]

    if foot_height.dim() == 2 and foot_height.shape[1] > 1:
        foot_height_penalty = (foot_height - height_threshold).clamp(min=0.0).sum(dim=1)
    else:
        foot_height_penalty = (foot_height.squeeze() - height_threshold).clamp(min=0.0)

    # 只在翻转后惩罚（修复：从1.5s缩放到2.25s）
    after_backflip = (current_time > 2.25).float()

    return foot_height_penalty * after_backflip


def actions_symmetry_l2(env: "BaseEnv") -> torch.Tensor:
    """
    动作对称性惩罚：左右腿pitch关节应该对称（RO2专用）

    RO2关节顺序（按字母排序）：
    索引：  0,      1,        2,       3,      4,        5,
            6,      7,        8,       9,      10,       11
    关节：  lb_yaw, lb_thigh, lb_calf, lf_yaw, lf_thigh, lf_calf,
            rb_yaw, rb_thigh, rb_calf, rf_yaw, rf_thigh, rf_calf

    只惩罚pitch关节（thigh/calf），不惩罚yaw关节
    Pitch对称性：action_L + action_R ≈ 0
    """
    actions = env.action_buffer._circular_buffer.buffer[:, -1, :]

    # 死区阈值，允许轻微晃动
    threshold = 0.1

    # === 前腿pitch对称 LF(4,5) vs RF(10,11) ===
    lf_rf_pitch_sum = actions[:, 4:6] + actions[:, 10:12]
    lf_rf_error = torch.relu(torch.abs(lf_rf_pitch_sum) - threshold)
    sym_error = (lf_rf_error ** 2).sum(dim=-1)

    # === 后腿pitch对称 LB(1,2) vs RB(7,8) ===
    lb_rb_pitch_sum = actions[:, 1:3] + actions[:, 7:9]
    lb_rb_error = torch.relu(torch.abs(lb_rb_pitch_sum) - threshold)
    sym_error += (lb_rb_error ** 2).sum(dim=-1)

    return sym_error


def base_ang_vel_x_l1(env: "BaseEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """y轴角速度惩罚（侧向翻滚，不期望）- RO2坐标系修正
    RO2前方是+Y，后空翻绕X轴旋转（期望），侧向翻滚绕Y轴（不期望）
    添加死区：只惩罚超过1.0 rad/s的旋转
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel = torch.abs(asset.data.root_ang_vel_b[:, 1])  # Y轴，RO2的侧向翻滚
    return torch.relu(ang_vel - 1.0)  # 死区1.0 rad/s


def base_ang_vel_z_l1(env: "BaseEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """z轴角速度惩罚（偏航，不期望）
    添加死区：只惩罚超过1.0 rad/s的旋转
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel = torch.abs(asset.data.root_ang_vel_b[:, 2])
    return torch.relu(ang_vel - 1.0)  # 死区1.0 rad/s


def gravity_y_l2(env: "BaseEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """侧向重力分量惩罚（保持左右平衡）- RO2坐标系修正
    RO2的侧向是X轴（Y轴是前方），所以惩罚gravity_b[:, 0]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b
    return torch.square(projected_gravity[:, 0])  # X轴是RO2的侧向


def early_knee_height_reward(
    env: "BaseEnv",
    height_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    早期膝盖高度奖励：在前0.5s内，奖励膝盖离地
    防止机器人在初期就"躺平"采用膝盖着地姿态
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt

    # 获取膝盖（大腿body）的高度
    # body_ids应该指向大腿bodies（.*2_.*）
    if hasattr(asset_cfg, 'body_ids') and len(asset_cfg.body_ids) > 0:
        knee_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
        # 计算膝盖离地的奖励（高于阈值=好）
        if knee_height.dim() == 2 and knee_height.shape[1] > 1:
            # 多个膝盖，取平均
            knee_above_threshold = (knee_height - height_threshold).clamp(min=0.0).mean(dim=1)
        else:
            knee_above_threshold = (knee_height.squeeze() - height_threshold).clamp(min=0.0)
    else:
        # 如果没有body_ids，用COM高度作为近似
        knee_above_threshold = (asset.data.root_pos_w[:, 2] - height_threshold).clamp(min=0.0)

    # 只在早期（前0.5s）给奖励
    early_phase = (current_time < 0.5).float()

    return knee_above_threshold * early_phase


def feet_distance_y_exp(
    env: "BaseEnv",
    stance_width: float = 0.25,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    足端站距稳定性奖励（参考go2backflip）
    鼓励机器人保持合理的左右站距，避免站姿不稳

    Args:
        stance_width: 期望的左右站距（m），RO2约0.25m
        std: 高斯函数标准差
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取足端在世界坐标系的位置，转换到body frame
    cur_footsteps_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)

    # 转换到body frame
    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = quat_apply(
            quat_conjugate(asset.data.root_quat_w), cur_footsteps_translated[:, i, :]
        )

    # 期望的Y坐标（body frame）
    # RO2足端顺序（字母排序）：[lb3, lf3, rb3, rf3]
    # 左侧 Y > 0，右侧 Y < 0
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = torch.cat(
        [stance_width_tensor / 2, stance_width_tensor / 2,
         -stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # 计算偏差并用高斯函数转换为奖励
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std ** 2))

    # 乘以直立度因子（只有站直时才给奖励）
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

    return reward


def hip_joint_position_penalty(
    env: "BaseEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    髋关节位置惩罚：防止髋关节（yaw）偏离默认角度太多
    保持躯干与大腿连接处的稳定性，避免躯干晃动

    RO2髋关节索引（字母排序）：
    - 0: lb1_joint_yaw (左后髋)
    - 3: lf1_joint_yaw (左前髋)
    - 6: rb1_joint_yaw (右后髋)
    - 9: rf1_joint_yaw (右前髋)
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取所有关节位置
    joint_pos = asset.data.joint_pos
    default_joint_pos = asset.data.default_joint_pos

    # RO2髋关节索引：lb1(0), lf1(3), rb1(6), rf1(9)
    hip_indices = [0, 3, 6, 9]

    # 计算髋关节偏离默认位置的平方误差
    hip_deviation = torch.square(joint_pos[:, hip_indices] - default_joint_pos[:, hip_indices])

    # 返回总偏差（所有4个髋关节的平方和）
    return torch.sum(hip_deviation, dim=1)


def airtime_reward(
    env: "BaseEnv",
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    腾空奖励：只在跳跃/腾空阶段（Stage 2/3）奖励机器人离开地面
    当所有足端都离地时给予正向奖励，奖励值与当前高度成正比
    """
    asset: Articulation = env.scene["robot"]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 检测足端接触
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, :]
    is_contact = (torch.norm(net_contact_forces, dim=-1) > 10.0).float()

    # 腾空 = 所有足都离地
    airborne = (is_contact.mean(dim=1) < 0.1).float()

    # 只在Stage 2/3（跳跃/腾空阶段）给奖励
    in_jump_stage = env.stage_buf[:, 2] + env.stage_buf[:, 3]

    # 奖励 = 腾空状态 * 当前高度 * 阶段掩码
    com_height = asset.data.root_pos_w[:, 2]
    reward = airborne * com_height * in_jump_stage

    return reward


def stage0_body_balance_reward(
    env: "BaseEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Stage 0身体平衡奖励：站立准备阶段保持直立
    只在Stage 0时给予正向奖励，鼓励机器人保持初始直立姿态
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 计算body_z方向（世界坐标系的Z轴在body frame的投影）
    body_z = math_utils.quat_rotate_inverse(
        asset.data.root_quat_w,
        torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    )

    # 直立度奖励：body_z[2]越接近1越好（范围0-1）
    upright_reward = torch.clamp(body_z[:, 2], 0.0, 1.0)

    # 只在Stage 0时给奖励
    in_stage_0 = env.stage_buf[:, 0]

    return upright_reward * in_stage_0


def stage4_body_balance_reward(
    env: "BaseEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Stage 4身体平衡奖励：着陆阶段强力奖励恢复直立
    只在Stage 4时给予大正向奖励，确保机器人成功着陆并稳定
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 计算body_z方向
    body_z = math_utils.quat_rotate_inverse(
        asset.data.root_quat_w,
        torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    )

    # 直立度奖励（指数形式，更敏感）
    # 当body_z[2]=1时，奖励=1；当body_z[2]<0.7时，奖励快速衰减
    upright_reward = torch.exp(5.0 * (body_z[:, 2] - 1.0))  # 指数衰减，惩罚偏离

    # 只在Stage 4时给奖励
    in_stage_4 = env.stage_buf[:, 4]

    return upright_reward * in_stage_4


def airtime_height_reward(
    env: "BaseEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    腾空高度奖励：只在Stage 2/3时奖励更高的跳跃
    高度越高，奖励越大，鼓励机器人跳得更高以完成后空翻
    无高度上限，鼓励极限跳跃
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取COM高度（无上限）
    com_height = asset.data.root_pos_w[:, 2]

    # 只在Stage 2/3（跳跃/腾空阶段）给奖励
    in_flip_stage = env.stage_buf[:, 2] + env.stage_buf[:, 3]

    # 奖励 = 高度 * 阶段掩码（高度越高奖励越大，无上限）
    return com_height * in_flip_stage


def stage2_3_inaction_penalty(
    env: "BaseEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Stage 2/3不行动惩罚：强制机器人在翻转阶段必须行动
    如果在Stage 2/3时旋转速度太低，给予惩罚
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取X轴角速度（后空翻方向）
    ang_vel_x = torch.abs(asset.data.root_ang_vel_b[:, 0])

    # 如果旋转速度 < 0.5 rad/s，惩罚
    inaction = torch.relu(0.5 - ang_vel_x)

    # 只在Stage 2/3时惩罚
    in_flip_stage = env.stage_buf[:, 2] + env.stage_buf[:, 3]

    return inaction * in_flip_stage
