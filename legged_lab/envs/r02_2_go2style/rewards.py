"""
RO2后空翻奖励函数 - 完全复现go2backflip
坐标系适配：RO2的Y轴在前，后空翻绕X轴旋转

Go2坐标系: X-前, Y-左, Z-上 -> 后空翻绕Y轴
RO2坐标系: Y-前, X-左, Z-上 -> 后空翻绕X轴

坐标轴映射:
- go2的ang_vel_y (后空翻) -> RO2的ang_vel_x
- go2的ang_vel_x (侧翻) -> RO2的ang_vel_y
- go2的gravity_y (侧向) -> RO2的gravity_x
"""

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul, quat_from_angle_axis, quat_apply, quat_conjugate
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv


def base_ang_vel_x(env: "BaseEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    X轴角速度奖励：后空翻旋转的核心驱动
    RO2的Y轴在前，后空翻绕X轴旋转（对应go2的ang_vel_y）
    时间窗口：0.5-1.0s
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt

    # 正X轴角速度 = 后仰旋转（后空翻方向）
    # 右手定则：大拇指指向+X（左），四指从Y（前）卷向Z（上）= 后空翻
    ang_vel = asset.data.root_ang_vel_b[:, 0]
    ang_vel = torch.clamp(ang_vel, min=-7.2, max=7.2)

    # 只在翻转阶段给奖励
    in_flip_window = torch.logical_and(current_time > 0.5, current_time < 1.0)
    return ang_vel * in_flip_window


def base_ang_vel_y_l1(env: "BaseEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Y轴角速度惩罚：防止侧翻
    RO2的侧翻绕Y轴（对应go2的ang_vel_x）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_ang_vel_b[:, 1])


def base_ang_vel_z_l1(env: "BaseEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Z轴角速度惩罚：防止偏航"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_ang_vel_b[:, 2])


def base_lin_vel_z(env: "BaseEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Z轴线速度奖励：起跳的核心驱动
    时间窗口：0.5-0.75s
    降低上限避免跳太高（RO2更轻，2.0m/s→20cm高度）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt

    # 上限2.0 m/s → 跳高约20cm（合理）
    lin_vel = torch.clamp(asset.data.root_lin_vel_w[:, 2], max=2.5)

    # 只在起跳阶段给奖励
    in_takeoff_window = torch.logical_and(current_time > 0.5, current_time < 0.75)
    return lin_vel * in_takeoff_window


def orientation_control_l2(env: "BaseEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    动态姿态追踪惩罚
    RO2绕X轴旋转（roll），对应go2绕Y轴旋转（pitch）
    目标：0.5秒内旋转4π（2圈）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt

    # 计算动态roll目标
    phase = (current_time - 0.5).clamp(min=0, max=0.5)
    target_roll_angle = 4 * phase * torch.pi

    # 创建目标四元数（绕X轴旋转）
    target_roll_quat = quat_from_angle_axis(
        target_roll_angle,
        torch.tensor([1.0, 0.0, 0.0], device=env.device)  # X轴，RO2的后空翻轴
    )

    # 与初始姿态结合
    base_init_quat = asset.data.default_root_state[:, 3:7]
    desired_base_quat = quat_mul(target_roll_quat, base_init_quat)

    # 计算期望投影重力
    inv_desired_base_quat = quat_conjugate(desired_base_quat)
    global_gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, -1)
    desired_projected_gravity = quat_apply(inv_desired_base_quat, global_gravity)

    # 计算姿态误差
    orientation_diff = torch.sum(torch.square(asset.data.projected_gravity_b - desired_projected_gravity), dim=1)
    return orientation_diff


def height_control_l2(env: "BaseEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    高度控制惩罚：非翻转阶段保持目标高度
    目标高度：0.28m（RO2比Go2矮，Go2是0.3m）
    时间窗口外：<0.4s 或 >1.4s
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt

    target_height = 0.3  # RO2的站立高度约28cm
    height_diff = torch.square(target_height - asset.data.root_pos_w[:, 2])

    # 只在非翻转阶段惩罚
    non_flip_phase = torch.logical_or(current_time < 0.4, current_time > 1.4)
    return height_diff * non_flip_phase


def actions_symmetry_l2(env: "BaseEnv") -> torch.Tensor:
    """
    动作对称性惩罚（仅pitch轴）

    RO2关节顺序（字母排序）：
    0-2: lb (左后) yaw, thigh, calf
    3-5: lf (左前) yaw, thigh, calf
    6-8: rb (右后) yaw, thigh, calf
    9-11: rf (右前) yaw, thigh, calf

    对称性要求（RO2左右pitch符号相反）：
    - thigh/calf关节: L + R = 0
    """
    actions = env.action_buffer._circular_buffer.buffer[:, -1, :]

    # 前腿pitch对称: LF vs RF (thigh/calf)
    actions_diff = torch.square(actions[:, 4:6] + actions[:, 10:12]).sum(dim=-1)

    # 后腿pitch对称: LB vs RB (thigh/calf)
    actions_diff += torch.square(actions[:, 1:3] + actions[:, 7:9]).sum(dim=-1)

    return actions_diff


def gravity_x_l2(env: "BaseEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    侧向重力分量惩罚
    RO2的侧向是X轴（对应go2的gravity_y）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 0])


def feet_distance_y_exp(
    env: "BaseEnv",
    stance_width: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    足端站距稳定性奖励
    RO2足端顺序（字母排序）：[lb3, lf3, rb3, rf3]
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取足端位置，转换到body frame
    cur_footsteps_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = quat_apply(
            quat_conjugate(asset.data.root_quat_w), cur_footsteps_translated[:, i, :]
        )

    # 期望Y坐标（RO2足端顺序：lb, lf, rb, rf）
    # 左侧 Y > 0，右侧 Y < 0
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = torch.cat(
        [stance_width_tensor / 2, stance_width_tensor / 2,
         -stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)

    # 乘以直立度因子
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

    return reward


def feet_height_before_backflip_l1(
    env: "BaseEnv",
    height_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    翻转前足部离地惩罚
    时间窗口：<0.5s
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt

    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - height_threshold
    foot_height_penalty = foot_height.clamp(min=0).sum(dim=1)

    # 只在翻转开始前惩罚
    before_backflip = current_time < 0.5

    return foot_height_penalty * before_backflip


def undesired_contacts_foot(
    env: "BaseEnv",
    threshold: float,
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """腿部不期望接触惩罚"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def undesired_contacts_base(
    env: "BaseEnv",
    threshold: float,
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """躯干接触惩罚"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def phase_encoding(env: "BaseEnv") -> torch.Tensor:
    """
    Phase编码：sin/cos多频编码（6维）
    与go2backflip完全一致
    """
    current_time = env.episode_length_buf * env.step_dt
    max_time = 2.0  # Episode长度

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


def knee_contact_penalty(
    env: "BaseEnv",
    threshold: float,
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    膝盖触地惩罚（全程）
    惩罚大腿（.*2_.*）接触地面
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def knee_height_reward(
    env: "BaseEnv",
    min_height: float,
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    膝盖高度惩罚（全程）
    惩罚膝盖（大腿link）低于min_height
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取所有膝盖（大腿）的高度
    knee_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (num_envs, 4)

    # 计算所有膝盖低于min_height的数量（全程惩罚）
    knees_below_threshold = (knee_heights < min_height).float().sum(dim=1)  # (num_envs,)

    return knees_below_threshold


def hip_joint_angle_penalty(
    env: "BaseEnv",
    threshold: float,
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    髋关节角度惩罚（站立+着陆阶段）
    惩罚yaw关节偏离默认值（0.0 rad）
    时间窗口：<0.5s（准备） 或 >1.4s（着陆）

    防止机器人在起跳/落地时过度使用髋关节导致用膝盖接触地面
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt

    # 获取所有yaw关节角度相对于默认值的偏差
    # RO2关节顺序（字母排序）：lb(0-2), lf(3-5), rb(6-8), rf(9-11)
    # yaw关节索引：0(lb_yaw), 3(lf_yaw), 6(rb_yaw), 9(rf_yaw)
    # 默认yaw都是0.0
    yaw_joint_indices = [0, 3, 6, 9]
    hip_joint_deviation = asset.data.joint_pos[:, yaw_joint_indices]  # (num_envs, 4)

    # 计算超过阈值的关节数量
    exceed_threshold = (torch.abs(hip_joint_deviation) > threshold).float().sum(dim=1)  # (num_envs,)

    # 只在站立和着陆阶段惩罚
    stable_phase = torch.logical_or(current_time < 0.5, current_time > 1.4)

    return exceed_threshold * stable_phase
