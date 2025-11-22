# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def undesired_contacts(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_air_time_quadruped(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """四足机器人腾空时间奖励：奖励每只脚的腾空时间，鼓励交替抬脚

    Args:
        env: 环境
        threshold: 腾空时间上限（秒）
        sensor_cfg: 接触传感器配置，body_ids应该是所有足端

    Returns:
        torch.Tensor: 所有足端腾空时间的平均值（经过阈值裁剪）
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取每只脚的腾空时间
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    # 裁剪到阈值
    air_time = torch.clamp(air_time, max=threshold)
    # 计算平均腾空时间（鼓励所有脚都有抬起的时间）
    reward = torch.mean(air_time, dim=1)
    # 只在有移动命令时给奖励
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_slide(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def body_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def no_fly_penalty(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.05) -> torch.Tensor:
    """惩罚拖脚行为：如果所有脚都在地面附近（高度很低），说明没有抬腿

    Args:
        env: 环境
        asset_cfg: 机器人asset配置，body_ids应该是所有足端
        threshold: 高度阈值（米），低于此值认为在地面

    Returns:
        torch.Tensor: 若所有脚都贴地返回1.0（严重拖脚），否则返回0.0
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 获取所有足端的高度（世界坐标系Z轴）
    feet_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    # 检查每只脚是否在地面附近
    feet_on_ground = feet_heights < threshold
    # 如果所有脚都贴地，说明在拖脚
    all_feet_on_ground = torch.all(feet_on_ground, dim=1).float()
    return all_feet_on_ground


def base_angular_velocity_penalty(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """IMU稳定性奖励：惩罚躯体角速度过大（抖动、翻滚）

    Args:
        env: 环境
        asset_cfg: 机器人asset配置

    Returns:
        torch.Tensor: 躯体角速度的L2范数（模拟IMU读数）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 使用身体坐标系的角速度（模拟IMU）
    ang_vel = asset.data.root_ang_vel_b
    # 返回角速度的L2范数（越大越不稳定）
    return torch.sum(torch.square(ang_vel), dim=1)


def knee_height_reward(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), min_height: float = 0.1) -> torch.Tensor:
    """奖励膝盖保持离地高度，防止膝盖着地

    Args:
        env: 环境
        asset_cfg: 机器人asset配置，body_ids应该是所有膝盖连杆
        min_height: 最低允许高度（米）

    Returns:
        torch.Tensor: 膝盖高度奖励（越高越好）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 获取膝盖连杆的高度（世界坐标系Z轴）
    knee_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    # 计算低于最低高度的惩罚
    penalty = torch.clamp(min_height - knee_heights, min=0.0)
    # 返回平均惩罚（所有膝盖）
    return torch.mean(penalty, dim=1)


def shank_joint_deviation_penalty(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold_deg: float = 10.0
) -> torch.Tensor:
    """惩罚小腿关节（pitch3）偏离初始角度，防止小腿过度弯曲

    Args:
        env: 环境
        asset_cfg: 机器人asset配置，joint_names应该是所有pitch3关节
        threshold_deg: 允许的偏离角度（度），超过此值会受到较大惩罚

    Returns:
        torch.Tensor: 小腿弯曲惩罚
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 获取小腿关节的当前角度和默认角度
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    # 计算偏离量（弧度）
    deviation = torch.abs(joint_pos - default_pos)

    # 转换阈值为弧度
    threshold_rad = threshold_deg * 3.14159 / 180.0

    # 只惩罚超过阈值的部分（使用平方惩罚，超过越多惩罚越大）
    excess = torch.clamp(deviation - threshold_rad, min=0.0)
    penalty = torch.sum(torch.square(excess), dim=1)

    return penalty


def gait_reward_contact_force(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
    sigma: float = 100.0
) -> torch.Tensor:
    """基于步态相位的接触力奖励（摆动相惩罚接触力，支撑相鼓励接触力）

    基于walk-these-ways实现，使用步态生成器的期望接触状态

    Args:
        env: 环境
        sensor_cfg: 接触传感器配置，body_ids应该是所有足端
        force_threshold: 判定接触的力阈值
        sigma: 高斯奖励的标准差

    Returns:
        torch.Tensor: 步态接触力跟踪奖励
    """
    if not hasattr(env, 'gait_generator'):
        return torch.zeros(env.num_envs, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取足端接触力（只看Z轴）
    foot_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)

    # 获取期望接触状态（0=摆动，1=支撑）
    desired_contact = env.gait_generator.desired_contact_states

    # 奖励：摆动相时足端力应该为0
    reward = 0
    for i in range(4):
        # 惩罚摆动相时有接触力（desired_contact=0时）
        reward += -(1 - desired_contact[:, i]) * (
            1 - torch.exp(-foot_forces[:, i] ** 2 / sigma)
        )

    return reward / 4  # 平均到4只脚


def gait_reward_foot_velocity(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.25
) -> torch.Tensor:
    """基于步态相位的足端速度奖励（支撑相惩罚足端滑动）

    基于walk-these-ways实现

    Args:
        env: 环境
        sensor_cfg: 接触传感器配置
        asset_cfg: 机器人asset配置
        sigma: 高斯奖励的标准差

    Returns:
        torch.Tensor: 支撑相足端速度惩罚
    """
    if not hasattr(env, 'gait_generator'):
        return torch.zeros(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    # 获取足端速度（世界坐标系）
    foot_velocities = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    # 获取期望接触状态
    desired_contact = env.gait_generator.desired_contact_states

    # 惩罚支撑相时足端有速度（滑动）
    reward = 0
    for i in range(4):
        reward += -(desired_contact[:, i] * (
            1 - torch.exp(-foot_velocities[:, i] ** 2 / sigma)
        ))

    return reward / 4


def gait_reward_foot_clearance(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.1,
    sigma: float = 0.02
) -> torch.Tensor:
    """摆动相足端离地高度奖励

    基于walk-these-ways实现，鼓励摆动相时足端抬到目标高度

    Args:
        env: 环境
        asset_cfg: 机器人asset配置，body_ids应该是所有足端
        target_height: 目标离地高度（米）
        sigma: 高斯奖励的标准差

    Returns:
        torch.Tensor: 足端离地高度奖励
    """
    if not hasattr(env, 'gait_generator'):
        return torch.zeros(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    # 获取足端高度（世界坐标系Z轴）
    foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    # 获取期望接触状态（1-desired = 摆动相）
    desired_contact = env.gait_generator.desired_contact_states

    # 只在摆动相时奖励高度
    # phases: 0-1，摆动相时接近0
    phases = 1 - torch.abs(
        1.0 - torch.clip((env.gait_generator.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0
    )
    target_heights = target_height * phases + 0.02  # 偏移2cm（足端半径）

    # 高度误差
    height_error = torch.square(target_heights - foot_heights)

    # 只在摆动相时计算（desired_contact接近0时）
    rew_clearance = height_error * (1 - desired_contact)

    return torch.sum(rew_clearance, dim=1)
