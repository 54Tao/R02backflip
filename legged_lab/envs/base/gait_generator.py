# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# Gait phase generator based on walk-these-ways implementation

import torch
import numpy as np


class GaitGenerator:
    """生成步态相位，用于四足机器人的步态控制

    基于walk-these-ways (Improbable-AI)的实现
    使用正弦函数生成各足端的相位，并计算期望接触状态
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        device: str,
        frequency: float = 3.0,  # 步频 Hz
        phase_offset: tuple = (0.0, 0.5, 0.5, 0.0),  # 四足相位偏移 (FR, FL, RR, RL)
        duty_cycle: float = 0.6,  # 支撑相占比
        kappa: float = 0.15,  # 平滑参数（高斯分布标准差）
    ):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device

        # 步态参数
        self.frequency = frequency
        self.phase_offset = torch.tensor(phase_offset, device=device, dtype=torch.float32)
        self.duty_cycle = duty_cycle
        self.kappa = kappa

        # 步态索引（随时间递增的相位，0-1循环）
        self.gait_indices = torch.zeros(num_envs, device=device, dtype=torch.float32)

        # 各足端相位（0-1）
        self.foot_indices = torch.zeros(num_envs, 4, device=device, dtype=torch.float32)

        # 期望接触状态（0-1，1表示应该接触地面）
        self.desired_contact_states = torch.zeros(num_envs, 4, device=device, dtype=torch.float32)

        # 正弦时钟输入（用于观测，可选）
        self.clock_inputs = torch.zeros(num_envs, 4, device=device, dtype=torch.float32)

    def reset(self, env_ids):
        """重置指定环境的步态相位"""
        self.gait_indices[env_ids] = torch.rand(len(env_ids), device=self.device)

    def step(self):
        """更新步态相位（每个控制步调用一次）"""
        # 更新主相位（0-1循环）
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * self.frequency, 1.0)

        # 计算各足端相位（加上偏移）
        # 顺序：FR, FL, RR, RL (前右、前左、后右、后左)
        for i in range(4):
            self.foot_indices[:, i] = torch.remainder(
                self.gait_indices + self.phase_offset[i], 1.0
            )

        # 计算期望接触状态（使用平滑过渡）
        self._compute_desired_contact_states()

        # 计算正弦时钟输入
        self.clock_inputs = torch.sin(2 * np.pi * self.foot_indices)

    def _compute_desired_contact_states(self):
        """计算期望接触状态（支撑相=1，摆动相=0，带平滑过渡）"""
        # 使用高斯CDF进行平滑
        smoothing_cdf = torch.distributions.normal.Normal(0, self.kappa).cdf

        for i in range(4):
            phase = self.foot_indices[:, i]

            # 计算平滑系数（在相位0.0和0.5附近平滑过渡）
            # 这创建了一个从1到0再到1的平滑曲线
            smoothing_multiplier = (
                smoothing_cdf(torch.remainder(phase, 1.0)) *
                (1 - smoothing_cdf(torch.remainder(phase, 1.0) - 0.5)) +
                smoothing_cdf(torch.remainder(phase, 1.0) - 1) *
                (1 - smoothing_cdf(torch.remainder(phase, 1.0) - 0.5 - 1))
            )

            self.desired_contact_states[:, i] = smoothing_multiplier

    def get_swing_phase(self, foot_idx: int = None):
        """获取摆动相位（0=刚离地，1=即将着地）

        Args:
            foot_idx: 足端索引(0-3)。如果为None，返回所有足端

        Returns:
            摆动相位，范围0-1。支撑相时返回0
        """
        if foot_idx is not None:
            phase = self.foot_indices[:, foot_idx]
            is_swing = phase > self.duty_cycle
            swing_phase = torch.where(
                is_swing,
                (phase - self.duty_cycle) / (1.0 - self.duty_cycle),
                torch.zeros_like(phase)
            )
            return swing_phase
        else:
            # 返回所有足端
            phases = self.foot_indices
            is_swing = phases > self.duty_cycle
            swing_phases = torch.where(
                is_swing,
                (phases - self.duty_cycle) / (1.0 - self.duty_cycle),
                torch.zeros_like(phases)
            )
            return swing_phases
