"""
Go2Style Environment

与 BaseEnv 的区别：
- 观测空间（48维 policy, 52维 critic）
- 移除 command 观测（后空翻不需要）
- 移除 clock_inputs 观测（go2不使用）
- 移除 backflip 专用观测（com_height, stage等）
- 仅保留核心观测 + phase 编码
"""

import torch
from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.r02_2_go2style import rewards as go2_rewards


class Go2StyleEnv(BaseEnv):
    """
    Go2Style Environment 

    观测维度：
    - Policy: 48维
      * base_ang_vel: 3
      * projected_gravity: 3
      * joint_pos (relative): 12
      * joint_vel (relative): 12
      * last_action: 12
      * phase encoding: 6

    - Critic: 52维
      * base_pos_z: 1
      * base_lin_vel: 3
      * policy_obs: 48
    """

    def compute_current_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算当前观测

        Returns:
            actor_obs: (num_envs, 48) - policy observations
            critic_obs: (num_envs, 52) - critic observations
        """
        robot = self.scene["robot"]

        # ========== 基础观测 ==========
        # 1. 角速度 (3D)
        ang_vel = robot.data.root_ang_vel_b

        # 2. 投影重力 (3D)
        projected_gravity = robot.data.projected_gravity_b

        # 3. 关节位置（相对于默认值）(12D)
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos

        # 4. 关节速度（相对于默认值）(12D)
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel

        # 5. 上一步动作 (12D)
        last_action = self.action_buffer._circular_buffer.buffer[:, -1, :]

        # 6. Phase 编码 (6D)
        phase_obs = go2_rewards.phase_encoding(self)

        # ========== Policy 观测 (48D) ==========
        current_actor_obs = torch.cat([
            ang_vel * self.obs_scales.ang_vel,
            projected_gravity * self.obs_scales.projected_gravity,
            joint_pos * self.obs_scales.joint_pos,
            joint_vel * self.obs_scales.joint_vel,
            last_action * self.obs_scales.actions,
            phase_obs,  # 不缩放
        ], dim=-1)

        # ========== Critic 专用观测 ==========
        # 1. 身体高度 (1D)
        base_pos_z = robot.data.root_pos_w[:, 2:3]

        # 2. 身体线速度 (3D)
        base_lin_vel = robot.data.root_lin_vel_b

        # ========== Critic 观测 (52D) ==========
        # base_pos_z(1) + base_lin_vel(3) + actor_obs(48) = 52D
        current_critic_obs = torch.cat([
            base_pos_z,  # 不缩放
            base_lin_vel * self.obs_scales.lin_vel,
            current_actor_obs,
        ], dim=-1)

        return current_actor_obs, current_critic_obs
