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

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from rsl_rl.env import VecEnv

from legged_lab.envs.base.base_env_config import BaseEnvCfg
from legged_lab.envs.base.gait_generator import GaitGenerator
from legged_lab.utils.env_utils.scene import SceneCfg


class BaseEnv(VecEnv):
    def __init__(self, cfg: BaseEnvCfg, headless):
        self.cfg: BaseEnvCfg

        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        # 添加unwrapped属性以兼容gym接口
        self.unwrapped = self

        # ========== 后空翻Stage管理 ==========
        # Stage buffers: one-hot encoding for current stage
        # stage 0: stand, stage 1: down, stage 2: jump, stage 3: air turn, stage 4: land
        self.stage_buf = None  # Will be initialized in reset
        self.is_half_turn_buf = None  # Track if robot completed half turn
        self.is_one_turn_buf = None  # Track if robot completed full turn
        self.start_time_buf = None  # Random start time for diversity

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)

        # === 修复R02_2坐标系：机器人前方是+Y而不是+X ===
        # IsaacLab假设+X是前方，但R02_2的URDF定义+Y是前方
        # 需要旋转命令坐标系90度：(vx, vy) -> (vy, -vx)
        original_resample = self.command_generator._resample_command
        def _resample_command_rotated(env_ids):
            original_resample(env_ids)
            # 旋转线速度命令90度（+X -> +Y）
            vx_old = self.command_generator.vel_command_b[env_ids, 0].clone()
            vy_old = self.command_generator.vel_command_b[env_ids, 1].clone()
            self.command_generator.vel_command_b[env_ids, 0] = -vy_old  # 新的vx = 原来的-vy
            self.command_generator.vel_command_b[env_ids, 1] = vx_old   # 新的vy = 原来的vx
            # angular velocity保持不变（绕Z轴旋转）

        self.command_generator._resample_command = _resample_command_rotated
        print(f"[DEBUG] 已应用R02_2坐标系修正：前方从+X旋转到+Y")

        # === DEBUG: 打印命令值来诊断箭头方向问题 ===
        original_debug_vis = self.command_generator._debug_vis_callback
        debug_counter = [0]
        def debug_vis_with_print(event):
            if debug_counter[0] % 100 == 0:  # 每100帧打印一次
                cmd = self.command_generator.command[0]  # 只看第一个环境
                print(f"[CMD_DEBUG] Frame {debug_counter[0]}: command=(vx={cmd[0]:.2f}, vy={cmd[1]:.2f}, wz={cmd[2]:.2f})")
            debug_counter[0] += 1
            original_debug_vis(event)

        if self.cfg.commands.debug_vis:
            self.command_generator._debug_vis_callback = debug_vis_with_print

        print(f"[DEBUG] 命令配置:")
        print(f"[DEBUG]   heading_command={command_cfg.heading_command}")
        print(f"[DEBUG]   rel_heading_envs={command_cfg.rel_heading_envs}")
        print(f"[DEBUG]   rel_standing_envs={command_cfg.rel_standing_envs}")
        print(f"[DEBUG]   lin_vel_x range: {command_cfg.ranges.lin_vel_x}")
        print(f"[DEBUG]   lin_vel_y range: {command_cfg.ranges.lin_vel_y}")
        print(f"[DEBUG]   ang_vel_z range: {command_cfg.ranges.ang_vel_z}")

        # 检查reset_base的yaw范围配置
        print(f"[DEBUG] 域随机化配置:")
        print(f"[DEBUG]   reset_base yaw range: {self.cfg.domain_rand.events.reset_base.params['pose_range']['yaw']}")

        self.reward_manager = RewardManager(self.cfg.reward, self)

        # 存储初始奖励权重用于课程学习
        self.initial_reward_weights = {}
        for term_name, term_cfg in self.cfg.reward.__dict__.items():
            if hasattr(term_cfg, 'weight'):
                self.initial_reward_weights[term_name] = term_cfg.weight

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

        # 初始化extras["observations"]以支持RND
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {
            "critic": critic_obs,
            "rnd_state": actor_obs.clone(),
        }

    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # 课程学习计数器（用于动态调整奖励权重）
        self.curriculum_iteration = 0

        # ========== 初始化Stage Buffers（用于后空翻等特技动作） ==========
        self.stage_buf = torch.zeros((self.num_envs, 5), dtype=torch.float32, device=self.device)
        self.stage_buf[:, 0] = 1.0  # 初始时所有环境都在stage 0 (stand)
        self.is_half_turn_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.is_one_turn_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.start_time_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # 初始化步态生成器
        self.gait_generator = GaitGenerator(
            num_envs=self.num_envs,
            dt=self.step_dt,
            device=self.device,
            frequency=3.0,  # 步频3Hz
            phase_offset=(0.0, 0.5, 0.5, 0.0),  # trot步态: FR, FL, RR, RL
            duty_cycle=0.6,  # 支撑相占60%
            kappa=0.15,  # 平滑参数
        )
        print(f"[GAIT] 步态生成器已初始化: frequency=3.0Hz, duty_cycle=0.6, trot gait")

        self.init_obs_buffer()

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]

        # 步态时钟输入（参考walk-these-ways）
        # 正弦时钟信号帮助策略理解步态相位
        clock_inputs = self.gait_generator.clock_inputs  # shape: (num_envs, 4)

        # ========== 后空翻专用观测 ==========
        # 添加stage、COM高度、翻转状态等关键信息
        backflip_obs = []
        if hasattr(self.cfg, 'task_type') and self.cfg.task_type == 'backflip':
            # COM高度（归一化到0-0.5m范围）
            com_height = robot.data.root_pos_w[:, 2:3] / 0.5
            # Stage信息（one-hot编码，5维）
            stage_info = self.stage_buf  # (num_envs, 5)
            # 翻转状态（2维）
            flip_status = torch.stack([
                self.is_half_turn_buf.float(),
                self.is_one_turn_buf.float()
            ], dim=-1)
            # 身体直立度（body向上向量的z分量）
            body_quat = robot.data.root_quat_w
            # 创建向上向量[0,0,1]，shape: (num_envs, 3)
            up_dir = torch.zeros(self.num_envs, 3, device=self.device)
            up_dir[:, 2] = 1.0
            # 从四元数计算向上向量
            from isaaclab.utils.math import quat_rotate
            up_vec = quat_rotate(body_quat, up_dir)
            body_upright = up_vec[:, 2:3]  # (num_envs, 1)

            # Phase编码（6维sin/cos，提供平滑时间信息）
            current_time = self.episode_length_buf * self.step_dt
            max_time = 3.0  # Episode长度
            phase = torch.pi * current_time / max_time
            phase_encoding = torch.stack([
                torch.sin(phase),
                torch.cos(phase),
                torch.sin(phase / 2.0),
                torch.cos(phase / 2.0),
                torch.sin(phase / 4.0),
                torch.cos(phase / 4.0),
            ], dim=-1)  # (num_envs, 6)

            backflip_obs = [com_height, stage_info, flip_status, body_upright, phase_encoding]

        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                command * self.obs_scales.commands,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
                clock_inputs,  # 添加步态时钟（4维，每只脚一个）
            ] + backflip_obs,  # 后空翻专用观测（仅在backflip任务时）
            dim=-1,
        )

        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1
        )

        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        self.gait_generator.reset(env_ids)  # 重置步态相位

        # ========== 重置Stage Buffers（用于后空翻任务） ==========
        if hasattr(self.cfg, 'task_type') and self.cfg.task_type == 'backflip':
            # 重置stage为stage 0 (stand)
            self.stage_buf[env_ids] = 0.0
            self.stage_buf[env_ids, 0] = 1.0
            # 重置翻转检测
            self.is_half_turn_buf[env_ids] = 0
            self.is_one_turn_buf[env_ids] = 0
            # 随机启动时间 (0-5秒，增加多样性)
            self.start_time_buf[env_ids] = torch.rand(len(env_ids), device=self.device) * 5.0

        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        self.scene.write_data_to_sim()
        self.sim.forward()

    def update_curriculum(self):
        """更新课程学习：三阶段调整feet_air_time权重

        阶段1 (0-200轮): 权重从0.2逐渐增加到2.0 - 先学站立，再学抬腿
        阶段2 (200-500轮): 保持2.0 - 强化步态学习
        阶段3 (500-1200轮): 从2.0衰减到0.2 - 避免过度优化步态
        """
        if "feet_air_time" not in self.initial_reward_weights:
            return

        # 每个step调用一次，每24步（一个iteration）更新一次
        if self.sim_step_counter % (24 * self.cfg.sim.decimation) == 0:
            self.curriculum_iteration += 1

            # 找到feet_air_time在列表中的索引
            term_idx = self.reward_manager._term_names.index("feet_air_time")

            # 课程学习的目标权重（硬编码，不依赖配置文件）
            min_weight = 0.2   # 最小权重（初期和后期）
            max_weight = 2.0   # 最大权重（中期）

            if self.curriculum_iteration <= 200:
                # 阶段1: 0-200轮，从0.2逐渐增加到2.0
                # 让机器人先学站立和平衡，再学抬腿
                progress = self.curriculum_iteration / 200.0
                current_weight = min_weight + (max_weight - min_weight) * progress
                stage = "Stage1(Learning to Stand)"

            elif self.curriculum_iteration <= 500:
                # 阶段2: 200-500轮，保持满权重2.0
                # 强化步态学习
                current_weight = max_weight
                stage = "Stage2(Learning Gait)"

            else:
                # 阶段3: 500-1200轮，从2.0衰减到0.2
                # 避免过度优化步态，让速度跟踪主导
                decay_iterations = 700  # 从500到1200，共700轮
                progress = min((self.curriculum_iteration - 500) / decay_iterations, 1.0)
                current_weight = max_weight - (max_weight - min_weight) * progress  # 从2.0衰减到0.2
                stage = "Stage3(Speed Tracking Focus)"

            # 更新权重
            old_weight = self.reward_manager._term_cfgs[term_idx].weight
            self.reward_manager._term_cfgs[term_idx].weight = current_weight

            # 每50轮打印一次（更频繁的反馈）
            if self.curriculum_iteration % 50 == 0 or self.curriculum_iteration == 1:
                print(f"[CURRICULUM] Iteration {self.curriculum_iteration} - {stage}: "
                      f"feet_air_time weight {old_weight:.3f} -> {current_weight:.3f}")

    def step(self, actions: torch.Tensor):

        delayed_actions = self.action_buffer.compute(actions)

        cliped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        processed_actions = cliped_actions * self.action_scale + self.robot.data.default_joint_pos

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self.command_generator.compute(self.step_dt)
        self.gait_generator.step()  # 更新步态相位

        # ========== 更新后空翻Stage状态（如果任务需要） ==========
        # 仅在后空翻任务时调用，通过检查是否有update_backflip_stages函数
        if hasattr(self.cfg, 'task_type') and self.cfg.task_type == 'backflip':
            from legged_lab.mdp.backflip_rewards import update_backflip_stages
            update_backflip_stages(self)

        # 更新课程学习（动态调整奖励权重）
        self.update_curriculum()

        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {
            "critic": critic_obs,
            "rnd_state": actor_obs.clone(),  # RND使用actor观测作为状态
        }

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        reset_buf = torch.any(
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                    dim=-1,
                ),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[6:9] = 0
            noise_vec[9 : 9 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[9 + self.num_actions : 9 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[9 + self.num_actions * 2 : 9 + self.num_actions * 3] = 0.0
            self.noise_scale_vec = noise_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {"Curriculum/terrain_levels": torch.mean(self.scene.terrain.terrain_levels.float())}
        return extras

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {
            "critic": critic_obs,
            "rnd_state": actor_obs.clone(),  # RND使用actor观测作为状态
        }
        return actor_obs, self.extras

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)
