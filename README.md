# R02 Backflip Training

RO2 四足机器人后空翻强化学习训练项目，基于 IsaacLab 和 RSL-RL 框架。

## 项目简介

本项目实现了 RO2 四足机器人的后空翻动作训练。主要特点：

- 基于 PPO 算法的后空翻训练
- 针对 RO2 机器人坐标系设计（Y轴朝前，后空翻绕X轴旋转）
- 时间窗口化奖励设计
- 多种约束惩罚（膝盖接触、动作对称等）

## 文件结构

```
R02backflip/
├── README.md                           # 本文件
├── legged_lab/
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── base/                       # 基础环境类
│   │   │   ├── base_env.py             # 环境基类
│   │   │   ├── base_env_config.py      # 配置基类
│   │   │   ├── base_config.py
│   │   │   └── gait_generator.py       # 步态生成器
│   │   └── r02_2_go2style/             # 后空翻环境
│   │       ├── __init__.py
│   │       ├── config.py               # 环境和奖励配置
│   │       ├── rewards.py              # 奖励函数定义
│   │       └── go2style_env.py         # 环境实现
│   ├── assets/
│   │   ├── __init__.py
│   │   └── r02_2/
│   │       ├── __init__.py
│   │       └── r02_2.py                # 机器人配置（需修改USD路径）
│   ├── mdp/
│   │   ├── __init__.py
│   │   ├── rewards.py                  # 通用奖励函数
│   │   └── backflip_rewards.py         # 后空翻专用奖励
│   ├── scripts/
│   │   ├── train.py                    # 训练脚本
│   │   └── play.py                     # 可视化脚本
│   └── utils/
│       ├── __init__.py
│       ├── cli_args.py                 # 命令行参数
│       ├── task_registry.py            # 任务注册
│       ├── keyboard.py                 # 键盘控制
│       └── env_utils/
│           ├── __init__.py
│           └── scene.py                # 场景工具
```

## 环境要求

- Ubuntu 22.04
- CUDA 11.8+
- Python 3.10
- Isaac Sim 4.5+
- IsaacLab
- RSL-RL

## 复现步骤

### 1. 安装依赖

```bash
# 安装 IsaacLab（参考官方文档）
# https://isaac-sim.github.io/IsaacLab/

# 安装 RSL-RL
pip install rsl-rl
```

### 2. 准备机器人模型

修改 `legged_lab/assets/r02_2/r02_2.py` 中的 USD 路径：

```python
usd_path="/path/to/your/robot.usd",  # 替换为你的机器人USD路径
```

### 3. 训练

```bash
cd /path/to/IsaacLab

# 开始训练（headless模式）
./isaaclab.sh -p /path/to/R02backflip/legged_lab/scripts/train.py \
    --task r02_go2style \
    --headless \
    --num_envs 2048 \
    --max_iterations 10000 \
    --logger tensorboard

# 可视化训练（调试用）
./isaaclab.sh -p /path/to/R02backflip/legged_lab/scripts/train.py \
    --task r02_go2style \
    --num_envs 64 \
    --max_iterations 100
```

### 4. 可视化测试

```bash
./isaaclab.sh -p /path/to/R02backflip/legged_lab/scripts/play.py \
    --task r02_go2style \
    --num_envs 4 \
    --load_run /path/to/logs/experiment_name \
    --checkpoint model_5000.pt
```

### 5. TensorBoard 监控

```bash
tensorboard --logdir=/path/to/logs --port=6006
```

## 训练配置说明

### 关键参数 (config.py)

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_envs` | 4096 | 并行环境数 |
| `max_episode_length_s` | 2.0 | 单次尝试时长 |
| `stiffness` | 35.0 | PD控制器刚度 |
| `damping` | 0.5 | PD控制器阻尼 |
| `action_scale` | 0.4 | 动作缩放 |

### 奖励函数

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| `ang_vel_x` | +35.0 | 后空翻旋转奖励（0.5-1.0s） |
| `lin_vel_z` | +35.0 | 起跳奖励（0.5-0.75s） |
| `orientation_control` | -2.0 | 姿态追踪 |
| `height_control` | -20.0 | 高度保持 |
| `action_sym` | -3.0 | 动作对称性 |
| `knee_contact` | -7.0 | 膝盖触地惩罚 |
| `knee_height` | -10.0 | 膝盖高度惩罚 |

### 时间阶段

| 阶段 | 时间 | 动作 |
|------|------|------|
| 准备 | 0-0.5s | 站立、蓄力 |
| 起跳 | 0.5-0.75s | 向上跳跃 |
| 翻转 | 0.5-1.0s | 后空翻旋转 |
| 着陆 | 1.0-2.0s | 稳定着陆 |

## 坐标系说明

RO2 机器人坐标系：

| 轴 | 方向 |
|----|------|
| Y | 前进方向 |
| X | 侧向（左） |
| Z | 垂直向上 |
| 后空翻旋转轴 | X轴 |

