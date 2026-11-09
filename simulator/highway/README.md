# HighwayEnv 场景执行

本目录执行直道交通交互，记录真实碰撞、最小间距、TTC 和车辆轨迹。

| 文件 | 职责 |
|---|---|
| [collision_env.py](collision_env.py) | `CollisionWorld` 支持跟车、前车制动、切入、切出暴露障碍和换道后车冲突，共五类基础场景 |
| [interaction_env.py](interaction_env.py) | `InteractionWorld` 实现邻道开放／受限的配对切入 |
| [observation.py](observation.py) | 生成控制器可见状态并处理遮挡 |
| [training_env.py](training_env.py) | 提供 PPO／DQN 的观测、动作和训练奖励 |

当前论文使用跟车、前车制动、切出和换道后车冲突；切入实现继续供控制器训练与相关接口使用。场景参数和校验见 [scenarios.py](../../experiments/scenarios.py)。

物理频率为 20 Hz，决策频率为 5 Hz，默认执行上限为 10 秒。自车动作由控制器决定，背景车辆按场景脚本运动；可见性由当前几何状态确定。评估端在查询后从实际轨迹派生 RSS，控制器接收当前可见观测。

控制器见 [controllers/](../../controllers/README.md)，采集与逐次反馈见 [experiments/](../../experiments/README.md)。
