# HighwayEnv 场景执行

本目录将场景参数和驾驶控制器组合成闭环仿真，记录碰撞、最小间距、TTC 和车辆轨迹。当前任务包括前车制动、邻车切入、前车切出暴露障碍、换道后车冲突，以及邻道开放／受限切入。

| 文件 | 用途 |
|---|---|
| [collision_env.py](collision_env.py) | 创建道路与车辆，执行四类基础场景，记录碰撞和轨迹 |
| [interaction_env.py](interaction_env.py) | 执行邻道开放／受限的配对切入场景 |
| [observation.py](observation.py) | 生成控制器可见的交通状态，处理障碍物遮挡 |
| [training_env.py](training_env.py) | 为 PPO、DQN 提供训练环境、动作与奖励 |

物理仿真频率20 Hz，驾驶决策频率5 Hz，单场景最长10秒。被测车辆由驾驶控制器决定动作，背景车辆按场景脚本行驶；障碍物可见性由当前几何关系决定。

场景定义见 [scenarios.py](../../experiments/scenarios.py)，控制器见 [controllers](../../controllers/README.md)，采集和预算查询流程见 [experiments](../../experiments/README.md)。预先采集的仿真执行保存到 `test_records/`，回放时由查询接口逐次提供反馈。
