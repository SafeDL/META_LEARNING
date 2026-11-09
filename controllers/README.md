# 被测驾驶控制器

本目录实现被测 ADS。控制器接收当前可见交通状态，输出加速、制动和换道动作；场景选择由 [testing_methods/](../testing_methods/README.md) 负责。

| 位置 | 职责 |
|---|---|
| [highway/catalog.py](highway/catalog.py) | 登记系统身份、控制路线、参数和训练种子 |
| [highway/factory.py](highway/factory.py) | 创建控制器并接入统一决策接口 |
| [highway/rule.py](highway/rule.py)、[optimization.py](highway/optimization.py)、[search.py](highway/search.py) | IDM／MOBIL、FVDM、线性 MPC、非线性轨迹优化、UCT 和束搜索 |
| [highway/learning.py](highway/learning.py)、[train.py](highway/train.py) | 加载和训练 PPO／DQN |
| [metadrive/](metadrive/) | MetaDrive 的 IDM 控制器与驾驶配置 |

HighwayEnv 登记了八个历史系统和八个独立目标。控制器仅使用可见观测，仿真执行、遮挡和轨迹记录见 [simulator/](../simulator/README.md)。

[checkpoints/collision/](highway/checkpoints/collision/) 保存历史与目标 PPO／DQN 的四组已选权重，每组包含 `best.zip`、训练记录和资格评估曲线。[资格记录](../artifacts/runs/qualification/reports/)保存各系统的驾驶能力检查结果。

从头准备控制器时，先执行 `python -B -m experiments.qualification`，再用 `python -B -m controllers.highway.train <ADS_ID>` 训练学习控制器。可选 ID 为 `history_ppo`、`history_dqn`、`independent_ppo` 和 `independent_dqn`；训练输出目录须为空。
