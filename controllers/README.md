# 被测驾驶系统

本目录提供被测车辆的驾驶控制器：接收可见交通状态，决定加速、制动和换道。测试场景选择算法位于 [testing_methods](../testing_methods/README.md)。

## HighwayEnv

| 类别 | 实现路线 |
|---|---|
| 规则控制 | IDM／MOBIL、FVDM／换道规则 |
| 优化规划 | 线性 MPC、非线性轨迹优化 |
| 搜索规划 | UCT、束搜索 |
| 学习策略 | PPO、DQN |

[highway/catalog.py](highway/catalog.py) 登记系统角色、参数和策略检查点，[factory.py](highway/factory.py) 创建对应控制器。当前实验使用8个历史系统、4个开发系统和8个独立目标。注册表还保留8个早期目标的定义与资格记录。

上述28个系统均有通过正常跟车、可避免制动、完整停车、合法换道和后车约束的资格记录。各系统使用相同的可见交通信息规则、车辆动力学和动作限制；仿真频率20 Hz，决策频率5 Hz。

PPO、DQN 使用83维观测和左换道、保持、右换道、加速、减速五种动作。工程中保留7个学习系统的训练检查点，包括独立目标 PPO、DQN；独立训练种子分别为5107和5113。训练实现见 [train.py](highway/train.py)，独立对象的准备入口见 [independent_preparation.py](../experiments/independent_preparation.py)。

训练记录位于 [highway/checkpoints/collision](highway/checkpoints/collision/)，基本驾驶能力记录位于 [qualification](../experiment_outputs/benchmark/collision/qualification/)。场景划分和实验预算见 [experiments](../experiments/README.md)。

## MetaDrive

[metadrive](metadrive/) 提供 IDM 控制器及六种驾驶配置，由 [registry.py](metadrive/registry.py) 登记、[idm.py](metadrive/idm.py) 执行动作，供 [MetaDrive 平台](../simulator/metadrive/README.md) 的场景执行器调用。
