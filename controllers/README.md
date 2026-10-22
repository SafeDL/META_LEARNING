# 被测驾驶系统

本目录存放“实际开车”的控制器。测试选择算法在 [methods](../methods/README.md)，负责选择下一次测试场景；控制器接收交通信息，决定被测车辆如何行驶。

## HighwayEnv 控制器

[highway/catalog.py](highway/catalog.py) 登记了 20 个对象，由 [highway/factory.py](highway/factory.py) 创建。8 个历史系统提供测试经验，4 个开发系统用于选参，8 个目标系统用于正式比较；每个对象记录实现路线、版本或独立训练运行。

| 类别 | 实现路线 |
|---|---|
| 规则 | IDM/MOBIL；FVDM/换道规则 |
| 优化 | 线性化 MPC；非线性约束轨迹优化 |
| 搜索 | UCT 树搜索；确定性束搜索 |
| 学习 | 独立训练的 PPO；DQN，采用 Stable-Baselines3 实现 |

所有对象接收相同规则下可见的交通状态，每秒决策 5 次，车辆运动每秒更新 20 次；共享动力学和动作限制，支持持续制动直到停车。20 个对象均已通过正常跟车、可避免制动、完整停车、合法换道和后车约束五项资格检查。

PPO 和 DQN 使用左换道、保持、右换道、加速、减速五种动作；83 维输入包含可见交通状态及控制目标。五个正式学习对象各自从头训练，入口为 `python -B -m controllers.highway.train`。正式训练设置和资格记录保留在 `highway/checkpoints/collision/` 与 `results/benchmark/collision/qualification/`；其中的早期建设记录用于追溯成本，正式对象由注册表指定。

## MetaDrive 控制器

[metadrive](metadrive/) 保留在工程中，供 [MetaDrive 仿真平台](../simulation/metadrive/README.md) 使用。[registry.py](metadrive/registry.py) 登记六种 IDM 驾驶配置，[idm.py](metadrive/idm.py) 将交通状态转换为控制动作。它们与 HighwayEnv 的 20 个正式对象分开管理。

历史系统与新系统如何划分、场景和预算如何设置，见 [项目说明](../README.md)。
