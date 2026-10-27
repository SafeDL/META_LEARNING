# 交通仿真平台

本目录提供交通场景执行、车辆运动、可见观测和安全响应记录。

| 平台 | 当前用途 |
|---|---|
| [HighwayEnv](highway/README.md) | 运行五类碰撞发现任务，采集测试档案，并训练 PPO、DQN 驾驶策略 |
| [MetaDrive](metadrive/README.md) | 提供切入、汇入、环岛场景，以及独立的测试与场景搜索工具 |

当前论文的独立验证结果来自 HighwayEnv。驾驶控制器由 [controllers](../controllers/README.md) 提供，主实验的数据采集、预算查询和结果统计由 [experiments](../experiments/README.md) 组织。
