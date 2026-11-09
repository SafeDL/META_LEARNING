# 交通仿真

本目录将场景参数和驾驶控制器组合成闭环执行，生成可见观测，记录车辆轨迹、碰撞事件及安全测量。

| 平台 | 职责 |
|---|---|
| [HighwayEnv](highway/README.md) | 当前论文的直道车辆交互、遮挡观测和 PPO／DQN 训练环境 |
| [MetaDrive](metadrive/README.md) | 独立的切入、汇入、环岛执行，以及地图、交通控制和场景搜索 |

驾驶策略来自 [controllers/](../controllers/README.md)。[experiments/](../experiments/README.md)负责采集、预算查询和轨迹 RSS 派生，执行档案保存到 [artifacts/data/](../artifacts/data/README.md)。
