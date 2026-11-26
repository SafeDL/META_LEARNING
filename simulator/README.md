# 交通仿真

本目录将场景物理参数和被测驾驶控制器组合成闭环仿真执行，生成受遮挡约束的可见观测，并记录完整车辆轨迹、原生碰撞事件及连续安全距离测量。

| 平台 | 职责 |
|---|---|
| [HighwayEnv](highway/README.md) | 论文主实验使用的四类直道交互闭环仿真环境（20 Hz 物理／5 Hz 决策）、几何遮挡观测、策略修复课程以及离散／连续强化学习训练环境 |
| [MetaDrive](metadrive/README.md) | 独立的切入、汇入、环岛闭环执行，以及地图令牌化、交通控制、反事实教师和场景搜索（预留用于后续跨平台扩展） |

被测驾驶策略来自 [controllers/](../controllers/README.md)。[experiments/](../experiments/README.md) 负责候选采样、物理执行库采集、受限付费查询和轨迹 RSS 裕度派生。12-ADS 核心物理执行库（16,000 条历史 + 96,000 条目标）分别封存于 [原队列物理库](../results/k_shot_adaptation/physical/)、[修订 DQN 目标库](../results/dqn_continual_learning/k_shot_adaptation/physical/target/) 与 [跨架构目标物理库](../results/public_ads/physical/target/)。
