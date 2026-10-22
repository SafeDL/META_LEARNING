# 仿真环境

项目保留 HighwayEnv 和 MetaDrive 两个平台。当前碰撞发现主实验使用 HighwayEnv；MetaDrive 提供另一套场景、控制器和测试工具，继续在工程内维护。

| 目录 | 用途 |
|---|---|
| [highway](highway/README.md) | 前车制动、邻车切入、切出暴露障碍、换道后车冲突，以及驾驶策略训练 |
| [metadrive](metadrive/README.md) | 切入、汇入和环岛场景，地图处理、车辆控制、安全测量及场景搜索 |

驾驶控制器位于 [controllers](../controllers/README.md)，按平台分目录存放。HighwayEnv 的数据采集、候选库回放和连续搜索由 [benchmark](../benchmark/README.md) 调度；MetaDrive 的运行脚本和测试位于其子目录。两个平台的结果分别记录。
