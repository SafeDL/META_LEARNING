# 共享方法组件

| 文件 | 职责 |
|---|---|
| [prior.py](prior.py) | 历史来源均值、物理核、档案支持协方差与联合先验数据结构 |
| [features.py](features.py) | 基础核与历史关联编码器的候选特征计算、固定候选集预测缓存 |
| [workspace.py](workspace.py) | 按功能场景过滤的历史数据与模型产物位置 |
| [config.py](config.py) | 风险模型共用的观测噪声与数值精度设置 |
| [student.py](student.py) | Student-t 概率的 GPU 数值计算 |
| [counts.py](counts.py) | 零膨胀 Beta-Binomial 计数分布的概率、期望和累积分布 |
| [catalog.py](catalog.py) | 方法全称、缩写与既有实验记录 ID |
| [snapshots.py](snapshots.py) | 原始实验快照的路径对应与源码核验 |

GCRT 与基线直接导入共享源码。历史关联编码器和基础核的训练实现分别复用 [DART](../baselines/dart/README.md) 与 [HGT](../baselines/hgt/README.md) 的组件。
