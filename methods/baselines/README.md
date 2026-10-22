# 对比方法

本目录存放与主方法比较的测试选择算法。候选库比较从同一批场景中选点；连续搜索允许算法提出新场景，两类结果分开报告。实验设置见 [项目说明](../../README.md)。

| 实现 | 如何选择场景 |
|---|---|
| [response_baselines](response_baselines/README.md) | 随机、参数空间分散选点、历史危险排序、最近来源加残差修正、无历史 GP |
| [pooled_ft](pooled_ft/README.md) | 混合历史测试样本预训练，再用目标反馈微调 |
| [fsbo](fsbo/README.md) | 把各历史系统当作独立任务学习共享模型，再用目标反馈微调 |
| [indago/search.py](indago/search.py) | 用历史分类器预测危险程度，沿影响预测较大的参数寻找更危险的新场景 |
| [samota/search.py](samota/search.py) | 结合多个预测模型、全局搜索和分组后的局部搜索提出场景 |
| [deep_hyperion/search.py](deep_hyperion/search.py) | 按两个场景特征划分格子，在各格中保留更危险的场景，再变异生成新场景 |

各方法共用查询预算、初始化场景、允许使用的历史数据和目标反馈权限。Pooled-FT 和 FSBO 的分类模型从历史与开发档案构建；当前无历史 GP 按碰撞概率贪心选例。

构建使用 `python -B -m benchmark.build_collision_models core` 和 `complete`；比较与报告入口见 [实验运行说明](../../benchmark/README.md)。

以上为已经实现并执行比较的方法。文献核查还识别出与主任务直接相关的 ENS 主动搜索、RGPE 多来源加权迁移，以及自动驾驶场景排序方法；这些尚未加入当前比较，现有集合不表述为已穷尽相关基线。

DART、HGT、LRT、CG-BRT 属于内部未发表版本，原源码及旧风险实验已在工程外归档，不作为外部基线。
