# 实测数据

当前正式碰撞实验的数据集中在 [collision](collision/)。这里保存场景参数、驾驶系统实际运行后的反馈、完整轨迹和调用记录；模型及比较报告位于 [results](../results/README.md)。

| 位置 | 保存什么 |
|---|---|
| [collision/history](collision/history/) | 8 个历史系统在四类场景中的反馈，共 32 份档案 |
| [collision/development](collision/development/) | 4 个开发系统在四类场景中的反馈，共 16 份档案 |
| [collision/physical](collision/physical/) | 历史与开发档案对应的原始执行轨迹、尝试记录和耗时 |
| [collision/targets](collision/targets/) | 8 个目标系统、四类场景、三种数据条件的真实执行库，共 96 库 |
| [collision/certification](collision/certification/) | 为确认版本新增及历史集合特有碰撞而补做的参考系统执行 |
| [frozen_protocol.json](collision/frozen_protocol.json) | 正式候选库实验采用的系统、参数范围、预算、模型和代码版本 |
| [frozen_continuous.json](collision/frozen_continuous.json) | 连续搜索采用的独立任务、参数和模型配置 |

每份历史或开发档案包含 512 个场景。目标库分为历史坐标复用、同分布新坐标和留出的困难组合三种条件，每个系统、场景和条件各有 512 个参数点。

目标库由实验执行器管理。测试选择算法只能通过 [查询接口](../benchmark/query.py) 获取已经查询的反馈，不能直接读取未查询的碰撞标签。连续搜索的新坐标由算法提出并实际执行，其轨迹随运行结果保存。

早期 `data/history/` 和 `data/scenarios/` 风险实验已移至工程外的 `META_LEARNING_archive/legacy/data/`，原文件保留。当前冻结配置及正式数据保持原路径；MetaDrive 的源码、场景配置和测试继续保留在 [simulation/metadrive](../simulation/metadrive/README.md)。

系统和指标的通俗解释见 [项目说明](../README.md)，采集及复核命令见 [实验运行说明](../benchmark/README.md)。
