# 数据、模型与结果索引

| 目录 | 保存内容 |
|---|---|
| [data/](data/README.md) | 历史响应、原始执行、轨迹和评估端目标真值 |
| [runs/](runs/README.md) | 模型、付费运行记录、协议、报告和图片 |
| [maintenance/](maintenance/README.md) | 来源身份、文件整理及回归核验记录 |

`cache/` 由工具和测试按需生成，可在运行结束后清理。数据准备与导出入口见 [experiments/](../experiments/README.md)。

## 当前论文结果

[paper/reports/results.md](runs/joint_response/paper/reports/results.md)汇总统一候选后的主比较。该目录同时保存[协议](runs/joint_response/paper/reports/protocol.json)、[复用清单](runs/joint_response/paper/reports/reuse.json)、[模型选择](runs/joint_response/paper/reports/model_selection.json)及[图片索引](runs/joint_response/paper/reports/visual_check.json)。

| 位置 | 用途 |
|---|---|
| [共享／独立核模型](runs/joint_response/method_exploration/e2/models/) | 当前选定结构的基础核与历史编码器 |
| [原联合核模型](runs/joint_response/models/) | 原结构及相关机制对照 |
| [基线模型](runs/joint_response/pilot/models/) | 按场景训练的历史分类器 |
| [参数来源](runs/joint_response/reports/parameter_sources/parameters.csv) | 参数含义、原文位置及来源数值 |
| [历史响应分析](runs/joint_response/reports/history_response/report.md) | 已执行历史数据的碰撞与 RSS 分布 |

## 支持材料

以下记录按各自冻结配置读取，模型和候选范围可能与当前主比较不同。

| 内容 | 报告 |
|---|---|
| 固定历史、机制对照 | [fixed](runs/joint_response/fixed/reports/results.md)、[mechanisms](runs/joint_response/mechanisms/reports/results.md) |
| 知识积累、候选规模、计算规模 | [sequential](runs/joint_response/sequential/reports/results.md)、[sensitivity](runs/joint_response/sensitivity/reports/results.md)、[scaling](runs/joint_response/scaling/reports/results.md) |
| 共同反馈与历史来源对照 | [attribution](runs/joint_response/attribution/reports/results.md) |
| 局部权重、共享／独立核、联合前瞻 | [e1](runs/joint_response/method_exploration/e1/reports/results.md)、[e2](runs/joint_response/method_exploration/e2/reports/results.md)、[e3](runs/joint_response/method_exploration/e3/reports/results.md) |
| 发现集合与查询时机分解 | [gain_analysis](runs/joint_response/method_exploration/reports/gain_analysis.md) |

`result.json.gz` 保存选择、反馈、指标和耗时，`completed.json.gz` 保存付费归档；轨迹通过 `execution_ref` 共享。目标完整真值与方法已查询记录分别保存。文件迁移和核验见[维护记录](maintenance/README.md)。
