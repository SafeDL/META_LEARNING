# 本文方法的模型与运行记录

`gcrt` 是历史响应驱动的跨系统碰撞场景发现方法在已有实验中的登记标识。本目录保存该方法的模型、历史后验和逐次查询结果。

| 位置 | 内容 |
|---|---|
| [uniform_models](collision/uniform_models/) | 5类场景 × 5种来源范围的25组最终模型 |
| [consistent_models](collision/consistent_models/) | 最终基础核复用的原检查点与训练记录 |
| [独立运行](collision/discovery/independent_validation/) | 主方法、静态历史、聚合历史教师加反馈的查询记录 |
| [开发运行](collision/discovery/domain_comparison/) | 开发比较、协方差和匹配机制对照 |

每组最终模型包含基础核 `target_kernel.pt`、历史响应编码器 `historical_encoder.pt` 和来源后验 `source_priors.npz`，对应 JSON 文件记录训练、数据范围与构建成本。五种来源范围为全来源，以及分别排除规则、优化、搜索、学习类别。

独立运行共480项，其中本文方法400项、两个交互机制对照各40项。每项记录场景选择顺序、已查询碰撞反馈、查询前概率、权重与运行耗时。

冻结前单精度数值问题的原因与修复结论见 [诊断记录](collision/discovery/domain_comparison/numerical_diagnosis/stage_a/numerical_diagnosis.json)。训练流程见 [主方法说明](../../testing_methods/history_transfer/README.md)，发现效率、迁移与成本见 [独立报告](../benchmark/collision/discovery/independent_validation/report.md)。
