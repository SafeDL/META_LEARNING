# 实验协议、报告与审计

本目录集中保存实验配置、运行索引、数据与预算审计、统计报告和图表。方法模型与逐次查询文件分布在 `experiment_outputs/` 下的各方法目录，原始仿真记录位于 [test_records](../../test_records/README.md)。

| 入口 | 内容 |
|---|---|
| [独立验证报告](collision/discovery/independent_validation/report.md) | 全来源比较、严格留家族迁移、交互机制与成本 |
| [冻结协议](collision/discovery/independent_validation/protocol.json) | 方法、任务、预算、模型和统计设置 |
| [全量审计](collision/discovery/independent_validation/audit.json) | 1,880项运行、240,640次查询和600条复用引用的核对 |
| [完成核对](collision/discovery/independent_validation/completion_audit.json) | 21项实验要求及对应证据 |
| [开发比较](collision/discovery/domain_comparison/report.md) | 开发系统上的基线与机制比较 |
| [方法定型](collision/discovery/domain_comparison/diagonal_controls/finalization.json) | 协方差和匹配历史教师的选择依据 |
| [资格检查](collision/qualification/) | 跟车、制动、停车、换道及后车约束的基本驾驶能力 |
| [历史响应概览](collision/data_overview/) | 历史系统的碰撞分布与响应差异 |

成本记录区分数据采集、模型训练、目标适配和选例；并行任务的耗时与工作流总耗时分别记录。实验入口见 [experiments](../../experiments/README.md)。

原始协议与源码快照保存执行时的定义，当前维护记录见 [source_maintenance.json](collision/discovery/independent_validation/model_construction/source_maintenance.json)。
