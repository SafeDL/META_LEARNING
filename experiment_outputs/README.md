# 实验产物

本目录保存测试方法的模型、预算内选例记录、比较报告和审计材料。一次驾驶系统执行产生的场景参数、响应与轨迹保存在 [test_records](../test_records/README.md)；多个测试方法可以使用同一批执行记录进行比较。

## 查看结果

| 入口 | 内容 |
|---|---|
| [独立验证报告](benchmark/collision/discovery/independent_validation/report.md) | 整体发现效率、严格留家族迁移、交通交互与成本 |
| [开发比较](benchmark/collision/discovery/domain_comparison/report.md) | 基线比较及历史信息、目标反馈的开发期对照 |
| [方法定型](benchmark/collision/discovery/domain_comparison/diagonal_controls/finalization.json) | 协方差选择与匹配机制对照 |
| [实验记录索引](benchmark/README.md) | 协议、预算审计、驾驶资格和历史响应概览 |
| [主方法模型与运行](gcrt/README.md) | 基础核、历史编码器、来源后验与逐次查询 |

独立验证包含1,880项实际运行、240,640次回放查询和600条跨条件复用引用；目标候选库由20,480次仿真执行组成。比较范围与查询规则见 [实验设置](../experiments/README.md)。

## 目录内容

`benchmark/` 汇总协议、审计、统计、图表与报告；其余目录按方法保存模型和逐次运行。`gcrt/` 对应本文方法，`random/`、`av_bo_rf_ei/`、`samota/`、`scenario_fuzz/`、`paired_system_active_evaluation/` 和 `indago/` 对应六个基线。

每次方法运行记录场景选择顺序、已查询反馈及耗时。概率模型还按实现保存查询前预测、来源权重或模型更新信息。

## 记录维护

当前保留独立验证、方法定型开发比较及其数据、模型、成本和审计依赖。已完成的正式运行按实验任务完整保存，数值诊断和索引恢复的处理结论单独保留。

过时文件的删除清单见 [维护记录](benchmark/collision/discovery/independent_validation/model_construction/source_maintenance.json) 中的 `obsolete_output_cleanup`。原始协议和模型元数据保留登记时的内容；其中的 `results/` 根路径读取时定位到本目录。`benchmark` 和 `gcrt` 也是原实验沿用的标识，当前源码入口见 [experiments](../experiments/README.md) 与 [history_transfer](../testing_methods/history_transfer/README.md)。
