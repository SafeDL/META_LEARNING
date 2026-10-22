# 实验执行与结果复核

当前碰撞实验的记录集中在 [collision](collision/)。这里回答三件事：驾驶系统是否具备基本能力，实验是否按设定执行，报告能否从原始数据重新计算得到。

| 位置 | 内容 |
|---|---|
| [qualification](collision/qualification/) | 跟车、制动、停车、换道和后车约束五项能力检查及轨迹 |
| [policy_construction](collision/policy_construction/) | 学习型驾驶策略的训练登记、资格和配置确定记录 |
| [development](collision/development/)、[continuous_development](collision/continuous_development/) | 候选库和连续搜索的开发比较 |
| [data_overview](collision/data_overview/) | 历史系统的碰撞分布与响应差异概览 |
| [final_result_audit.json](collision/final_result_audit.json) | 10880 项候选库运行的原始轨迹、标签、指标及预算复算 |
| [certification_result_audit.json](collision/certification_result_audit.json) | 96 份参考系统认证与两种新增碰撞标签的核对 |
| [continuous_result_audit.json](collision/continuous_result_audit.json) | 100 项真实连续运行的原始结果、预算和成本复核 |
| [statistical_report_audit.json](collision/statistical_report_audit.json) | 472 项统计比较及结果明细的独立核对 |
| [completion_audit.json](collision/completion_audit.json) | 原正式验收时逐项要求与对应证据 |

成本复核文件为 `construction_cost_inventory.json`、`finite_cost_report_verification.json` 和 `continuous_cost_report_verification.json`；图表复核文件为 `finite_figure_verification.json`、`continuous_figure_verification.json`。这些记录区分真实仿真、回放计算和前期模型建设，避免重复累计。

历史验收记录保留原样，其中原说明文件的内容已保存为 [验收时的说明快照](collision/documentation_snapshot_before_readme_migration.txt)。当前说明以根目录 [README](../../README.md) 为准。旧风险实验的记录已移至工程外 `META_LEARNING_archive/legacy/results/benchmark/`。

命令与复核步骤见 [实验运行说明](../../benchmark/README.md)，比较结果见 [结果索引](../gcrt/README.md)。
