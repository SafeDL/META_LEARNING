# 数据准备与实验运行

本目录连接场景、驾驶控制器和测试方法，负责执行库准备、预算查询、任务归档及报告导出。方法实现见 [testing_methods/](../testing_methods/README.md)，已有产物见[结果索引](../artifacts/README.md)。

## 模块分工

| 模块 | 职责 |
|---|---|
| [scenarios.py](scenarios.py)、[domain.py](domain.py)、[scenario_sampling.py](scenario_sampling.py) | 定义参数、校验物理约束，生成并归一化候选 |
| [executions.py](executions.py)、[schema.py](schema.py)、[storage.py](storage.py) | 采集和读取执行库，保存响应、轨迹及版本凭据 |
| [query.py](query.py)、[rss.py](rss.py)、[domain_risk.py](domain_risk.py) | 按预算开放本次执行反馈，派生 RSS 与基线所需风险量 |
| [history.py](history.py)、[retention.py](retention.py) | 归档完成任务、构建来源、更新共享模型和复核旧任务 |
| [failure_metrics.py](failure_metrics.py)、[costs.py](costs.py) | 统计失效、覆盖、参数距离和计算成本 |

完整目标真值由评估端持有。选择器只接收公开候选、获准历史信息及已付费反馈；任务归档以 `execution_ref` 共享原始轨迹。

## 数据和模型

| 入口 | 实际操作 |
|---|---|
| `python -B -m experiments.prepare_joint` | 核验原始历史记录，补采、派生 RSS 并合并至 `artifacts/data/joint_response/` |
| `python -B -m experiments.prepare_targets` | 准备评估端目标执行库；默认处理首轮两个目标 |
| `python -B -m testing_methods.history_transfer.training` | 按场景构建原联合核及历史编码器，输出至 `joint_response/models/` |
| [共享／独立核构建](../testing_methods/history_transfer/README.md#构建与使用) | 显式传入核类型，从历史档案训练所选结构 |

`prepare_joint.merge_scenario` 可合并已完成批次，`training.build_scope` 可指定数据清单。设备由训练配置的 `device` 指定。

## 方法运行

以下入口会按各自登记执行采集、训练或选择器回放。表中的输出目录均相对于 `artifacts/runs/joint_response/`。

| 入口 | 用途与输出 |
|---|---|
| `python -B -m experiments.paper_validation` | 复用论文数据、模型和已完成运行，补齐登记缺口；输出 `paper/` |
| `python -B -m experiments.parallel_comparisons` | 按 ADS 并行执行原配置固定历史比较；输出 `fixed/` |
| `python -B -m experiments.joint_validation --suite sequential` | 按任务顺序开放历史、归档和更新模型；输出 `sequential/` |
| `parallel_comparisons.run('mechanisms')` | 原配置机制对照；输出 `mechanisms/` |
| `parallel_comparisons.run('sensitivity')` | 嵌套候选库比较；输出 `sensitivity/` |
| `python -B -m experiments.scale_joint` | 测量来源构建、适配和预测的 CPU／CUDA 成本；输出 `scaling/` |

固定比较由 [pilot.py](pilot.py) 调度，各任务内部顺序查询。`result.json.gz` 保存选择与反馈，`completed.json.gz` 保存付费归档，`reports/` 保存协议和汇总。函数入口可用 `python -B -c "from experiments.parallel_comparisons import run; run('mechanisms')"` 调用。

## 核验与导出

| 入口 | 用途 |
|---|---|
| [check_training.py](check_training.py) | 使用临时执行和模型检查训练、适配及反馈隔离 |
| [paper_report.py](paper_report.py) | 复核论文结果、补齐完成归档，导出表格及图片 |
| [audit_joint.py](audit_joint.py)、[audit_sources.py](audit_sources.py)、[audit_accumulation.py](audit_accumulation.py) | 核验指标与账本、来源权限及任务间积累 |
| [report_joint.py](report_joint.py)、[report_extensions.py](report_extensions.py) | 导出原配置比较及规模实验的报告 |
| [attribution.py](attribution.py)、[attribution_report.py](attribution_report.py) | 在已有执行库上回放共同反馈和来源对照，并导出收益归因 |
| [rss_exploration_report.py](rss_exploration_report.py)、[analyze_gains.py](analyze_gains.py) | 读取已完成的方法比较，复算预测、发现集合及查询时机 |
| [report_history.py](report_history.py)、[schematics.py](schematics.py) | 导出历史响应分析、场景及方法示意图 |
| [render_joint.py](render_joint.py)、[animation.py](animation.py) | 从实际付费轨迹生成 PNG 和按原始时序播放的 GIF |

命令从项目根目录运行。通用的 `audit_joint`、`audit_sources`、`report_joint`、`render_joint` 默认读取 `pilot/`；其他目录通过对应函数的 `root` 参数指定。正式图片保存到所属运行的 `reports/figures/`。
