# 碰撞发现实验结果

当前结果集中在 [collision](collision/)。其中不仅有主方法，也包含相同任务下各基线的查询结果。候选库结果与连续搜索结果分别汇总。

| 位置 | 保存什么 |
|---|---|
| [候选库报告](collision/report/report.md) | 三种数据条件、整体比较、留类别、来源替换、消融和成本 |
| [候选库结果明细](collision/report/runs.csv) | 每项运行的场景、方法、种子、发现数量及耗时 |
| [runs](collision/runs/)、[run_index.json](collision/run_index.json) | 10880 项候选库运行的逐次查询记录与索引 |
| [models](collision/models/) | 本方法的 116 个模型及训练记录 |
| [certificates](collision/certificates/)、[new_collision_metrics.json](collision/new_collision_metrics.json) | 参考系统真实执行支持的新增碰撞认证及逐次发现统计 |
| [连续报告](collision/continuous/report/report.json) | 100 项连续搜索的发现数量、曲线、统计和成本 |
| [连续运行索引](collision/continuous/run_index.json) | 每项真实连续运行的反馈与执行记录位置 |
| [pipeline](collision/pipeline/) | 执行阶段、日志和原始退出码 |

图表与对应报告保存在同一目录，各有 PNG、SVG、PDF 三种格式。连续报告另有 `equal_count_behavior_coverage.json` 和 `auxiliary_safety.json`，分别记录同量碰撞下的行为覆盖和辅助安全测量。

## 阅读结果时看什么

主条件为同分布新坐标，每种方法有 160 项运行，每项预算 128 次且包含 8 次共同初始化。主指标为去重碰撞数，AUC 表示查询过程中累计发现数量的平均值；行为覆盖衡量碰撞类型的多样性。首次发现统计保留预算内没有找到碰撞的运行。

本方法主条件平均发现 33.04 个去重碰撞，历史排序为 21.58，FSBO 碰撞适配为 36.06。完整配对区间、其他条件及解释见 [报告](collision/report/report.md) 和根目录 [项目说明](../../README.md)。方法间的版本新增与历史集合特有碰撞分别计数，不能混成同一个指标。

旧的风险阈值、停止预测、收益校正和保留率实验已移至工程外归档。需要从现有运行重新生成当前报告时，使用：

```powershell
python -B -m benchmark.report_collision
python -B -m benchmark.report_continuous
```

实现见 [主方法说明](../../methods/gcrt/README.md)，原始数据见 [数据说明](../../data/README.md)，独立核对记录见 [复核结果](../benchmark/README.md)。
