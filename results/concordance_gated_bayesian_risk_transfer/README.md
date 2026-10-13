# CG-BRT 结果

已完成六系统 × 两池 × 五种子，共 60 条轨迹，每条预算 200。主方法只接收风险 `R`，高风险定义为 `R > 0.5`。

| 指标 | 五种子平均结果 |
|---|---:|
| 发现曲线面积 Area | 46.170% |
| 高风险覆盖率 Recall | 72.695% |
| H50 / H100 / H200 | 45.65 / 89.32 / 141.25 |
| 预算覆盖率理论上限 | 78.663% |
| 固定初始化后覆盖率上限 | 78.487% |
| 辅助碰撞 Area / Recall | 52.518% / 76.661% |

H50、H100、H200 是对应测试次数下发现的高风险数量。Recall 是发现数占该池全部高风险场景的比例；Area 是 200 步覆盖率的平均值，反映发现速度。百分比按轨迹计算后取平均。

| 文件 | 内容 |
|---|---|
| [comparison.md](comparison.md)、[comparison.json](comparison.json) | 完整对照、逐系统上限与统计 |
| [driving/](driving/) | 60 条主方法轨迹及反馈、更新记录 |
| [protocol.json](protocol.json)、[baseline_catalog.json](baseline_catalog.json) | 实验设置与基线信息 |
| [status.json](status.json)、[verification.json](verification.json) | 完成状态与核验 |
| [source_inputs.zip](source_inputs.zip) | 34 份源码和 6 份历史数据、核输入快照 |
| [cleanup.json](cleanup.json) | 清理记录 |

当前 Area 和 Recall 在五种子表中最高；相对原方法的差距较小，尚无显著优势。比较与核验命令见[实验说明](../../experiments/concordance_gated_bayesian_risk_transfer/README.md)。
