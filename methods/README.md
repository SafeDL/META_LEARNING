# 方法与基线

当前主方法是 [CG-BRT](concordance_gated_bayesian_risk_transfer/README.md)，通过已测试场景的风险反馈发现 `R > 0.5` 的场景。

| 目录 | 作用 |
|---|---|
| [concordance_gated_bayesian_risk_transfer](concordance_gated_bayesian_risk_transfer/README.md) | 主方法：风险预测、排序门控与选例 |
| [history_guided_testing](history_guided_testing/README.md) | 公共风险 GP、场景、历史数据和标准基线 |
| [historical_expert_mixture](historical_expert_mixture/README.md) | 六来源历史风险预测与高斯混合基线 |
| [scale_adaptive_expert_mixture](scale_adaptive_expert_mixture/README.md) | Student-t 密度与尺度混合基线 |
| [ras_frt_uq](ras_frt_uq/README.md) | 使用碰撞反馈的迁移基线 |
| [response_baselines](response_baselines/README.md) | 冻结响应模型与基线推断 |
| [egfd](egfd/README.md) | 使用风险和碰撞反馈的冻结对照 |

这些目录用于当前方法或结果比较。历史分支保存在[恢复归档](../archives/README.md)，实验入口见[experiments](../experiments/README.md)。
