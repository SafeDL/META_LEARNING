# PPO 权重继承链 双向版本变化测试

协议：`ppo-offset-calibration-bidirectional-v1`；数据身份：`offset_calibration_confirmation`；1089 个冻结场景，3267 次配对物理执行。
主比较方法：`directed_offset_calibrated_ucb`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| ppo_release_v0→ppo_release_v1 | R | 0 | random | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | R | 0 | static_risk | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | R | 0 | static_boundary | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | R | 0 | center_residual | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | R | 0 | coordinate_residual | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | R | 0 | target_only | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | R | 0 | directed_residual | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | R | 0 | coordinate_context_ucb_pure | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | R | 0 | directed_context_ucb_pure | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | R | 0 | directed_context_ucb_loose | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | R | 0 | coordinate_offset_calibrated_ucb | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | R | 0 | directed_offset_calibrated_ucb | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | I | 46 | random | 3.40 | 0.163 |
| ppo_release_v0→ppo_release_v1 | I | 46 | static_risk | 1.00 | 0.033 |
| ppo_release_v0→ppo_release_v1 | I | 46 | static_boundary | 6.00 | 0.186 |
| ppo_release_v0→ppo_release_v1 | I | 46 | center_residual | 1.00 | 0.062 |
| ppo_release_v0→ppo_release_v1 | I | 46 | coordinate_residual | 1.00 | 0.067 |
| ppo_release_v0→ppo_release_v1 | I | 46 | target_only | 11.00 | 0.552 |
| ppo_release_v0→ppo_release_v1 | I | 46 | directed_residual | 1.00 | 0.062 |
| ppo_release_v0→ppo_release_v1 | I | 46 | coordinate_context_ucb_pure | 6.00 | 0.100 |
| ppo_release_v0→ppo_release_v1 | I | 46 | directed_context_ucb_pure | 6.00 | 0.343 |
| ppo_release_v0→ppo_release_v1 | I | 46 | directed_context_ucb_loose | 6.00 | 0.243 |
| ppo_release_v0→ppo_release_v1 | I | 46 | coordinate_offset_calibrated_ucb | 6.00 | 0.286 |
| ppo_release_v0→ppo_release_v1 | I | 46 | directed_offset_calibrated_ucb | 6.00 | 0.371 |
| ppo_release_v1→ppo_release_v2 | R | 21 | random | 0.80 | 0.044 |
| ppo_release_v1→ppo_release_v2 | R | 21 | static_risk | 8.00 | 0.467 |
| ppo_release_v1→ppo_release_v2 | R | 21 | static_boundary | 4.00 | 0.229 |
| ppo_release_v1→ppo_release_v2 | R | 21 | center_residual | 7.00 | 0.424 |
| ppo_release_v1→ppo_release_v2 | R | 21 | coordinate_residual | 7.00 | 0.395 |
| ppo_release_v1→ppo_release_v2 | R | 21 | target_only | 1.00 | 0.052 |
| ppo_release_v1→ppo_release_v2 | R | 21 | directed_residual | 6.00 | 0.410 |
| ppo_release_v1→ppo_release_v2 | R | 21 | coordinate_context_ucb_pure | 11.00 | 0.438 |
| ppo_release_v1→ppo_release_v2 | R | 21 | directed_context_ucb_pure | 5.00 | 0.352 |
| ppo_release_v1→ppo_release_v2 | R | 21 | directed_context_ucb_loose | 6.00 | 0.357 |
| ppo_release_v1→ppo_release_v2 | R | 21 | coordinate_offset_calibrated_ucb | 13.00 | 0.495 |
| ppo_release_v1→ppo_release_v2 | R | 21 | directed_offset_calibrated_ucb | 11.00 | 0.443 |
| ppo_release_v1→ppo_release_v2 | I | 13 | random | 1.60 | 0.069 |
| ppo_release_v1→ppo_release_v2 | I | 13 | static_risk | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 13 | static_boundary | 3.00 | 0.157 |
| ppo_release_v1→ppo_release_v2 | I | 13 | center_residual | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 13 | coordinate_residual | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 13 | target_only | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 13 | directed_residual | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 13 | coordinate_context_ucb_pure | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 13 | directed_context_ucb_pure | 2.00 | 0.043 |
| ppo_release_v1→ppo_release_v2 | I | 13 | directed_context_ucb_loose | 2.00 | 0.090 |
| ppo_release_v1→ppo_release_v2 | I | 13 | coordinate_offset_calibrated_ucb | 2.00 | 0.033 |
| ppo_release_v1→ppo_release_v2 | I | 13 | directed_offset_calibrated_ucb | 2.00 | 0.024 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
