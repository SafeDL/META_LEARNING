# PPO 权重继承链 双向版本变化测试

协议：`ppo-regression-bootstrap-bidirectional-v1`；数据身份：`regression_bootstrap_confirmation`；1089 个冻结场景，3267 次配对物理执行。
主比较方法：`directed_regression_bootstrap_ucb`；所有方法共用固定方向日程与查询预算。
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
| ppo_release_v0→ppo_release_v1 | R | 0 | coordinate_bootstrap_offset_ucb | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | R | 0 | directed_bootstrap_offset_ucb | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | R | 0 | coordinate_regression_bootstrap_ucb | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | R | 0 | directed_regression_bootstrap_ucb | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | R | 0 | static_role_coverage2 | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | R | 0 | coordinate_role_gated | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | R | 0 | prequential_bma_ucb | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | I | 42 | random | 2.50 | 0.132 |
| ppo_release_v0→ppo_release_v1 | I | 42 | static_risk | 1.00 | 0.090 |
| ppo_release_v0→ppo_release_v1 | I | 42 | static_boundary | 6.00 | 0.290 |
| ppo_release_v0→ppo_release_v1 | I | 42 | center_residual | 1.00 | 0.090 |
| ppo_release_v0→ppo_release_v1 | I | 42 | coordinate_residual | 1.00 | 0.090 |
| ppo_release_v0→ppo_release_v1 | I | 42 | target_only | 2.00 | 0.052 |
| ppo_release_v0→ppo_release_v1 | I | 42 | directed_residual | 1.00 | 0.090 |
| ppo_release_v0→ppo_release_v1 | I | 42 | coordinate_context_ucb_pure | 8.00 | 0.314 |
| ppo_release_v0→ppo_release_v1 | I | 42 | directed_context_ucb_pure | 9.00 | 0.381 |
| ppo_release_v0→ppo_release_v1 | I | 42 | directed_context_ucb_loose | 8.00 | 0.229 |
| ppo_release_v0→ppo_release_v1 | I | 42 | coordinate_offset_calibrated_ucb | 8.00 | 0.224 |
| ppo_release_v0→ppo_release_v1 | I | 42 | directed_offset_calibrated_ucb | 10.00 | 0.562 |
| ppo_release_v0→ppo_release_v1 | I | 42 | coordinate_bootstrap_offset_ucb | 3.00 | 0.029 |
| ppo_release_v0→ppo_release_v1 | I | 42 | directed_bootstrap_offset_ucb | 6.00 | 0.129 |
| ppo_release_v0→ppo_release_v1 | I | 42 | coordinate_regression_bootstrap_ucb | 5.00 | 0.105 |
| ppo_release_v0→ppo_release_v1 | I | 42 | directed_regression_bootstrap_ucb | 8.00 | 0.219 |
| ppo_release_v0→ppo_release_v1 | I | 42 | static_role_coverage2 | 10.00 | 0.500 |
| ppo_release_v0→ppo_release_v1 | I | 42 | coordinate_role_gated | 2.00 | 0.129 |
| ppo_release_v0→ppo_release_v1 | I | 42 | prequential_bma_ucb | 1.00 | 0.067 |
| ppo_release_v1→ppo_release_v2 | R | 10 | random | 0.40 | 0.017 |
| ppo_release_v1→ppo_release_v2 | R | 10 | static_risk | 4.00 | 0.262 |
| ppo_release_v1→ppo_release_v2 | R | 10 | static_boundary | 3.00 | 0.210 |
| ppo_release_v1→ppo_release_v2 | R | 10 | center_residual | 4.00 | 0.300 |
| ppo_release_v1→ppo_release_v2 | R | 10 | coordinate_residual | 4.00 | 0.271 |
| ppo_release_v1→ppo_release_v2 | R | 10 | target_only | 1.00 | 0.029 |
| ppo_release_v1→ppo_release_v2 | R | 10 | directed_residual | 4.00 | 0.257 |
| ppo_release_v1→ppo_release_v2 | R | 10 | coordinate_context_ucb_pure | 3.00 | 0.152 |
| ppo_release_v1→ppo_release_v2 | R | 10 | directed_context_ucb_pure | 8.00 | 0.276 |
| ppo_release_v1→ppo_release_v2 | R | 10 | directed_context_ucb_loose | 5.00 | 0.167 |
| ppo_release_v1→ppo_release_v2 | R | 10 | coordinate_offset_calibrated_ucb | 7.00 | 0.333 |
| ppo_release_v1→ppo_release_v2 | R | 10 | directed_offset_calibrated_ucb | 10.00 | 0.371 |
| ppo_release_v1→ppo_release_v2 | R | 10 | coordinate_bootstrap_offset_ucb | 6.00 | 0.129 |
| ppo_release_v1→ppo_release_v2 | R | 10 | directed_bootstrap_offset_ucb | 5.00 | 0.076 |
| ppo_release_v1→ppo_release_v2 | R | 10 | coordinate_regression_bootstrap_ucb | 7.00 | 0.152 |
| ppo_release_v1→ppo_release_v2 | R | 10 | directed_regression_bootstrap_ucb | 6.00 | 0.119 |
| ppo_release_v1→ppo_release_v2 | R | 10 | static_role_coverage2 | 7.00 | 0.152 |
| ppo_release_v1→ppo_release_v2 | R | 10 | coordinate_role_gated | 4.00 | 0.105 |
| ppo_release_v1→ppo_release_v2 | R | 10 | prequential_bma_ucb | 4.00 | 0.095 |
| ppo_release_v1→ppo_release_v2 | I | 34 | random | 2.20 | 0.127 |
| ppo_release_v1→ppo_release_v2 | I | 34 | static_risk | 3.00 | 0.267 |
| ppo_release_v1→ppo_release_v2 | I | 34 | static_boundary | 2.00 | 0.152 |
| ppo_release_v1→ppo_release_v2 | I | 34 | center_residual | 2.00 | 0.186 |
| ppo_release_v1→ppo_release_v2 | I | 34 | coordinate_residual | 2.00 | 0.186 |
| ppo_release_v1→ppo_release_v2 | I | 34 | target_only | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 34 | directed_residual | 2.00 | 0.186 |
| ppo_release_v1→ppo_release_v2 | I | 34 | coordinate_context_ucb_pure | 5.00 | 0.338 |
| ppo_release_v1→ppo_release_v2 | I | 34 | directed_context_ucb_pure | 6.00 | 0.410 |
| ppo_release_v1→ppo_release_v2 | I | 34 | directed_context_ucb_loose | 6.00 | 0.329 |
| ppo_release_v1→ppo_release_v2 | I | 34 | coordinate_offset_calibrated_ucb | 5.00 | 0.386 |
| ppo_release_v1→ppo_release_v2 | I | 34 | directed_offset_calibrated_ucb | 6.00 | 0.419 |
| ppo_release_v1→ppo_release_v2 | I | 34 | coordinate_bootstrap_offset_ucb | 8.00 | 0.438 |
| ppo_release_v1→ppo_release_v2 | I | 34 | directed_bootstrap_offset_ucb | 8.00 | 0.438 |
| ppo_release_v1→ppo_release_v2 | I | 34 | coordinate_regression_bootstrap_ucb | 5.00 | 0.410 |
| ppo_release_v1→ppo_release_v2 | I | 34 | directed_regression_bootstrap_ucb | 6.00 | 0.414 |
| ppo_release_v1→ppo_release_v2 | I | 34 | static_role_coverage2 | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 34 | coordinate_role_gated | 6.00 | 0.367 |
| ppo_release_v1→ppo_release_v2 | I | 34 | prequential_bma_ucb | 4.00 | 0.333 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
