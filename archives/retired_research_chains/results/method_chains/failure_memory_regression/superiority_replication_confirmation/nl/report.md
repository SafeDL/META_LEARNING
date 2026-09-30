# NL-IDM 双向版本变化测试

协议：`nl-superiority-replication-bidirectional-v1`；数据身份：`superiority_replication_confirmation`；1089 个冻结场景，3267 次配对物理执行。
主比较方法：`directed_regression_bootstrap_ucb`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| nl_v0→nl2_v1 | R | 26 | random | 1.30 | 0.061 |
| nl_v0→nl2_v1 | R | 26 | static_risk | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 26 | static_boundary | 1.00 | 0.095 |
| nl_v0→nl2_v1 | R | 26 | center_residual | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 26 | coordinate_residual | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 26 | target_only | 1.00 | 0.067 |
| nl_v0→nl2_v1 | R | 26 | directed_residual | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 26 | coordinate_context_ucb_pure | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 26 | directed_context_ucb_pure | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 26 | directed_context_ucb_loose | 1.00 | 0.033 |
| nl_v0→nl2_v1 | R | 26 | coordinate_offset_calibrated_ucb | 1.00 | 0.010 |
| nl_v0→nl2_v1 | R | 26 | directed_offset_calibrated_ucb | 1.00 | 0.029 |
| nl_v0→nl2_v1 | R | 26 | coordinate_bootstrap_offset_ucb | 14.00 | 0.576 |
| nl_v0→nl2_v1 | R | 26 | directed_bootstrap_offset_ucb | 13.00 | 0.529 |
| nl_v0→nl2_v1 | R | 26 | coordinate_regression_bootstrap_ucb | 14.00 | 0.576 |
| nl_v0→nl2_v1 | R | 26 | directed_regression_bootstrap_ucb | 12.00 | 0.481 |
| nl_v0→nl2_v1 | R | 26 | static_role_coverage2 | 5.00 | 0.352 |
| nl_v0→nl2_v1 | R | 26 | coordinate_role_gated | 16.00 | 0.667 |
| nl_v0→nl2_v1 | I | 2 | random | 0.00 | 0.000 |
| nl_v0→nl2_v1 | I | 2 | static_risk | 2.00 | 0.167 |
| nl_v0→nl2_v1 | I | 2 | static_boundary | 0.00 | 0.000 |
| nl_v0→nl2_v1 | I | 2 | center_residual | 2.00 | 0.086 |
| nl_v0→nl2_v1 | I | 2 | coordinate_residual | 2.00 | 0.143 |
| nl_v0→nl2_v1 | I | 2 | target_only | 0.00 | 0.000 |
| nl_v0→nl2_v1 | I | 2 | directed_residual | 2.00 | 0.076 |
| nl_v0→nl2_v1 | I | 2 | coordinate_context_ucb_pure | 2.00 | 0.143 |
| nl_v0→nl2_v1 | I | 2 | directed_context_ucb_pure | 2.00 | 0.133 |
| nl_v0→nl2_v1 | I | 2 | directed_context_ucb_loose | 2.00 | 0.157 |
| nl_v0→nl2_v1 | I | 2 | coordinate_offset_calibrated_ucb | 2.00 | 0.148 |
| nl_v0→nl2_v1 | I | 2 | directed_offset_calibrated_ucb | 2.00 | 0.119 |
| nl_v0→nl2_v1 | I | 2 | coordinate_bootstrap_offset_ucb | 2.00 | 0.167 |
| nl_v0→nl2_v1 | I | 2 | directed_bootstrap_offset_ucb | 2.00 | 0.167 |
| nl_v0→nl2_v1 | I | 2 | coordinate_regression_bootstrap_ucb | 2.00 | 0.167 |
| nl_v0→nl2_v1 | I | 2 | directed_regression_bootstrap_ucb | 2.00 | 0.167 |
| nl_v0→nl2_v1 | I | 2 | static_role_coverage2 | 0.00 | 0.000 |
| nl_v0→nl2_v1 | I | 2 | coordinate_role_gated | 2.00 | 0.162 |
| nl2_v1→nl2_v2 | R | 0 | random | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | static_risk | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | static_boundary | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | center_residual | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_residual | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | target_only | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_residual | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_context_ucb_pure | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_context_ucb_pure | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_context_ucb_loose | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_offset_calibrated_ucb | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_offset_calibrated_ucb | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_bootstrap_offset_ucb | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_bootstrap_offset_ucb | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_regression_bootstrap_ucb | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_regression_bootstrap_ucb | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | static_role_coverage2 | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_role_gated | 0.00 | NA |
| nl2_v1→nl2_v2 | I | 56 | random | 2.50 | 0.130 |
| nl2_v1→nl2_v2 | I | 56 | static_risk | 19.00 | 0.971 |
| nl2_v1→nl2_v2 | I | 56 | static_boundary | 7.00 | 0.395 |
| nl2_v1→nl2_v2 | I | 56 | center_residual | 14.00 | 0.805 |
| nl2_v1→nl2_v2 | I | 56 | coordinate_residual | 13.00 | 0.805 |
| nl2_v1→nl2_v2 | I | 56 | target_only | 2.00 | 0.110 |
| nl2_v1→nl2_v2 | I | 56 | directed_residual | 13.00 | 0.776 |
| nl2_v1→nl2_v2 | I | 56 | coordinate_context_ucb_pure | 19.00 | 0.971 |
| nl2_v1→nl2_v2 | I | 56 | directed_context_ucb_pure | 18.00 | 0.952 |
| nl2_v1→nl2_v2 | I | 56 | directed_context_ucb_loose | 17.00 | 0.824 |
| nl2_v1→nl2_v2 | I | 56 | coordinate_offset_calibrated_ucb | 18.00 | 0.919 |
| nl2_v1→nl2_v2 | I | 56 | directed_offset_calibrated_ucb | 19.00 | 0.957 |
| nl2_v1→nl2_v2 | I | 56 | coordinate_bootstrap_offset_ucb | 15.00 | 0.686 |
| nl2_v1→nl2_v2 | I | 56 | directed_bootstrap_offset_ucb | 15.00 | 0.729 |
| nl2_v1→nl2_v2 | I | 56 | coordinate_regression_bootstrap_ucb | 19.00 | 0.962 |
| nl2_v1→nl2_v2 | I | 56 | directed_regression_bootstrap_ucb | 19.00 | 0.976 |
| nl2_v1→nl2_v2 | I | 56 | static_role_coverage2 | 4.00 | 0.348 |
| nl2_v1→nl2_v2 | I | 56 | coordinate_role_gated | 20.00 | 1.000 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
