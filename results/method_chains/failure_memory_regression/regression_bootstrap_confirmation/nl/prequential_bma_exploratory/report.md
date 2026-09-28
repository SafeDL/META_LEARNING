# NL-IDM 双向版本变化测试

协议：`nl-regression-bootstrap-bidirectional-v1`；数据身份：`regression_bootstrap_confirmation`；1089 个冻结场景，3267 次配对物理执行。
主比较方法：`directed_regression_bootstrap_ucb`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| nl_v0→nl2_v1 | R | 20 | random | 1.00 | 0.046 |
| nl_v0→nl2_v1 | R | 20 | static_risk | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 20 | static_boundary | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 20 | center_residual | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 20 | coordinate_residual | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 20 | target_only | 1.00 | 0.081 |
| nl_v0→nl2_v1 | R | 20 | directed_residual | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 20 | coordinate_context_ucb_pure | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 20 | directed_context_ucb_pure | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 20 | directed_context_ucb_loose | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 20 | coordinate_offset_calibrated_ucb | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 20 | directed_offset_calibrated_ucb | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 20 | coordinate_bootstrap_offset_ucb | 11.00 | 0.376 |
| nl_v0→nl2_v1 | R | 20 | directed_bootstrap_offset_ucb | 12.00 | 0.429 |
| nl_v0→nl2_v1 | R | 20 | coordinate_regression_bootstrap_ucb | 10.00 | 0.329 |
| nl_v0→nl2_v1 | R | 20 | directed_regression_bootstrap_ucb | 12.00 | 0.429 |
| nl_v0→nl2_v1 | R | 20 | static_role_coverage2 | 1.00 | 0.076 |
| nl_v0→nl2_v1 | R | 20 | coordinate_role_gated | 14.00 | 0.552 |
| nl_v0→nl2_v1 | R | 20 | prequential_bma_ucb | 0.00 | 0.000 |
| nl_v0→nl2_v1 | I | 0 | random | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | static_risk | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | static_boundary | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | center_residual | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | coordinate_residual | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | target_only | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_residual | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | coordinate_context_ucb_pure | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_context_ucb_pure | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_context_ucb_loose | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | coordinate_offset_calibrated_ucb | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_offset_calibrated_ucb | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | coordinate_bootstrap_offset_ucb | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_bootstrap_offset_ucb | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | coordinate_regression_bootstrap_ucb | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_regression_bootstrap_ucb | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | static_role_coverage2 | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | coordinate_role_gated | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | prequential_bma_ucb | 0.00 | NA |
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
| nl2_v1→nl2_v2 | R | 0 | prequential_bma_ucb | 0.00 | NA |
| nl2_v1→nl2_v2 | I | 70 | random | 2.70 | 0.150 |
| nl2_v1→nl2_v2 | I | 70 | static_risk | 15.00 | 0.843 |
| nl2_v1→nl2_v2 | I | 70 | static_boundary | 11.00 | 0.624 |
| nl2_v1→nl2_v2 | I | 70 | center_residual | 16.00 | 0.848 |
| nl2_v1→nl2_v2 | I | 70 | coordinate_residual | 14.00 | 0.810 |
| nl2_v1→nl2_v2 | I | 70 | target_only | 1.00 | 0.043 |
| nl2_v1→nl2_v2 | I | 70 | directed_residual | 16.00 | 0.857 |
| nl2_v1→nl2_v2 | I | 70 | coordinate_context_ucb_pure | 16.00 | 0.767 |
| nl2_v1→nl2_v2 | I | 70 | directed_context_ucb_pure | 16.00 | 0.752 |
| nl2_v1→nl2_v2 | I | 70 | directed_context_ucb_loose | 17.00 | 0.824 |
| nl2_v1→nl2_v2 | I | 70 | coordinate_offset_calibrated_ucb | 17.00 | 0.800 |
| nl2_v1→nl2_v2 | I | 70 | directed_offset_calibrated_ucb | 17.00 | 0.824 |
| nl2_v1→nl2_v2 | I | 70 | coordinate_bootstrap_offset_ucb | 13.00 | 0.652 |
| nl2_v1→nl2_v2 | I | 70 | directed_bootstrap_offset_ucb | 14.00 | 0.681 |
| nl2_v1→nl2_v2 | I | 70 | coordinate_regression_bootstrap_ucb | 15.00 | 0.700 |
| nl2_v1→nl2_v2 | I | 70 | directed_regression_bootstrap_ucb | 16.00 | 0.771 |
| nl2_v1→nl2_v2 | I | 70 | static_role_coverage2 | 0.00 | 0.000 |
| nl2_v1→nl2_v2 | I | 70 | coordinate_role_gated | 14.00 | 0.671 |
| nl2_v1→nl2_v2 | I | 70 | prequential_bma_ucb | 16.00 | 0.810 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
