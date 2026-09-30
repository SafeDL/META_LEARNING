# NL-IDM 双向版本变化测试

协议：`nl-context-bootstrap-bidirectional-v1`；数据身份：`context_bootstrap_confirmation`；1089 个冻结场景，3267 次配对物理执行。
主比较方法：`directed_bootstrap_offset_ucb`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| nl_v0→nl2_v1 | R | 47 | random | 1.40 | 0.070 |
| nl_v0→nl2_v1 | R | 47 | static_risk | 4.00 | 0.086 |
| nl_v0→nl2_v1 | R | 47 | static_boundary | 4.00 | 0.276 |
| nl_v0→nl2_v1 | R | 47 | center_residual | 8.00 | 0.248 |
| nl_v0→nl2_v1 | R | 47 | coordinate_residual | 7.00 | 0.233 |
| nl_v0→nl2_v1 | R | 47 | target_only | 1.00 | 0.090 |
| nl_v0→nl2_v1 | R | 47 | directed_residual | 6.00 | 0.219 |
| nl_v0→nl2_v1 | R | 47 | coordinate_context_ucb_pure | 11.00 | 0.671 |
| nl_v0→nl2_v1 | R | 47 | directed_context_ucb_pure | 11.00 | 0.733 |
| nl_v0→nl2_v1 | R | 47 | directed_context_ucb_loose | 11.00 | 0.733 |
| nl_v0→nl2_v1 | R | 47 | coordinate_offset_calibrated_ucb | 11.00 | 0.538 |
| nl_v0→nl2_v1 | R | 47 | directed_offset_calibrated_ucb | 11.00 | 0.733 |
| nl_v0→nl2_v1 | R | 47 | coordinate_bootstrap_offset_ucb | 16.00 | 0.648 |
| nl_v0→nl2_v1 | R | 47 | directed_bootstrap_offset_ucb | 16.00 | 0.648 |
| nl_v0→nl2_v1 | R | 47 | static_role_coverage2 | 12.00 | 0.600 |
| nl_v0→nl2_v1 | R | 47 | coordinate_role_gated | 12.00 | 0.600 |
| nl_v0→nl2_v1 | I | 2 | random | 0.10 | 0.001 |
| nl_v0→nl2_v1 | I | 2 | static_risk | 2.00 | 0.171 |
| nl_v0→nl2_v1 | I | 2 | static_boundary | 1.00 | 0.095 |
| nl_v0→nl2_v1 | I | 2 | center_residual | 2.00 | 0.143 |
| nl_v0→nl2_v1 | I | 2 | coordinate_residual | 2.00 | 0.176 |
| nl_v0→nl2_v1 | I | 2 | target_only | 1.00 | 0.005 |
| nl_v0→nl2_v1 | I | 2 | directed_residual | 2.00 | 0.152 |
| nl_v0→nl2_v1 | I | 2 | coordinate_context_ucb_pure | 2.00 | 0.148 |
| nl_v0→nl2_v1 | I | 2 | directed_context_ucb_pure | 2.00 | 0.157 |
| nl_v0→nl2_v1 | I | 2 | directed_context_ucb_loose | 2.00 | 0.138 |
| nl_v0→nl2_v1 | I | 2 | coordinate_offset_calibrated_ucb | 2.00 | 0.176 |
| nl_v0→nl2_v1 | I | 2 | directed_offset_calibrated_ucb | 2.00 | 0.157 |
| nl_v0→nl2_v1 | I | 2 | coordinate_bootstrap_offset_ucb | 2.00 | 0.148 |
| nl_v0→nl2_v1 | I | 2 | directed_bootstrap_offset_ucb | 2.00 | 0.133 |
| nl_v0→nl2_v1 | I | 2 | static_role_coverage2 | 0.00 | 0.000 |
| nl_v0→nl2_v1 | I | 2 | coordinate_role_gated | 2.00 | 0.176 |
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
| nl2_v1→nl2_v2 | R | 0 | static_role_coverage2 | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_role_gated | 0.00 | NA |
| nl2_v1→nl2_v2 | I | 73 | random | 2.80 | 0.151 |
| nl2_v1→nl2_v2 | I | 73 | static_risk | 16.00 | 0.914 |
| nl2_v1→nl2_v2 | I | 73 | static_boundary | 15.00 | 0.857 |
| nl2_v1→nl2_v2 | I | 73 | center_residual | 15.00 | 0.814 |
| nl2_v1→nl2_v2 | I | 73 | coordinate_residual | 15.00 | 0.800 |
| nl2_v1→nl2_v2 | I | 73 | target_only | 0.00 | 0.000 |
| nl2_v1→nl2_v2 | I | 73 | directed_residual | 15.00 | 0.833 |
| nl2_v1→nl2_v2 | I | 73 | coordinate_context_ucb_pure | 20.00 | 1.000 |
| nl2_v1→nl2_v2 | I | 73 | directed_context_ucb_pure | 20.00 | 1.000 |
| nl2_v1→nl2_v2 | I | 73 | directed_context_ucb_loose | 20.00 | 1.000 |
| nl2_v1→nl2_v2 | I | 73 | coordinate_offset_calibrated_ucb | 20.00 | 1.000 |
| nl2_v1→nl2_v2 | I | 73 | directed_offset_calibrated_ucb | 20.00 | 1.000 |
| nl2_v1→nl2_v2 | I | 73 | coordinate_bootstrap_offset_ucb | 15.00 | 0.610 |
| nl2_v1→nl2_v2 | I | 73 | directed_bootstrap_offset_ucb | 15.00 | 0.610 |
| nl2_v1→nl2_v2 | I | 73 | static_role_coverage2 | 8.00 | 0.614 |
| nl2_v1→nl2_v2 | I | 73 | coordinate_role_gated | 20.00 | 1.000 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
