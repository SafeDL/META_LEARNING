# NL-IDM 双向版本变化测试

协议：`nl-offset-calibration-bidirectional-v1`；数据身份：`offset_calibration_confirmation`；1089 个冻结场景，3267 次配对物理执行。
主比较方法：`directed_offset_calibrated_ucb`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| nl_v0→nl2_v1 | R | 51 | random | 1.60 | 0.078 |
| nl_v0→nl2_v1 | R | 51 | static_risk | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 51 | static_boundary | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 51 | center_residual | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 51 | coordinate_residual | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 51 | target_only | 3.00 | 0.143 |
| nl_v0→nl2_v1 | R | 51 | directed_residual | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 51 | coordinate_context_ucb_pure | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 51 | directed_context_ucb_pure | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 51 | directed_context_ucb_loose | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 51 | coordinate_offset_calibrated_ucb | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 51 | directed_offset_calibrated_ucb | 0.00 | 0.000 |
| nl_v0→nl2_v1 | I | 2 | random | 0.20 | 0.005 |
| nl_v0→nl2_v1 | I | 2 | static_risk | 2.00 | 0.148 |
| nl_v0→nl2_v1 | I | 2 | static_boundary | 0.00 | 0.000 |
| nl_v0→nl2_v1 | I | 2 | center_residual | 2.00 | 0.129 |
| nl_v0→nl2_v1 | I | 2 | coordinate_residual | 2.00 | 0.143 |
| nl_v0→nl2_v1 | I | 2 | target_only | 0.00 | 0.000 |
| nl_v0→nl2_v1 | I | 2 | directed_residual | 2.00 | 0.124 |
| nl_v0→nl2_v1 | I | 2 | coordinate_context_ucb_pure | 2.00 | 0.176 |
| nl_v0→nl2_v1 | I | 2 | directed_context_ucb_pure | 2.00 | 0.176 |
| nl_v0→nl2_v1 | I | 2 | directed_context_ucb_loose | 2.00 | 0.167 |
| nl_v0→nl2_v1 | I | 2 | coordinate_offset_calibrated_ucb | 2.00 | 0.176 |
| nl_v0→nl2_v1 | I | 2 | directed_offset_calibrated_ucb | 2.00 | 0.176 |
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
| nl2_v1→nl2_v2 | I | 83 | random | 3.40 | 0.164 |
| nl2_v1→nl2_v2 | I | 83 | static_risk | 20.00 | 1.000 |
| nl2_v1→nl2_v2 | I | 83 | static_boundary | 14.00 | 0.738 |
| nl2_v1→nl2_v2 | I | 83 | center_residual | 19.00 | 0.986 |
| nl2_v1→nl2_v2 | I | 83 | coordinate_residual | 20.00 | 1.000 |
| nl2_v1→nl2_v2 | I | 83 | target_only | 1.00 | 0.071 |
| nl2_v1→nl2_v2 | I | 83 | directed_residual | 18.00 | 0.986 |
| nl2_v1→nl2_v2 | I | 83 | coordinate_context_ucb_pure | 18.00 | 0.890 |
| nl2_v1→nl2_v2 | I | 83 | directed_context_ucb_pure | 18.00 | 0.900 |
| nl2_v1→nl2_v2 | I | 83 | directed_context_ucb_loose | 17.00 | 0.895 |
| nl2_v1→nl2_v2 | I | 83 | coordinate_offset_calibrated_ucb | 18.00 | 0.900 |
| nl2_v1→nl2_v2 | I | 83 | directed_offset_calibrated_ucb | 18.00 | 0.905 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
