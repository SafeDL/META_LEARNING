# NL-IDM 双向版本变化测试

协议：`nl-contextual-bidirectional-v1`；数据身份：`contextual_confirmation`；1089 个冻结场景，3267 次配对物理执行。
主比较方法：`directed_context_ucb_loose`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| nl_v0→nl2_v1 | R | 17 | random | 0.80 | 0.050 |
| nl_v0→nl2_v1 | R | 17 | static_risk | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 17 | static_boundary | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 17 | center_residual | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 17 | coordinate_residual | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 17 | target_only | 2.00 | 0.157 |
| nl_v0→nl2_v1 | R | 17 | directed_residual | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 17 | coordinate_context_ucb_pure | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 17 | directed_context_ucb_pure | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 17 | directed_context_ucb_loose | 0.00 | 0.000 |
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
| nl2_v1→nl2_v2 | I | 33 | random | 1.70 | 0.085 |
| nl2_v1→nl2_v2 | I | 33 | static_risk | 8.00 | 0.590 |
| nl2_v1→nl2_v2 | I | 33 | static_boundary | 10.00 | 0.567 |
| nl2_v1→nl2_v2 | I | 33 | center_residual | 11.00 | 0.633 |
| nl2_v1→nl2_v2 | I | 33 | coordinate_residual | 11.00 | 0.590 |
| nl2_v1→nl2_v2 | I | 33 | target_only | 0.00 | 0.000 |
| nl2_v1→nl2_v2 | I | 33 | directed_residual | 11.00 | 0.605 |
| nl2_v1→nl2_v2 | I | 33 | coordinate_context_ucb_pure | 19.00 | 0.995 |
| nl2_v1→nl2_v2 | I | 33 | directed_context_ucb_pure | 19.00 | 0.995 |
| nl2_v1→nl2_v2 | I | 33 | directed_context_ucb_loose | 19.00 | 0.995 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
