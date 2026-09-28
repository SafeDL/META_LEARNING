# PPO 权重继承链 双向版本变化测试

协议：`ppo-superiority-replication-bidirectional-v1`；数据身份：`superiority_replication_confirmation`；1089 个冻结场景，3267 次配对物理执行。
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
| ppo_release_v0→ppo_release_v1 | I | 29 | random | 1.50 | 0.066 |
| ppo_release_v0→ppo_release_v1 | I | 29 | static_risk | 2.00 | 0.090 |
| ppo_release_v0→ppo_release_v1 | I | 29 | static_boundary | 4.00 | 0.262 |
| ppo_release_v0→ppo_release_v1 | I | 29 | center_residual | 3.00 | 0.171 |
| ppo_release_v0→ppo_release_v1 | I | 29 | coordinate_residual | 4.00 | 0.195 |
| ppo_release_v0→ppo_release_v1 | I | 29 | target_only | 3.00 | 0.100 |
| ppo_release_v0→ppo_release_v1 | I | 29 | directed_residual | 3.00 | 0.186 |
| ppo_release_v0→ppo_release_v1 | I | 29 | coordinate_context_ucb_pure | 3.00 | 0.243 |
| ppo_release_v0→ppo_release_v1 | I | 29 | directed_context_ucb_pure | 3.00 | 0.243 |
| ppo_release_v0→ppo_release_v1 | I | 29 | directed_context_ucb_loose | 3.00 | 0.243 |
| ppo_release_v0→ppo_release_v1 | I | 29 | coordinate_offset_calibrated_ucb | 3.00 | 0.243 |
| ppo_release_v0→ppo_release_v1 | I | 29 | directed_offset_calibrated_ucb | 3.00 | 0.243 |
| ppo_release_v0→ppo_release_v1 | I | 29 | coordinate_bootstrap_offset_ucb | 3.00 | 0.143 |
| ppo_release_v0→ppo_release_v1 | I | 29 | directed_bootstrap_offset_ucb | 3.00 | 0.143 |
| ppo_release_v0→ppo_release_v1 | I | 29 | coordinate_regression_bootstrap_ucb | 3.00 | 0.143 |
| ppo_release_v0→ppo_release_v1 | I | 29 | directed_regression_bootstrap_ucb | 3.00 | 0.143 |
| ppo_release_v0→ppo_release_v1 | I | 29 | static_role_coverage2 | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 29 | coordinate_role_gated | 5.00 | 0.152 |
| ppo_release_v1→ppo_release_v2 | R | 7 | random | 0.30 | 0.014 |
| ppo_release_v1→ppo_release_v2 | R | 7 | static_risk | 2.00 | 0.086 |
| ppo_release_v1→ppo_release_v2 | R | 7 | static_boundary | 3.00 | 0.152 |
| ppo_release_v1→ppo_release_v2 | R | 7 | center_residual | 3.00 | 0.057 |
| ppo_release_v1→ppo_release_v2 | R | 7 | coordinate_residual | 3.00 | 0.057 |
| ppo_release_v1→ppo_release_v2 | R | 7 | target_only | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 7 | directed_residual | 3.00 | 0.052 |
| ppo_release_v1→ppo_release_v2 | R | 7 | coordinate_context_ucb_pure | 2.00 | 0.048 |
| ppo_release_v1→ppo_release_v2 | R | 7 | directed_context_ucb_pure | 5.00 | 0.205 |
| ppo_release_v1→ppo_release_v2 | R | 7 | directed_context_ucb_loose | 4.00 | 0.167 |
| ppo_release_v1→ppo_release_v2 | R | 7 | coordinate_offset_calibrated_ucb | 4.00 | 0.200 |
| ppo_release_v1→ppo_release_v2 | R | 7 | directed_offset_calibrated_ucb | 6.00 | 0.181 |
| ppo_release_v1→ppo_release_v2 | R | 7 | coordinate_bootstrap_offset_ucb | 1.00 | 0.024 |
| ppo_release_v1→ppo_release_v2 | R | 7 | directed_bootstrap_offset_ucb | 3.00 | 0.086 |
| ppo_release_v1→ppo_release_v2 | R | 7 | coordinate_regression_bootstrap_ucb | 3.00 | 0.043 |
| ppo_release_v1→ppo_release_v2 | R | 7 | directed_regression_bootstrap_ucb | 5.00 | 0.100 |
| ppo_release_v1→ppo_release_v2 | R | 7 | static_role_coverage2 | 3.00 | 0.071 |
| ppo_release_v1→ppo_release_v2 | R | 7 | coordinate_role_gated | 3.00 | 0.029 |
| ppo_release_v1→ppo_release_v2 | I | 33 | random | 2.10 | 0.110 |
| ppo_release_v1→ppo_release_v2 | I | 33 | static_risk | 3.00 | 0.271 |
| ppo_release_v1→ppo_release_v2 | I | 33 | static_boundary | 3.00 | 0.181 |
| ppo_release_v1→ppo_release_v2 | I | 33 | center_residual | 3.00 | 0.271 |
| ppo_release_v1→ppo_release_v2 | I | 33 | coordinate_residual | 3.00 | 0.271 |
| ppo_release_v1→ppo_release_v2 | I | 33 | target_only | 2.00 | 0.090 |
| ppo_release_v1→ppo_release_v2 | I | 33 | directed_residual | 3.00 | 0.271 |
| ppo_release_v1→ppo_release_v2 | I | 33 | coordinate_context_ucb_pure | 3.00 | 0.271 |
| ppo_release_v1→ppo_release_v2 | I | 33 | directed_context_ucb_pure | 3.00 | 0.271 |
| ppo_release_v1→ppo_release_v2 | I | 33 | directed_context_ucb_loose | 6.00 | 0.295 |
| ppo_release_v1→ppo_release_v2 | I | 33 | coordinate_offset_calibrated_ucb | 3.00 | 0.271 |
| ppo_release_v1→ppo_release_v2 | I | 33 | directed_offset_calibrated_ucb | 7.00 | 0.390 |
| ppo_release_v1→ppo_release_v2 | I | 33 | coordinate_bootstrap_offset_ucb | 3.00 | 0.229 |
| ppo_release_v1→ppo_release_v2 | I | 33 | directed_bootstrap_offset_ucb | 5.00 | 0.252 |
| ppo_release_v1→ppo_release_v2 | I | 33 | coordinate_regression_bootstrap_ucb | 3.00 | 0.271 |
| ppo_release_v1→ppo_release_v2 | I | 33 | directed_regression_bootstrap_ucb | 4.00 | 0.276 |
| ppo_release_v1→ppo_release_v2 | I | 33 | static_role_coverage2 | 1.00 | 0.081 |
| ppo_release_v1→ppo_release_v2 | I | 33 | coordinate_role_gated | 8.00 | 0.457 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
