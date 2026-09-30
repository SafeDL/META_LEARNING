# PPO 权重继承链 双向版本变化测试

协议：`ppo-context-bootstrap-bidirectional-v1`；数据身份：`context_bootstrap_confirmation`；1089 个冻结场景，3267 次配对物理执行。
主比较方法：`directed_bootstrap_offset_ucb`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| ppo_release_v0→ppo_release_v1 | R | 10 | random | 0.50 | 0.028 |
| ppo_release_v0→ppo_release_v1 | R | 10 | static_risk | 1.00 | 0.029 |
| ppo_release_v0→ppo_release_v1 | R | 10 | static_boundary | 1.00 | 0.086 |
| ppo_release_v0→ppo_release_v1 | R | 10 | center_residual | 1.00 | 0.043 |
| ppo_release_v0→ppo_release_v1 | R | 10 | coordinate_residual | 1.00 | 0.043 |
| ppo_release_v0→ppo_release_v1 | R | 10 | target_only | 1.00 | 0.057 |
| ppo_release_v0→ppo_release_v1 | R | 10 | directed_residual | 1.00 | 0.057 |
| ppo_release_v0→ppo_release_v1 | R | 10 | coordinate_context_ucb_pure | 1.00 | 0.062 |
| ppo_release_v0→ppo_release_v1 | R | 10 | directed_context_ucb_pure | 1.00 | 0.086 |
| ppo_release_v0→ppo_release_v1 | R | 10 | directed_context_ucb_loose | 1.00 | 0.090 |
| ppo_release_v0→ppo_release_v1 | R | 10 | coordinate_offset_calibrated_ucb | 1.00 | 0.048 |
| ppo_release_v0→ppo_release_v1 | R | 10 | directed_offset_calibrated_ucb | 1.00 | 0.086 |
| ppo_release_v0→ppo_release_v1 | R | 10 | coordinate_bootstrap_offset_ucb | 10.00 | 0.400 |
| ppo_release_v0→ppo_release_v1 | R | 10 | directed_bootstrap_offset_ucb | 9.00 | 0.367 |
| ppo_release_v0→ppo_release_v1 | R | 10 | static_role_coverage2 | 7.00 | 0.371 |
| ppo_release_v0→ppo_release_v1 | R | 10 | coordinate_role_gated | 2.00 | 0.181 |
| ppo_release_v0→ppo_release_v1 | I | 29 | random | 1.50 | 0.069 |
| ppo_release_v0→ppo_release_v1 | I | 29 | static_risk | 1.00 | 0.019 |
| ppo_release_v0→ppo_release_v1 | I | 29 | static_boundary | 1.00 | 0.081 |
| ppo_release_v0→ppo_release_v1 | I | 29 | center_residual | 3.00 | 0.171 |
| ppo_release_v0→ppo_release_v1 | I | 29 | coordinate_residual | 3.00 | 0.148 |
| ppo_release_v0→ppo_release_v1 | I | 29 | target_only | 2.00 | 0.057 |
| ppo_release_v0→ppo_release_v1 | I | 29 | directed_residual | 4.00 | 0.186 |
| ppo_release_v0→ppo_release_v1 | I | 29 | coordinate_context_ucb_pure | 6.00 | 0.238 |
| ppo_release_v0→ppo_release_v1 | I | 29 | directed_context_ucb_pure | 6.00 | 0.433 |
| ppo_release_v0→ppo_release_v1 | I | 29 | directed_context_ucb_loose | 6.00 | 0.443 |
| ppo_release_v0→ppo_release_v1 | I | 29 | coordinate_offset_calibrated_ucb | 8.00 | 0.348 |
| ppo_release_v0→ppo_release_v1 | I | 29 | directed_offset_calibrated_ucb | 6.00 | 0.386 |
| ppo_release_v0→ppo_release_v1 | I | 29 | coordinate_bootstrap_offset_ucb | 5.00 | 0.238 |
| ppo_release_v0→ppo_release_v1 | I | 29 | directed_bootstrap_offset_ucb | 6.00 | 0.281 |
| ppo_release_v0→ppo_release_v1 | I | 29 | static_role_coverage2 | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 29 | coordinate_role_gated | 6.00 | 0.195 |
| ppo_release_v1→ppo_release_v2 | R | 17 | random | 0.40 | 0.016 |
| ppo_release_v1→ppo_release_v2 | R | 17 | static_risk | 5.00 | 0.319 |
| ppo_release_v1→ppo_release_v2 | R | 17 | static_boundary | 2.00 | 0.129 |
| ppo_release_v1→ppo_release_v2 | R | 17 | center_residual | 5.00 | 0.352 |
| ppo_release_v1→ppo_release_v2 | R | 17 | coordinate_residual | 5.00 | 0.348 |
| ppo_release_v1→ppo_release_v2 | R | 17 | target_only | 3.00 | 0.176 |
| ppo_release_v1→ppo_release_v2 | R | 17 | directed_residual | 5.00 | 0.352 |
| ppo_release_v1→ppo_release_v2 | R | 17 | coordinate_context_ucb_pure | 5.00 | 0.257 |
| ppo_release_v1→ppo_release_v2 | R | 17 | directed_context_ucb_pure | 5.00 | 0.229 |
| ppo_release_v1→ppo_release_v2 | R | 17 | directed_context_ucb_loose | 3.00 | 0.195 |
| ppo_release_v1→ppo_release_v2 | R | 17 | coordinate_offset_calibrated_ucb | 3.00 | 0.224 |
| ppo_release_v1→ppo_release_v2 | R | 17 | directed_offset_calibrated_ucb | 3.00 | 0.210 |
| ppo_release_v1→ppo_release_v2 | R | 17 | coordinate_bootstrap_offset_ucb | 10.00 | 0.343 |
| ppo_release_v1→ppo_release_v2 | R | 17 | directed_bootstrap_offset_ucb | 10.00 | 0.343 |
| ppo_release_v1→ppo_release_v2 | R | 17 | static_role_coverage2 | 4.00 | 0.214 |
| ppo_release_v1→ppo_release_v2 | R | 17 | coordinate_role_gated | 12.00 | 0.424 |
| ppo_release_v1→ppo_release_v2 | I | 47 | random | 2.50 | 0.142 |
| ppo_release_v1→ppo_release_v2 | I | 47 | static_risk | 1.00 | 0.048 |
| ppo_release_v1→ppo_release_v2 | I | 47 | static_boundary | 1.00 | 0.062 |
| ppo_release_v1→ppo_release_v2 | I | 47 | center_residual | 14.00 | 0.690 |
| ppo_release_v1→ppo_release_v2 | I | 47 | coordinate_residual | 15.00 | 0.705 |
| ppo_release_v1→ppo_release_v2 | I | 47 | target_only | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 47 | directed_residual | 14.00 | 0.619 |
| ppo_release_v1→ppo_release_v2 | I | 47 | coordinate_context_ucb_pure | 15.00 | 0.686 |
| ppo_release_v1→ppo_release_v2 | I | 47 | directed_context_ucb_pure | 15.00 | 0.738 |
| ppo_release_v1→ppo_release_v2 | I | 47 | directed_context_ucb_loose | 18.00 | 0.905 |
| ppo_release_v1→ppo_release_v2 | I | 47 | coordinate_offset_calibrated_ucb | 16.00 | 0.714 |
| ppo_release_v1→ppo_release_v2 | I | 47 | directed_offset_calibrated_ucb | 17.00 | 0.767 |
| ppo_release_v1→ppo_release_v2 | I | 47 | coordinate_bootstrap_offset_ucb | 10.00 | 0.362 |
| ppo_release_v1→ppo_release_v2 | I | 47 | directed_bootstrap_offset_ucb | 10.00 | 0.362 |
| ppo_release_v1→ppo_release_v2 | I | 47 | static_role_coverage2 | 2.00 | 0.167 |
| ppo_release_v1→ppo_release_v2 | I | 47 | coordinate_role_gated | 15.00 | 0.614 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
