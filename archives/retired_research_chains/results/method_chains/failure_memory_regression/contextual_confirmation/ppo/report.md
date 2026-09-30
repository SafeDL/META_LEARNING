# PPO 权重继承链 双向版本变化测试

协议：`ppo-contextual-bidirectional-v1`；数据身份：`contextual_confirmation`；1089 个冻结场景，3267 次配对物理执行。
主比较方法：`directed_context_ucb_loose`；所有方法共用固定方向日程与查询预算。
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
| ppo_release_v0→ppo_release_v1 | I | 35 | random | 1.80 | 0.089 |
| ppo_release_v0→ppo_release_v1 | I | 35 | static_risk | 1.00 | 0.076 |
| ppo_release_v0→ppo_release_v1 | I | 35 | static_boundary | 1.00 | 0.081 |
| ppo_release_v0→ppo_release_v1 | I | 35 | center_residual | 1.00 | 0.071 |
| ppo_release_v0→ppo_release_v1 | I | 35 | coordinate_residual | 1.00 | 0.071 |
| ppo_release_v0→ppo_release_v1 | I | 35 | target_only | 2.00 | 0.114 |
| ppo_release_v0→ppo_release_v1 | I | 35 | directed_residual | 1.00 | 0.071 |
| ppo_release_v0→ppo_release_v1 | I | 35 | coordinate_context_ucb_pure | 1.00 | 0.043 |
| ppo_release_v0→ppo_release_v1 | I | 35 | directed_context_ucb_pure | 1.00 | 0.010 |
| ppo_release_v0→ppo_release_v1 | I | 35 | directed_context_ucb_loose | 1.00 | 0.005 |
| ppo_release_v1→ppo_release_v2 | R | 24 | random | 0.90 | 0.054 |
| ppo_release_v1→ppo_release_v2 | R | 24 | static_risk | 2.00 | 0.071 |
| ppo_release_v1→ppo_release_v2 | R | 24 | static_boundary | 4.00 | 0.257 |
| ppo_release_v1→ppo_release_v2 | R | 24 | center_residual | 2.00 | 0.129 |
| ppo_release_v1→ppo_release_v2 | R | 24 | coordinate_residual | 2.00 | 0.086 |
| ppo_release_v1→ppo_release_v2 | R | 24 | target_only | 4.00 | 0.210 |
| ppo_release_v1→ppo_release_v2 | R | 24 | directed_residual | 2.00 | 0.138 |
| ppo_release_v1→ppo_release_v2 | R | 24 | coordinate_context_ucb_pure | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 24 | directed_context_ucb_pure | 1.00 | 0.048 |
| ppo_release_v1→ppo_release_v2 | R | 24 | directed_context_ucb_loose | 2.00 | 0.124 |
| ppo_release_v1→ppo_release_v2 | I | 25 | random | 2.00 | 0.088 |
| ppo_release_v1→ppo_release_v2 | I | 25 | static_risk | 4.00 | 0.257 |
| ppo_release_v1→ppo_release_v2 | I | 25 | static_boundary | 4.00 | 0.276 |
| ppo_release_v1→ppo_release_v2 | I | 25 | center_residual | 4.00 | 0.238 |
| ppo_release_v1→ppo_release_v2 | I | 25 | coordinate_residual | 4.00 | 0.229 |
| ppo_release_v1→ppo_release_v2 | I | 25 | target_only | 1.00 | 0.062 |
| ppo_release_v1→ppo_release_v2 | I | 25 | directed_residual | 5.00 | 0.257 |
| ppo_release_v1→ppo_release_v2 | I | 25 | coordinate_context_ucb_pure | 9.00 | 0.490 |
| ppo_release_v1→ppo_release_v2 | I | 25 | directed_context_ucb_pure | 9.00 | 0.552 |
| ppo_release_v1→ppo_release_v2 | I | 25 | directed_context_ucb_loose | 9.00 | 0.395 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
