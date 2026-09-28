# NL-IDM 双向版本变化测试

协议：`nl-bidirectional-v1`；数据身份：`ppo_confirmation`；225 个冻结场景，675 次配对物理执行。
主比较方法：`directed_residual`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| ppo_release_v0→ppo_release_v1 | R | 5 | random | 0.80 | 0.025 |
| ppo_release_v0→ppo_release_v1 | R | 5 | static_risk | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | R | 5 | static_boundary | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | R | 5 | center_residual | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | R | 5 | coordinate_residual | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | R | 5 | target_only | 2.00 | 0.067 |
| ppo_release_v0→ppo_release_v1 | R | 5 | directed_residual | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | R | 5 | directed_context_laplace | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | R | 5 | coordinate_margin_frontier | 5.00 | 0.252 |
| ppo_release_v0→ppo_release_v1 | R | 5 | directed_margin_frontier | 5.00 | 0.281 |
| ppo_release_v0→ppo_release_v1 | I | 10 | random | 1.50 | 0.078 |
| ppo_release_v0→ppo_release_v1 | I | 10 | static_risk | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 10 | static_boundary | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 10 | center_residual | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 10 | coordinate_residual | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 10 | target_only | 3.00 | 0.200 |
| ppo_release_v0→ppo_release_v1 | I | 10 | directed_residual | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 10 | directed_context_laplace | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 10 | coordinate_margin_frontier | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 10 | directed_margin_frontier | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 5 | random | 0.50 | 0.018 |
| ppo_release_v1→ppo_release_v2 | R | 5 | static_risk | 5.00 | 0.181 |
| ppo_release_v1→ppo_release_v2 | R | 5 | static_boundary | 4.00 | 0.200 |
| ppo_release_v1→ppo_release_v2 | R | 5 | center_residual | 5.00 | 0.176 |
| ppo_release_v1→ppo_release_v2 | R | 5 | coordinate_residual | 5.00 | 0.186 |
| ppo_release_v1→ppo_release_v2 | R | 5 | target_only | 3.00 | 0.157 |
| ppo_release_v1→ppo_release_v2 | R | 5 | directed_residual | 5.00 | 0.190 |
| ppo_release_v1→ppo_release_v2 | R | 5 | directed_context_laplace | 5.00 | 0.210 |
| ppo_release_v1→ppo_release_v2 | R | 5 | coordinate_margin_frontier | 5.00 | 0.262 |
| ppo_release_v1→ppo_release_v2 | R | 5 | directed_margin_frontier | 5.00 | 0.262 |
| ppo_release_v1→ppo_release_v2 | I | 12 | random | 3.20 | 0.135 |
| ppo_release_v1→ppo_release_v2 | I | 12 | static_risk | 1.00 | 0.005 |
| ppo_release_v1→ppo_release_v2 | I | 12 | static_boundary | 5.00 | 0.205 |
| ppo_release_v1→ppo_release_v2 | I | 12 | center_residual | 3.00 | 0.095 |
| ppo_release_v1→ppo_release_v2 | I | 12 | coordinate_residual | 3.00 | 0.090 |
| ppo_release_v1→ppo_release_v2 | I | 12 | target_only | 1.00 | 0.052 |
| ppo_release_v1→ppo_release_v2 | I | 12 | directed_residual | 4.00 | 0.076 |
| ppo_release_v1→ppo_release_v2 | I | 12 | directed_context_laplace | 1.00 | 0.024 |
| ppo_release_v1→ppo_release_v2 | I | 12 | coordinate_margin_frontier | 2.00 | 0.048 |
| ppo_release_v1→ppo_release_v2 | I | 12 | directed_margin_frontier | 2.00 | 0.052 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
