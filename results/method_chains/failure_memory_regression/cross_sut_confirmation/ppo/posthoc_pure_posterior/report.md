# NL-IDM 双向版本变化测试

协议：`nl-bidirectional-v1`；数据身份：`cross_sut_confirmation`；225 个冻结场景，675 次配对物理执行。
主比较方法：`directed_residual`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| ppo_release_v0→ppo_release_v1 | R | 15 | random | 2.80 | 0.131 |
| ppo_release_v0→ppo_release_v1 | R | 15 | static_risk | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | R | 15 | static_boundary | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | R | 15 | center_residual | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | R | 15 | coordinate_residual | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | R | 15 | target_only | 5.00 | 0.233 |
| ppo_release_v0→ppo_release_v1 | R | 15 | directed_residual | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | R | 15 | directed_context_laplace | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | R | 15 | coordinate_margin_frontier | 7.00 | 0.371 |
| ppo_release_v0→ppo_release_v1 | R | 15 | directed_margin_frontier | 7.00 | 0.390 |
| ppo_release_v0→ppo_release_v1 | I | 4 | random | 1.20 | 0.044 |
| ppo_release_v0→ppo_release_v1 | I | 4 | static_risk | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 4 | static_boundary | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 4 | center_residual | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 4 | coordinate_residual | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 4 | target_only | 1.00 | 0.033 |
| ppo_release_v0→ppo_release_v1 | I | 4 | directed_residual | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 4 | directed_context_laplace | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 4 | coordinate_margin_frontier | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 4 | directed_margin_frontier | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 0 | random | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | static_risk | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | static_boundary | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | center_residual | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | coordinate_residual | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | target_only | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | directed_residual | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | directed_context_laplace | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | coordinate_margin_frontier | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | directed_margin_frontier | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | I | 6 | random | 2.00 | 0.084 |
| ppo_release_v1→ppo_release_v2 | I | 6 | static_risk | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 6 | static_boundary | 2.00 | 0.062 |
| ppo_release_v1→ppo_release_v2 | I | 6 | center_residual | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 6 | coordinate_residual | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 6 | target_only | 1.00 | 0.019 |
| ppo_release_v1→ppo_release_v2 | I | 6 | directed_residual | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 6 | directed_context_laplace | 1.00 | 0.071 |
| ppo_release_v1→ppo_release_v2 | I | 6 | coordinate_margin_frontier | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 6 | directed_margin_frontier | 1.00 | 0.052 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
