# PPO 权重继承链双向版本变化测试

协议：`nl-bidirectional-v1`；数据身份：`core_confirmation`；1089 个冻结场景，3267 次配对物理执行。
主比较方法：`directed_residual`；所有方法共用固定方向日程与查询预算。
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
| ppo_release_v0→ppo_release_v1 | R | 0 | directed_context_laplace | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | R | 0 | directed_context_ucb | 0.00 | NA |
| ppo_release_v0→ppo_release_v1 | I | 26 | random | 1.50 | 0.088 |
| ppo_release_v0→ppo_release_v1 | I | 26 | static_risk | 2.00 | 0.162 |
| ppo_release_v0→ppo_release_v1 | I | 26 | static_boundary | 1.00 | 0.057 |
| ppo_release_v0→ppo_release_v1 | I | 26 | center_residual | 2.00 | 0.143 |
| ppo_release_v0→ppo_release_v1 | I | 26 | coordinate_residual | 2.00 | 0.148 |
| ppo_release_v0→ppo_release_v1 | I | 26 | target_only | 1.00 | 0.043 |
| ppo_release_v0→ppo_release_v1 | I | 26 | directed_residual | 2.00 | 0.152 |
| ppo_release_v0→ppo_release_v1 | I | 26 | directed_context_laplace | 2.00 | 0.152 |
| ppo_release_v0→ppo_release_v1 | I | 26 | directed_context_ucb | 2.00 | 0.152 |
| ppo_release_v1→ppo_release_v2 | R | 1 | random | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 1 | static_risk | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 1 | static_boundary | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 1 | center_residual | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 1 | coordinate_residual | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 1 | target_only | 1.00 | 0.033 |
| ppo_release_v1→ppo_release_v2 | R | 1 | directed_residual | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 1 | directed_context_laplace | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 1 | directed_context_ucb | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 20 | random | 1.20 | 0.059 |
| ppo_release_v1→ppo_release_v2 | I | 20 | static_risk | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 20 | static_boundary | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 20 | center_residual | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 20 | coordinate_residual | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 20 | target_only | 3.00 | 0.152 |
| ppo_release_v1→ppo_release_v2 | I | 20 | directed_residual | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 20 | directed_context_laplace | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 20 | directed_context_ucb | 0.00 | 0.000 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
