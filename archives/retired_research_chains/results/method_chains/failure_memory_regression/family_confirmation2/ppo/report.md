# NL-IDM 双向版本变化测试

协议：`nl-bidirectional-v1`；数据身份：`family_confirmation2`；588 个冻结场景，1764 次配对物理执行。
主比较方法：`directed_role_gated_edges`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| ppo_release_v0→ppo_release_v1 | R | 7 | random | 0.30 | 0.011 |
| ppo_release_v0→ppo_release_v1 | R | 7 | static_risk | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | R | 7 | static_boundary | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | R | 7 | coordinate_residual | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | R | 7 | directed_margin_frontier | 7.00 | 0.181 |
| ppo_release_v0→ppo_release_v1 | R | 7 | static_role_coverage2 | 2.00 | 0.048 |
| ppo_release_v0→ppo_release_v1 | R | 7 | coordinate_role_gated | 7.00 | 0.176 |
| ppo_release_v0→ppo_release_v1 | R | 7 | directed_role_gated | 7.00 | 0.176 |
| ppo_release_v0→ppo_release_v1 | R | 7 | directed_role_gated_edges | 7.00 | 0.162 |
| ppo_release_v0→ppo_release_v1 | R | 7 | coordinate_multi_frontier | 7.00 | 0.176 |
| ppo_release_v0→ppo_release_v1 | R | 7 | multi_frontier_role | 7.00 | 0.176 |
| ppo_release_v0→ppo_release_v1 | I | 7 | random | 1.50 | 0.059 |
| ppo_release_v0→ppo_release_v1 | I | 7 | static_risk | 2.00 | 0.162 |
| ppo_release_v0→ppo_release_v1 | I | 7 | static_boundary | 1.00 | 0.090 |
| ppo_release_v0→ppo_release_v1 | I | 7 | coordinate_residual | 2.00 | 0.143 |
| ppo_release_v0→ppo_release_v1 | I | 7 | directed_margin_frontier | 2.00 | 0.148 |
| ppo_release_v0→ppo_release_v1 | I | 7 | static_role_coverage2 | 2.00 | 0.029 |
| ppo_release_v0→ppo_release_v1 | I | 7 | coordinate_role_gated | 2.00 | 0.114 |
| ppo_release_v0→ppo_release_v1 | I | 7 | directed_role_gated | 2.00 | 0.081 |
| ppo_release_v0→ppo_release_v1 | I | 7 | directed_role_gated_edges | 2.00 | 0.076 |
| ppo_release_v0→ppo_release_v1 | I | 7 | coordinate_multi_frontier | 2.00 | 0.114 |
| ppo_release_v0→ppo_release_v1 | I | 7 | multi_frontier_role | 2.00 | 0.081 |
| ppo_release_v1→ppo_release_v2 | R | 4 | random | 0.30 | 0.021 |
| ppo_release_v1→ppo_release_v2 | R | 4 | static_risk | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 4 | static_boundary | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 4 | coordinate_residual | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 4 | directed_margin_frontier | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 4 | static_role_coverage2 | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 4 | coordinate_role_gated | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 4 | directed_role_gated | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 4 | directed_role_gated_edges | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 4 | coordinate_multi_frontier | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 4 | multi_frontier_role | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 2 | random | 0.60 | 0.022 |
| ppo_release_v1→ppo_release_v2 | I | 2 | static_risk | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 2 | static_boundary | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 2 | coordinate_residual | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 2 | directed_margin_frontier | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 2 | static_role_coverage2 | 1.00 | 0.090 |
| ppo_release_v1→ppo_release_v2 | I | 2 | coordinate_role_gated | 2.00 | 0.176 |
| ppo_release_v1→ppo_release_v2 | I | 2 | directed_role_gated | 2.00 | 0.176 |
| ppo_release_v1→ppo_release_v2 | I | 2 | directed_role_gated_edges | 2.00 | 0.176 |
| ppo_release_v1→ppo_release_v2 | I | 2 | coordinate_multi_frontier | 2.00 | 0.176 |
| ppo_release_v1→ppo_release_v2 | I | 2 | multi_frontier_role | 2.00 | 0.176 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
