# NL-IDM 双向版本变化测试

协议：`nl-bidirectional-v1`；数据身份：`ppo_confirmation`；225 个冻结场景，675 次配对物理执行。
主比较方法：`directed_role_gated_edges`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| ppo_release_v0→ppo_release_v1 | R | 5 | static_risk | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | R | 5 | coordinate_role_gated | 5.00 | 0.262 |
| ppo_release_v0→ppo_release_v1 | R | 5 | directed_role_gated | 5.00 | 0.262 |
| ppo_release_v0→ppo_release_v1 | R | 5 | coordinate_multi_frontier | 5.00 | 0.262 |
| ppo_release_v0→ppo_release_v1 | R | 5 | multi_frontier_role | 5.00 | 0.262 |
| ppo_release_v0→ppo_release_v1 | R | 5 | directed_role_gated_edges | 5.00 | 0.262 |
| ppo_release_v0→ppo_release_v1 | I | 10 | static_risk | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 10 | coordinate_role_gated | 5.00 | 0.319 |
| ppo_release_v0→ppo_release_v1 | I | 10 | directed_role_gated | 5.00 | 0.324 |
| ppo_release_v0→ppo_release_v1 | I | 10 | coordinate_multi_frontier | 5.00 | 0.319 |
| ppo_release_v0→ppo_release_v1 | I | 10 | multi_frontier_role | 5.00 | 0.324 |
| ppo_release_v0→ppo_release_v1 | I | 10 | directed_role_gated_edges | 5.00 | 0.324 |
| ppo_release_v1→ppo_release_v2 | R | 5 | static_risk | 5.00 | 0.181 |
| ppo_release_v1→ppo_release_v2 | R | 5 | coordinate_role_gated | 5.00 | 0.262 |
| ppo_release_v1→ppo_release_v2 | R | 5 | directed_role_gated | 5.00 | 0.262 |
| ppo_release_v1→ppo_release_v2 | R | 5 | coordinate_multi_frontier | 5.00 | 0.262 |
| ppo_release_v1→ppo_release_v2 | R | 5 | multi_frontier_role | 5.00 | 0.262 |
| ppo_release_v1→ppo_release_v2 | R | 5 | directed_role_gated_edges | 5.00 | 0.262 |
| ppo_release_v1→ppo_release_v2 | I | 12 | static_risk | 1.00 | 0.005 |
| ppo_release_v1→ppo_release_v2 | I | 12 | coordinate_role_gated | 8.00 | 0.538 |
| ppo_release_v1→ppo_release_v2 | I | 12 | directed_role_gated | 8.00 | 0.538 |
| ppo_release_v1→ppo_release_v2 | I | 12 | coordinate_multi_frontier | 9.00 | 0.543 |
| ppo_release_v1→ppo_release_v2 | I | 12 | multi_frontier_role | 9.00 | 0.543 |
| ppo_release_v1→ppo_release_v2 | I | 12 | directed_role_gated_edges | 8.00 | 0.543 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
