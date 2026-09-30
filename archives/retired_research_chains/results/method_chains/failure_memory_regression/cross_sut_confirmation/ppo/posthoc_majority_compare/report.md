# NL-IDM 双向版本变化测试

协议：`nl-bidirectional-v1`；数据身份：`cross_sut_confirmation`；225 个冻结场景，675 次配对物理执行。
主比较方法：`directed_majority_local`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| ppo_release_v0→ppo_release_v1 | R | 15 | static_role_coverage2 | 14.00 | 0.581 |
| ppo_release_v0→ppo_release_v1 | R | 15 | coordinate_majority_local | 8.00 | 0.400 |
| ppo_release_v0→ppo_release_v1 | R | 15 | directed_majority_local | 7.00 | 0.395 |
| ppo_release_v0→ppo_release_v1 | R | 15 | coordinate_role_gated | 7.00 | 0.381 |
| ppo_release_v0→ppo_release_v1 | R | 15 | directed_role_gated | 7.00 | 0.386 |
| ppo_release_v0→ppo_release_v1 | R | 15 | directed_role_gated_edges | 7.00 | 0.381 |
| ppo_release_v0→ppo_release_v1 | I | 4 | static_role_coverage2 | 3.00 | 0.119 |
| ppo_release_v0→ppo_release_v1 | I | 4 | coordinate_majority_local | 3.00 | 0.210 |
| ppo_release_v0→ppo_release_v1 | I | 4 | directed_majority_local | 3.00 | 0.210 |
| ppo_release_v0→ppo_release_v1 | I | 4 | coordinate_role_gated | 3.00 | 0.210 |
| ppo_release_v0→ppo_release_v1 | I | 4 | directed_role_gated | 3.00 | 0.210 |
| ppo_release_v0→ppo_release_v1 | I | 4 | directed_role_gated_edges | 3.00 | 0.210 |
| ppo_release_v1→ppo_release_v2 | R | 0 | static_role_coverage2 | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | coordinate_majority_local | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | directed_majority_local | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | coordinate_role_gated | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | directed_role_gated | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | directed_role_gated_edges | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | I | 6 | static_role_coverage2 | 1.00 | 0.071 |
| ppo_release_v1→ppo_release_v2 | I | 6 | coordinate_majority_local | 2.00 | 0.138 |
| ppo_release_v1→ppo_release_v2 | I | 6 | directed_majority_local | 2.00 | 0.138 |
| ppo_release_v1→ppo_release_v2 | I | 6 | coordinate_role_gated | 2.00 | 0.138 |
| ppo_release_v1→ppo_release_v2 | I | 6 | directed_role_gated | 2.00 | 0.138 |
| ppo_release_v1→ppo_release_v2 | I | 6 | directed_role_gated_edges | 2.00 | 0.138 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
