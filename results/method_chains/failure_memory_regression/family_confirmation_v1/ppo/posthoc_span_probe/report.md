# NL-IDM 双向版本变化测试

协议：`nl-bidirectional-v1`；数据身份：`family_confirmation`；200 个冻结场景，600 次配对物理执行。
主比较方法：`directed_role_gated_span`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| ppo_release_v0→ppo_release_v1 | R | 5 | static_risk | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | R | 5 | static_role_coverage2 | 5.00 | 0.286 |
| ppo_release_v0→ppo_release_v1 | R | 5 | static_role_coverage_latest | 5.00 | 0.286 |
| ppo_release_v0→ppo_release_v1 | R | 5 | static_role_coverage_span | 5.00 | 0.286 |
| ppo_release_v0→ppo_release_v1 | R | 5 | coordinate_role_gated | 5.00 | 0.257 |
| ppo_release_v0→ppo_release_v1 | R | 5 | coordinate_role_gated_latest | 5.00 | 0.257 |
| ppo_release_v0→ppo_release_v1 | R | 5 | coordinate_role_gated_span | 5.00 | 0.267 |
| ppo_release_v0→ppo_release_v1 | R | 5 | directed_role_gated | 5.00 | 0.257 |
| ppo_release_v0→ppo_release_v1 | R | 5 | directed_role_gated_latest | 5.00 | 0.267 |
| ppo_release_v0→ppo_release_v1 | R | 5 | directed_role_gated_span | 5.00 | 0.267 |
| ppo_release_v0→ppo_release_v1 | I | 2 | static_risk | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 2 | static_role_coverage2 | 2.00 | 0.024 |
| ppo_release_v0→ppo_release_v1 | I | 2 | static_role_coverage_latest | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 2 | static_role_coverage_span | 1.00 | 0.014 |
| ppo_release_v0→ppo_release_v1 | I | 2 | coordinate_role_gated | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 2 | coordinate_role_gated_latest | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 2 | coordinate_role_gated_span | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 2 | directed_role_gated | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 2 | directed_role_gated_latest | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 2 | directed_role_gated_span | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | R | 0 | static_risk | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | static_role_coverage2 | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | static_role_coverage_latest | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | static_role_coverage_span | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | coordinate_role_gated | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | coordinate_role_gated_latest | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | coordinate_role_gated_span | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | directed_role_gated | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | directed_role_gated_latest | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | R | 0 | directed_role_gated_span | 0.00 | NA |
| ppo_release_v1→ppo_release_v2 | I | 4 | static_risk | 1.00 | 0.019 |
| ppo_release_v1→ppo_release_v2 | I | 4 | static_role_coverage2 | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 4 | static_role_coverage_latest | 1.00 | 0.010 |
| ppo_release_v1→ppo_release_v2 | I | 4 | static_role_coverage_span | 0.00 | 0.000 |
| ppo_release_v1→ppo_release_v2 | I | 4 | coordinate_role_gated | 4.00 | 0.157 |
| ppo_release_v1→ppo_release_v2 | I | 4 | coordinate_role_gated_latest | 4.00 | 0.114 |
| ppo_release_v1→ppo_release_v2 | I | 4 | coordinate_role_gated_span | 4.00 | 0.157 |
| ppo_release_v1→ppo_release_v2 | I | 4 | directed_role_gated | 4.00 | 0.157 |
| ppo_release_v1→ppo_release_v2 | I | 4 | directed_role_gated_latest | 2.00 | 0.014 |
| ppo_release_v1→ppo_release_v2 | I | 4 | directed_role_gated_span | 4.00 | 0.157 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
