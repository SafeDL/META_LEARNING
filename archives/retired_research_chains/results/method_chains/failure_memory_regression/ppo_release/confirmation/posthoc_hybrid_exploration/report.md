# NL-IDM 双向版本变化测试

协议：`nl-bidirectional-v1`；数据身份：`ppo_confirmation`；225 个冻结场景，675 次配对物理执行。
主比较方法：`adaptive_role_hybrid`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| ppo_release_v0→ppo_release_v1 | R | 5 | static_risk | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | R | 5 | coordinate_residual | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | R | 5 | static_role_coverage2 | 5.00 | 0.143 |
| ppo_release_v0→ppo_release_v1 | R | 5 | coordinate_role_gated | 5.00 | 0.262 |
| ppo_release_v0→ppo_release_v1 | R | 5 | directed_role_gated | 5.00 | 0.262 |
| ppo_release_v0→ppo_release_v1 | R | 5 | adaptive_role_hybrid | 5.00 | 0.143 |
| ppo_release_v0→ppo_release_v1 | I | 10 | static_risk | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 10 | coordinate_residual | 0.00 | 0.000 |
| ppo_release_v0→ppo_release_v1 | I | 10 | static_role_coverage2 | 3.00 | 0.148 |
| ppo_release_v0→ppo_release_v1 | I | 10 | coordinate_role_gated | 5.00 | 0.319 |
| ppo_release_v0→ppo_release_v1 | I | 10 | directed_role_gated | 5.00 | 0.324 |
| ppo_release_v0→ppo_release_v1 | I | 10 | adaptive_role_hybrid | 5.00 | 0.319 |
| ppo_release_v1→ppo_release_v2 | R | 5 | static_risk | 5.00 | 0.181 |
| ppo_release_v1→ppo_release_v2 | R | 5 | coordinate_residual | 5.00 | 0.186 |
| ppo_release_v1→ppo_release_v2 | R | 5 | static_role_coverage2 | 5.00 | 0.262 |
| ppo_release_v1→ppo_release_v2 | R | 5 | coordinate_role_gated | 5.00 | 0.262 |
| ppo_release_v1→ppo_release_v2 | R | 5 | directed_role_gated | 5.00 | 0.262 |
| ppo_release_v1→ppo_release_v2 | R | 5 | adaptive_role_hybrid | 5.00 | 0.262 |
| ppo_release_v1→ppo_release_v2 | I | 12 | static_risk | 1.00 | 0.005 |
| ppo_release_v1→ppo_release_v2 | I | 12 | coordinate_residual | 3.00 | 0.090 |
| ppo_release_v1→ppo_release_v2 | I | 12 | static_role_coverage2 | 3.00 | 0.257 |
| ppo_release_v1→ppo_release_v2 | I | 12 | coordinate_role_gated | 8.00 | 0.538 |
| ppo_release_v1→ppo_release_v2 | I | 12 | directed_role_gated | 8.00 | 0.538 |
| ppo_release_v1→ppo_release_v2 | I | 12 | adaptive_role_hybrid | 8.00 | 0.538 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
