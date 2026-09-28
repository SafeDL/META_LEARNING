# NL-IDM 双向版本变化测试

协议：`nl-bidirectional-v1`；数据身份：`family_confirmation2`；588 个冻结场景，1764 次配对物理执行。
主比较方法：`directed_role_gated_edges`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| nl_v0→nl2_v1 | R | 110 | random | 5.80 | 0.282 |
| nl_v0→nl2_v1 | R | 110 | static_risk | 18.00 | 0.976 |
| nl_v0→nl2_v1 | R | 110 | static_boundary | 12.00 | 0.524 |
| nl_v0→nl2_v1 | R | 110 | coordinate_residual | 17.00 | 0.924 |
| nl_v0→nl2_v1 | R | 110 | directed_margin_frontier | 11.00 | 0.548 |
| nl_v0→nl2_v1 | R | 110 | static_role_coverage2 | 15.00 | 0.605 |
| nl_v0→nl2_v1 | R | 110 | coordinate_role_gated | 11.00 | 0.548 |
| nl_v0→nl2_v1 | R | 110 | directed_role_gated | 11.00 | 0.548 |
| nl_v0→nl2_v1 | R | 110 | directed_role_gated_edges | 11.00 | 0.548 |
| nl_v0→nl2_v1 | R | 110 | coordinate_multi_frontier | 15.00 | 0.605 |
| nl_v0→nl2_v1 | R | 110 | multi_frontier_role | 15.00 | 0.605 |
| nl_v0→nl2_v1 | I | 0 | random | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | static_risk | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | static_boundary | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | coordinate_residual | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_margin_frontier | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | static_role_coverage2 | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | coordinate_role_gated | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_role_gated | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_role_gated_edges | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | coordinate_multi_frontier | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | multi_frontier_role | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | random | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | static_risk | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | static_boundary | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_residual | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_margin_frontier | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | static_role_coverage2 | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_role_gated | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_role_gated | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_role_gated_edges | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_multi_frontier | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | multi_frontier_role | 0.00 | NA |
| nl2_v1→nl2_v2 | I | 12 | random | 0.20 | 0.010 |
| nl2_v1→nl2_v2 | I | 12 | static_risk | 8.00 | 0.610 |
| nl2_v1→nl2_v2 | I | 12 | static_boundary | 7.00 | 0.348 |
| nl2_v1→nl2_v2 | I | 12 | coordinate_residual | 9.00 | 0.581 |
| nl2_v1→nl2_v2 | I | 12 | directed_margin_frontier | 9.00 | 0.662 |
| nl2_v1→nl2_v2 | I | 12 | static_role_coverage2 | 0.00 | 0.000 |
| nl2_v1→nl2_v2 | I | 12 | coordinate_role_gated | 12.00 | 0.486 |
| nl2_v1→nl2_v2 | I | 12 | directed_role_gated | 12.00 | 0.486 |
| nl2_v1→nl2_v2 | I | 12 | directed_role_gated_edges | 12.00 | 0.486 |
| nl2_v1→nl2_v2 | I | 12 | coordinate_multi_frontier | 12.00 | 0.486 |
| nl2_v1→nl2_v2 | I | 12 | multi_frontier_role | 12.00 | 0.486 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
