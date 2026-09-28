# NL-IDM 双向版本变化测试

协议：`nl-bidirectional-v1`；数据身份：`family_confirmation`；200 个冻结场景，600 次配对物理执行。
主比较方法：`directed_role_gated_span`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| nl_v0→nl2_v1 | R | 35 | static_risk | 10.00 | 0.643 |
| nl_v0→nl2_v1 | R | 35 | static_role_coverage2 | 17.00 | 0.743 |
| nl_v0→nl2_v1 | R | 35 | static_role_coverage_latest | 17.00 | 0.743 |
| nl_v0→nl2_v1 | R | 35 | static_role_coverage_span | 17.00 | 0.743 |
| nl_v0→nl2_v1 | R | 35 | coordinate_role_gated | 13.00 | 0.524 |
| nl_v0→nl2_v1 | R | 35 | coordinate_role_gated_latest | 13.00 | 0.524 |
| nl_v0→nl2_v1 | R | 35 | coordinate_role_gated_span | 13.00 | 0.524 |
| nl_v0→nl2_v1 | R | 35 | directed_role_gated | 13.00 | 0.524 |
| nl_v0→nl2_v1 | R | 35 | directed_role_gated_latest | 13.00 | 0.524 |
| nl_v0→nl2_v1 | R | 35 | directed_role_gated_span | 13.00 | 0.524 |
| nl_v0→nl2_v1 | I | 0 | static_risk | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | static_role_coverage2 | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | static_role_coverage_latest | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | static_role_coverage_span | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | coordinate_role_gated | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | coordinate_role_gated_latest | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | coordinate_role_gated_span | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_role_gated | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_role_gated_latest | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_role_gated_span | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | static_risk | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | static_role_coverage2 | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | static_role_coverage_latest | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | static_role_coverage_span | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_role_gated | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_role_gated_latest | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_role_gated_span | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_role_gated | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_role_gated_latest | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_role_gated_span | 0.00 | NA |
| nl2_v1→nl2_v2 | I | 2 | static_risk | 2.00 | 0.186 |
| nl2_v1→nl2_v2 | I | 2 | static_role_coverage2 | 0.00 | 0.000 |
| nl2_v1→nl2_v2 | I | 2 | static_role_coverage_latest | 0.00 | 0.000 |
| nl2_v1→nl2_v2 | I | 2 | static_role_coverage_span | 0.00 | 0.000 |
| nl2_v1→nl2_v2 | I | 2 | coordinate_role_gated | 2.00 | 0.148 |
| nl2_v1→nl2_v2 | I | 2 | coordinate_role_gated_latest | 2.00 | 0.148 |
| nl2_v1→nl2_v2 | I | 2 | coordinate_role_gated_span | 2.00 | 0.148 |
| nl2_v1→nl2_v2 | I | 2 | directed_role_gated | 2.00 | 0.148 |
| nl2_v1→nl2_v2 | I | 2 | directed_role_gated_latest | 2.00 | 0.148 |
| nl2_v1→nl2_v2 | I | 2 | directed_role_gated_span | 2.00 | 0.148 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
