# NL-IDM 双向版本变化测试

协议：`nl-bidirectional-v1`；数据身份：`confirmation4`；441 个冻结场景，1323 次配对物理执行。
主比较方法：`directed_majority_local`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| nl_v0→nl2_v1 | R | 3 | static_role_coverage2 | 1.00 | 0.071 |
| nl_v0→nl2_v1 | R | 3 | coordinate_majority_local | 3.00 | 0.195 |
| nl_v0→nl2_v1 | R | 3 | directed_majority_local | 3.00 | 0.195 |
| nl_v0→nl2_v1 | R | 3 | coordinate_role_gated | 3.00 | 0.195 |
| nl_v0→nl2_v1 | R | 3 | directed_role_gated | 3.00 | 0.195 |
| nl_v0→nl2_v1 | R | 3 | directed_role_gated_edges | 3.00 | 0.195 |
| nl_v0→nl2_v1 | I | 0 | static_role_coverage2 | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | coordinate_majority_local | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_majority_local | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | coordinate_role_gated | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_role_gated | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_role_gated_edges | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | static_role_coverage2 | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_majority_local | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_majority_local | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_role_gated | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_role_gated | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_role_gated_edges | 0.00 | NA |
| nl2_v1→nl2_v2 | I | 17 | static_role_coverage2 | 0.00 | 0.000 |
| nl2_v1→nl2_v2 | I | 17 | coordinate_majority_local | 14.00 | 0.500 |
| nl2_v1→nl2_v2 | I | 17 | directed_majority_local | 10.00 | 0.452 |
| nl2_v1→nl2_v2 | I | 17 | coordinate_role_gated | 15.00 | 0.810 |
| nl2_v1→nl2_v2 | I | 17 | directed_role_gated | 17.00 | 0.905 |
| nl2_v1→nl2_v2 | I | 17 | directed_role_gated_edges | 17.00 | 0.905 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
