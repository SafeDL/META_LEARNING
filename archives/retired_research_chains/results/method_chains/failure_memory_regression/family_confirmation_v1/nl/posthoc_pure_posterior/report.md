# NL-IDM 双向版本变化测试

协议：`nl-bidirectional-v1`；数据身份：`family_confirmation`；200 个冻结场景，600 次配对物理执行。
主比较方法：`directed_residual`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| nl_v0→nl2_v1 | R | 35 | random | 6.30 | 0.281 |
| nl_v0→nl2_v1 | R | 35 | static_risk | 10.00 | 0.643 |
| nl_v0→nl2_v1 | R | 35 | static_boundary | 11.00 | 0.505 |
| nl_v0→nl2_v1 | R | 35 | center_residual | 11.00 | 0.667 |
| nl_v0→nl2_v1 | R | 35 | coordinate_residual | 10.00 | 0.629 |
| nl_v0→nl2_v1 | R | 35 | target_only | 5.00 | 0.262 |
| nl_v0→nl2_v1 | R | 35 | directed_residual | 11.00 | 0.571 |
| nl_v0→nl2_v1 | R | 35 | directed_context_laplace | 10.00 | 0.733 |
| nl_v0→nl2_v1 | R | 35 | coordinate_margin_frontier | 13.00 | 0.524 |
| nl_v0→nl2_v1 | R | 35 | directed_margin_frontier | 13.00 | 0.524 |
| nl_v0→nl2_v1 | I | 0 | random | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | static_risk | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | static_boundary | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | center_residual | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | coordinate_residual | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | target_only | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_residual | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_context_laplace | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | coordinate_margin_frontier | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_margin_frontier | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | random | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | static_risk | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | static_boundary | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | center_residual | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_residual | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | target_only | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_residual | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_context_laplace | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_margin_frontier | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_margin_frontier | 0.00 | NA |
| nl2_v1→nl2_v2 | I | 2 | random | 0.30 | 0.021 |
| nl2_v1→nl2_v2 | I | 2 | static_risk | 2.00 | 0.186 |
| nl2_v1→nl2_v2 | I | 2 | static_boundary | 2.00 | 0.148 |
| nl2_v1→nl2_v2 | I | 2 | center_residual | 2.00 | 0.186 |
| nl2_v1→nl2_v2 | I | 2 | coordinate_residual | 2.00 | 0.186 |
| nl2_v1→nl2_v2 | I | 2 | target_only | 0.00 | 0.000 |
| nl2_v1→nl2_v2 | I | 2 | directed_residual | 2.00 | 0.186 |
| nl2_v1→nl2_v2 | I | 2 | directed_context_laplace | 2.00 | 0.186 |
| nl2_v1→nl2_v2 | I | 2 | coordinate_margin_frontier | 2.00 | 0.186 |
| nl2_v1→nl2_v2 | I | 2 | directed_margin_frontier | 2.00 | 0.186 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
