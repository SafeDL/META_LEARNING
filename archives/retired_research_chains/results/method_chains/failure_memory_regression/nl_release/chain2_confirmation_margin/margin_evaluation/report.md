# NL-IDM 双向版本变化测试

协议：`nl-bidirectional-v1`；数据身份：`confirmation4`；441 个冻结场景，1323 次配对物理执行。
主比较方法：`directed_margin_frontier`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| nl_v0→nl2_v1 | R | 3 | random | 0.20 | 0.015 |
| nl_v0→nl2_v1 | R | 3 | static_risk | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 3 | static_boundary | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 3 | center_residual | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 3 | coordinate_residual | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 3 | target_only | 1.00 | 0.033 |
| nl_v0→nl2_v1 | R | 3 | directed_residual | 1.00 | 0.014 |
| nl_v0→nl2_v1 | R | 3 | directed_context_laplace | 0.00 | 0.000 |
| nl_v0→nl2_v1 | R | 3 | directed_context_ucb | 1.00 | 0.043 |
| nl_v0→nl2_v1 | R | 3 | static_margin_coverage | 1.00 | 0.071 |
| nl_v0→nl2_v1 | R | 3 | coordinate_margin_frontier | 3.00 | 0.190 |
| nl_v0→nl2_v1 | R | 3 | directed_margin_frontier | 3.00 | 0.195 |
| nl_v0→nl2_v1 | I | 0 | random | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | static_risk | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | static_boundary | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | center_residual | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | coordinate_residual | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | target_only | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_residual | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_context_laplace | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_context_ucb | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | static_margin_coverage | 0.00 | NA |
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
| nl2_v1→nl2_v2 | R | 0 | directed_context_ucb | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | static_margin_coverage | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_margin_frontier | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_margin_frontier | 0.00 | NA |
| nl2_v1→nl2_v2 | I | 17 | random | 1.50 | 0.077 |
| nl2_v1→nl2_v2 | I | 17 | static_risk | 6.00 | 0.443 |
| nl2_v1→nl2_v2 | I | 17 | static_boundary | 5.00 | 0.143 |
| nl2_v1→nl2_v2 | I | 17 | center_residual | 6.00 | 0.381 |
| nl2_v1→nl2_v2 | I | 17 | coordinate_residual | 7.00 | 0.410 |
| nl2_v1→nl2_v2 | I | 17 | target_only | 0.00 | 0.000 |
| nl2_v1→nl2_v2 | I | 17 | directed_residual | 7.00 | 0.376 |
| nl2_v1→nl2_v2 | I | 17 | directed_context_laplace | 13.00 | 0.800 |
| nl2_v1→nl2_v2 | I | 17 | directed_context_ucb | 13.00 | 0.676 |
| nl2_v1→nl2_v2 | I | 17 | static_margin_coverage | 6.00 | 0.443 |
| nl2_v1→nl2_v2 | I | 17 | coordinate_margin_frontier | 7.00 | 0.429 |
| nl2_v1→nl2_v2 | I | 17 | directed_margin_frontier | 13.00 | 0.771 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
