# NL-IDM 双向版本变化测试

协议：`nl-bidirectional-v1`；数据身份：`confirmation3`；441 个冻结场景，1323 次配对物理执行。
主比较方法：`directed_context_ucb`；所有方法共用固定方向日程与查询预算。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| nl_v0→nl2_v1 | R | 6 | random | 0.30 | 0.008 |
| nl_v0→nl2_v1 | R | 6 | static_risk | 1.00 | 0.095 |
| nl_v0→nl2_v1 | R | 6 | static_boundary | 1.00 | 0.062 |
| nl_v0→nl2_v1 | R | 6 | center_residual | 1.00 | 0.095 |
| nl_v0→nl2_v1 | R | 6 | coordinate_residual | 1.00 | 0.095 |
| nl_v0→nl2_v1 | R | 6 | target_only | 2.00 | 0.129 |
| nl_v0→nl2_v1 | R | 6 | directed_residual | 1.00 | 0.095 |
| nl_v0→nl2_v1 | R | 6 | directed_context_laplace | 1.00 | 0.095 |
| nl_v0→nl2_v1 | R | 6 | directed_context_ucb | 1.00 | 0.095 |
| nl_v0→nl2_v1 | I | 0 | random | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | static_risk | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | static_boundary | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | center_residual | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | coordinate_residual | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | target_only | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_residual | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_context_laplace | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_context_ucb | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | random | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | static_risk | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | static_boundary | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | center_residual | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_residual | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | target_only | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_residual | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_context_laplace | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_context_ucb | 0.00 | NA |
| nl2_v1→nl2_v2 | I | 21 | random | 2.20 | 0.108 |
| nl2_v1→nl2_v2 | I | 21 | static_risk | 12.00 | 0.648 |
| nl2_v1→nl2_v2 | I | 21 | static_boundary | 13.00 | 0.743 |
| nl2_v1→nl2_v2 | I | 21 | center_residual | 14.00 | 0.771 |
| nl2_v1→nl2_v2 | I | 21 | coordinate_residual | 14.00 | 0.757 |
| nl2_v1→nl2_v2 | I | 21 | target_only | 3.00 | 0.190 |
| nl2_v1→nl2_v2 | I | 21 | directed_residual | 14.00 | 0.776 |
| nl2_v1→nl2_v2 | I | 21 | directed_context_laplace | 16.00 | 0.881 |
| nl2_v1→nl2_v2 | I | 21 | directed_context_ucb | 16.00 | 0.905 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
