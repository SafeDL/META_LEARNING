# NL-IDM 双向版本变化测试

协议：`nl-bidirectional-v1`；数据身份：`development2`；75 个冻结场景，225 次配对物理执行。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| nl_v0→nl2_v1 | R | 0 | random | 0.00 | NA |
| nl_v0→nl2_v1 | R | 0 | static_risk | 0.00 | NA |
| nl_v0→nl2_v1 | R | 0 | static_boundary | 0.00 | NA |
| nl_v0→nl2_v1 | R | 0 | center_residual | 0.00 | NA |
| nl_v0→nl2_v1 | R | 0 | coordinate_residual | 0.00 | NA |
| nl_v0→nl2_v1 | R | 0 | target_only | 0.00 | NA |
| nl_v0→nl2_v1 | R | 0 | directed_residual | 0.00 | NA |
| nl_v0→nl2_v1 | R | 0 | directed_context_laplace | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | random | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | static_risk | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | static_boundary | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | center_residual | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | coordinate_residual | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | target_only | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_residual | 0.00 | NA |
| nl_v0→nl2_v1 | I | 0 | directed_context_laplace | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | random | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | static_risk | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | static_boundary | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | center_residual | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | coordinate_residual | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | target_only | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_residual | 0.00 | NA |
| nl2_v1→nl2_v2 | R | 0 | directed_context_laplace | 0.00 | NA |
| nl2_v1→nl2_v2 | I | 0 | random | 0.00 | NA |
| nl2_v1→nl2_v2 | I | 0 | static_risk | 0.00 | NA |
| nl2_v1→nl2_v2 | I | 0 | static_boundary | 0.00 | NA |
| nl2_v1→nl2_v2 | I | 0 | center_residual | 0.00 | NA |
| nl2_v1→nl2_v2 | I | 0 | coordinate_residual | 0.00 | NA |
| nl2_v1→nl2_v2 | I | 0 | target_only | 0.00 | NA |
| nl2_v1→nl2_v2 | I | 0 | directed_residual | 0.00 | NA |
| nl2_v1→nl2_v2 | I | 0 | directed_context_laplace | 0.00 | NA |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
