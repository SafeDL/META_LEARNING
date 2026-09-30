# NL-IDM 双向版本变化测试

协议：`nl-bidirectional-v1`；数据身份：`development3`；147 个冻结场景，441 次配对物理执行。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| nl3_v0→nl3_v1 | R | 0 | random | 0.00 | NA |
| nl3_v0→nl3_v1 | R | 0 | static_risk | 0.00 | NA |
| nl3_v0→nl3_v1 | R | 0 | static_boundary | 0.00 | NA |
| nl3_v0→nl3_v1 | R | 0 | center_residual | 0.00 | NA |
| nl3_v0→nl3_v1 | R | 0 | coordinate_residual | 0.00 | NA |
| nl3_v0→nl3_v1 | R | 0 | target_only | 0.00 | NA |
| nl3_v0→nl3_v1 | R | 0 | directed_residual | 0.00 | NA |
| nl3_v0→nl3_v1 | I | 0 | random | 0.00 | NA |
| nl3_v0→nl3_v1 | I | 0 | static_risk | 0.00 | NA |
| nl3_v0→nl3_v1 | I | 0 | static_boundary | 0.00 | NA |
| nl3_v0→nl3_v1 | I | 0 | center_residual | 0.00 | NA |
| nl3_v0→nl3_v1 | I | 0 | coordinate_residual | 0.00 | NA |
| nl3_v0→nl3_v1 | I | 0 | target_only | 0.00 | NA |
| nl3_v0→nl3_v1 | I | 0 | directed_residual | 0.00 | NA |
| nl3_v1→nl3_v2 | R | 0 | random | 0.00 | NA |
| nl3_v1→nl3_v2 | R | 0 | static_risk | 0.00 | NA |
| nl3_v1→nl3_v2 | R | 0 | static_boundary | 0.00 | NA |
| nl3_v1→nl3_v2 | R | 0 | center_residual | 0.00 | NA |
| nl3_v1→nl3_v2 | R | 0 | coordinate_residual | 0.00 | NA |
| nl3_v1→nl3_v2 | R | 0 | target_only | 0.00 | NA |
| nl3_v1→nl3_v2 | R | 0 | directed_residual | 0.00 | NA |
| nl3_v1→nl3_v2 | I | 7 | random | 7.00 | 0.510 |
| nl3_v1→nl3_v2 | I | 7 | static_risk | 7.00 | 0.438 |
| nl3_v1→nl3_v2 | I | 7 | static_boundary | 7.00 | 0.381 |
| nl3_v1→nl3_v2 | I | 7 | center_residual | 7.00 | 0.324 |
| nl3_v1→nl3_v2 | I | 7 | coordinate_residual | 7.00 | 0.410 |
| nl3_v1→nl3_v2 | I | 7 | target_only | 7.00 | 0.419 |
| nl3_v1→nl3_v2 | I | 7 | directed_residual | 7.00 | 0.314 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
