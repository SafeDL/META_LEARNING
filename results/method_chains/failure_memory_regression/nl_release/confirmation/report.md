# NL-IDM 双向版本变化测试

协议：`nl-bidirectional-v1`；数据身份：`confirmation`；441 个冻结场景，1323 次配对物理执行。
终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，不增加独立物理样本。

| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |
|---|---|---:|---|---:|---:|
| nl_v0→nl_v1 | R | 0 | random | 0.00 | NA |
| nl_v0→nl_v1 | R | 0 | static_risk | 0.00 | NA |
| nl_v0→nl_v1 | R | 0 | static_boundary | 0.00 | NA |
| nl_v0→nl_v1 | R | 0 | center_residual | 0.00 | NA |
| nl_v0→nl_v1 | R | 0 | coordinate_residual | 0.00 | NA |
| nl_v0→nl_v1 | R | 0 | target_only | 0.00 | NA |
| nl_v0→nl_v1 | R | 0 | directed_residual | 0.00 | NA |
| nl_v0→nl_v1 | I | 3 | random | 0.20 | 0.010 |
| nl_v0→nl_v1 | I | 3 | static_risk | 3.00 | 0.271 |
| nl_v0→nl_v1 | I | 3 | static_boundary | 1.00 | 0.067 |
| nl_v0→nl_v1 | I | 3 | center_residual | 3.00 | 0.252 |
| nl_v0→nl_v1 | I | 3 | coordinate_residual | 3.00 | 0.262 |
| nl_v0→nl_v1 | I | 3 | target_only | 0.00 | 0.000 |
| nl_v0→nl_v1 | I | 3 | directed_residual | 3.00 | 0.262 |
| nl_v1→nl_v2 | R | 0 | random | 0.00 | NA |
| nl_v1→nl_v2 | R | 0 | static_risk | 0.00 | NA |
| nl_v1→nl_v2 | R | 0 | static_boundary | 0.00 | NA |
| nl_v1→nl_v2 | R | 0 | center_residual | 0.00 | NA |
| nl_v1→nl_v2 | R | 0 | coordinate_residual | 0.00 | NA |
| nl_v1→nl_v2 | R | 0 | target_only | 0.00 | NA |
| nl_v1→nl_v2 | R | 0 | directed_residual | 0.00 | NA |
| nl_v1→nl_v2 | I | 8 | random | 1.20 | 0.048 |
| nl_v1→nl_v2 | I | 8 | static_risk | 7.00 | 0.329 |
| nl_v1→nl_v2 | I | 8 | static_boundary | 5.00 | 0.295 |
| nl_v1→nl_v2 | I | 8 | center_residual | 6.00 | 0.310 |
| nl_v1→nl_v2 | I | 8 | coordinate_residual | 6.00 | 0.324 |
| nl_v1→nl_v2 | I | 8 | target_only | 1.00 | 0.038 |
| nl_v1→nl_v2 | I | 8 | directed_residual | 6.00 | 0.281 |

R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。
相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；本表不提供显著性结论。
