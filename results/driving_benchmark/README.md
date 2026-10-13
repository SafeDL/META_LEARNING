# 六系统共享响应池与基线轨迹

六个系统共用两个 2048 点场景池，共 12 份响应库、24576 条物理响应。每池含切入和前车急刹各 1024 个场景。

| 位置 | 内容 |
|---|---|
| `pool_*/SUT/responses.npz` | 场景坐标、实测风险与碰撞 |
| `pool_*/SUT/baselines/` | 最新比较使用的 996 条基线轨迹 |
| [protocol.json](protocol.json) | 原始采集协议 |
| [bank_audit.json](bank_audit.json)、[interface_audit.json](interface_audit.json) | 数据与执行接口核验 |
| [baseline_protocol.json](baseline_protocol.json)、[baseline_input_audit.json](baseline_input_audit.json) | 基线设置与输入核验 |

主方法的 60 条轨迹另存于 [CG-BRT 结果目录](../concordance_gated_bayesian_risk_transfer/README.md)。完整比较见[比较表](../concordance_gated_bayesian_risk_transfer/comparison.md)，基线历史信息与反馈见[清单](../concordance_gated_bayesian_risk_transfer/baseline_catalog.json)。
