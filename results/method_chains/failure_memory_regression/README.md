# FBRT 结果索引

当前正式方法为 `FBRT-Memory-Exploit-v3`。保留以下可复核结果：

| 路径 | 用途 |
| --- | --- |
| `standard_aligned/core/` | 旧场景的参考、候选与目标响应银行，供离线回放读取 |
| `memory_v2/` | 历史失效档案、MOBIL/PPO 测量银行、原始 Memory 基线及修正后的初始状态审计；是当前回放的冻结输入 |
| `memory_exploit_v3/` | 当前策略的留一物理种子簇分析与 compact 检查 |
| `repair_exploit_v3_fullbank/` | 六方法完整冻结银行回放、逐任务结果、清单和算法比较 |
| `interaction_holdout_age080/` | 事前冻结的八种子独立物理验证；目标回归池为零 |
| `current_method_statistical_comparison.md` | 当前方法与五种对照的统计解释和适用边界 |

当前方法在旧银行预算 20 的发现总数为 1106，原 Memory 为 1056；三个独立旧物理种子簇的精确双侧检验为 `p=0.25`，不能宣称显著提升。age080 独立验证的目标回归池为零，也不能作为方法优势证据。详见 [`统计对比`](current_method_statistical_comparison.md) 和 [`独立验证`](interaction_holdout_age080/validation_report.md)。

已归档的中间尝试包括：KDE 局部核原型（留出发现 542，低于原 Memory 的 943）、历史权重调节（942）、完整协方差（941）；三者均未优于当前方案。早期交互开发银行中 age0.30 目标没有形成有效回归，第二轮调整后的五个构建也全部没有有效 ego 碰撞，因此它们不承担当前结论。原始文件及旧源码快照位于 [`archives/experimental_results/`](../../../archives/experimental_results/README.md)，不在当前结果中重复展示。`memory_v2/validation_audit_v3/` 是独立物理诊断，仍保留以支持修正审计的对照。
