# Posterior-routed MoE 路线图

当前状态：dense PEARL 工程契约已按当前完整实现统一，正式后验适应证据尚未建立；MoE 只保留代码研究接口，不具有有效性结论。

阶段必须顺序执行：

1. 重新训练并验证当前 dense PEARL；
2. 在相同 replay/context/evaluation 契约下做 MoE 工程正确性测试；
3. 用容量/计算匹配基线和冻结干预验证 posterior 是否真正驱动路由；
4. 仅在前三阶段通过后研究主动 support、迁移性门控与约束场景挖掘。

所有阶段共同遵守：task ID、geometry ID、split、隐藏规则和 query 结果不得进入 encoder、policy 或 router；ID/OOD 与 posterior mean/sampled 必须分开报告；历史未版本化 checkpoint 和旧 pilot 不得恢复或引用。

进入下一阶段需要列出改动文件、命令、测试结果、当前 manifest/hash、未满足项和明确准入决定。smoke 只证明流程可运行，不证明性能或机制。
