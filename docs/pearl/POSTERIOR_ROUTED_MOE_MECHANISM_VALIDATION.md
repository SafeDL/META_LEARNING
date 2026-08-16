# Posterior-routed MoE 机制验证

状态：`PENDING DENSE PEARL VALIDATION`。

历史 MoE pilot 与 checkpoint 已随旧 PEARL 契约删除，其数值不能继续作为诊断或否定性证据。后续机制实验必须从当前 checkpoint 契约重新开始。

需要排除三种替代解释：收益只来自更多参数、router 只做静态拓扑分类、结果只来自单 seed/少量 query。最小实验矩阵包含 dense PEARL、容量匹配 dense、PEARL-MoE、MoE-SAC，以及 frozen-prior router、uniform router、expert knockout、posterior 输入消融。support 轨迹和 query 评价必须成对，router 不得读取 query。

准入条件：

- 当前 dense PEARL 后验适应已按 [`POSTERIOR_ADAPTATION_VALIDATION.md`](POSTERIOR_ADAPTATION_VALIDATION.md) 通过；
- 所有方法使用 transition-product encoder、recent context、disjoint RL replay 和无放回 meta-batch；
- 多训练 seed、任务级统计、容量/计算 profile 完整；
- full 相对 frozen/uniform 路由有稳定增益，并有可复核的 route hash、专家权重和动作差异证据；
- ID/OOD、posterior mean/sampled、逐任务负迁移分别报告。

未满足以上条件时，只能称为工程 smoke 或探索性 pilot，不能声称 posterior 驱动了有益专家专门化。
