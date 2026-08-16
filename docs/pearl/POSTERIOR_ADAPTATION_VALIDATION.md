# PEARL 后验适应验证

状态：`RESET / FORMAL RESULTS PENDING`。

旧 checkpoint 和旧 pilot 使用退役协议，已删除且不可作为证据。进入任何 MoE 性能/机制结论前，必须先用当前完整实现证明 dense PEARL 的 support 驱动后验适应。

## 冻结方法

- checkpoint：只接受 `pearl_checkpoint` 与 `transition_product_recent_context` 方法契约。
- encoder：默认逐 transition Product-of-Gaussians；episode-product 只能作为命名消融。
- context：每任务 recent buffer；与 SAC RL batch episode 严格隔离。
- meta-batch：任务无放回。
- K：`0/1/2/4/8`，固定嵌套 context。
- 主 checkpoint 选择：`posterior_mean_deterministic` validation。
- 完整评估：同时报告 `posterior_mean_deterministic` 与 `posterior_sampled`。
- 测试：`meta_test_template`（ID 已知逻辑类型）和 `meta_test_logical`（OOD 未见逻辑类型）独立报告。

## 必需对照

至少包含 `pearl_full`、`pearl_no_context`、`deterministic_latent`、`topology_only`、`context_only_no_topology`，并保留预算匹配的 scratch/pooled SAC。所有随机训练方法使用相同预注册 seeds；support case、query case 和实际环境步数逐任务可追溯。

## 通过条件

1. 多训练 seed 下，K>0 的 full 相对 no-context 有预注册的正向任务级置信区间。
2. 同几何异规则任务的 posterior 在 support 后出现高于机会水平的可复现区分，而 K=0 不可由隐藏规则泄漏区分。
3. posterior 变化与 query 性能变化相关，但不得把相关性写成因果证明。
4. posterior-sampled 报告没有隐藏均值策略下看不到的系统性失败或负迁移。
5. ID 与 OOD 结论分别成立，且报告所有任务而非只报告平均数。
6. invalid penalty、参数 hash、输入 hash、context/RL episode 隔离和无梯度约束全部通过。

任一条件未满足时，状态保持 `INCOMPLETE`，不得用 smoke、单 seed 或 validation 数字替代正式 holdout 结论。

## 产物要求

新产物必须包含当前 checkpoint manifest、resolved config、训练/评估 seed、任务与案例 hash、两种 query mode、两个测试 regime、逐任务逐 K 指标、统计汇总和 validation freeze。当前 [`results/pearl_learning`](../../results/pearl_learning/README.md) 明确记录“无正式性能结果”。
