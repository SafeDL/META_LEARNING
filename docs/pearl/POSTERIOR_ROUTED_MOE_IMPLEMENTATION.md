# Posterior-routed MoE 工程边界

MoE actor/router 是原始 PEARL 之外的可选扩展。它复用当前统一的 context encoder、recent context replay、无放回 meta-batch、无梯度适应和双模式/双域评估契约。

工程硬约束：router 输入字段必须 allowlist 并写入 metadata；task ID、geometry ID、split、隐藏规则和 query 信息禁止输入；一集内 route 固定且记录 route hash；router 的梯度边界、专家归一化、inactive expert 梯度和 checkpoint round-trip 由契约测试覆盖。

所有新 dense/MoE checkpoint 都必须使用 `pearl_checkpoint` 与 `transition_product_recent_context` 方法契约，architecture metadata 必须包含 actor 架构、context aggregation、router input mode 和专家配置。加载器不提供旧产物 fallback 或迁移。历史工程 smoke/机制结果已删除；当前代码正确性不构成性能或专门化结论。
