# 迁移性感知元强化学习：后续目标

本文只定义待验证研究目标，不包含当前性能结论。旧 PEARL checkpoint、后验诊断和迁移性数值已随退役 schema 删除。

迁移性应定义为：在预注册的新任务 support 交互预算与冻结 query 集下，使用已有元模型相对无迁移/在线基线的任务级收益。它不能由地图外观相似度或测试结果事后反推。

后续研究顺序：

1. 先完成当前 dense PEARL 的多 seed ID/OOD 后验适应验证；
2. 仅用任务先验描述和允许的 support 轨迹构造覆盖度、posterior uncertainty 与 transferability 特征；
3. 只在 validation 上校准阈值，并在证据不足、类别单一或 out-of-sample 覆盖不足时拒绝给出阈值；
4. validation freeze 后再运行独立 holdout；
5. 迁移门控分别报告允许适应、拒绝适应和负迁移，不用总体平均掩盖失败任务。

可选的 disentangled representation、主动 support selection 和 posterior-routed MoE 都是独立研究变量。它们不得读取 query、隐藏规则或未执行 rollout 的结果；必须与 fixed/random support、dense PEARL、no-context 和预算匹配 SAC 对照。后验方差下降只证明 encoder 变得更确定，不自动证明语义可解释、迁移有益或风险更低。

统计单位是任务和独立训练 seed，而不是同一任务内的 query episode。正式产物必须使用 `pearl_checkpoint` 与 `transition_product_recent_context` 方法契约，同时报告 posterior mean/sampled 和 ID/OOD，并由 taskbook、casebook、checkpoint 与 validation-freeze hash 完整追溯。
