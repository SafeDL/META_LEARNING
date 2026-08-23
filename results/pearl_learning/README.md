# PEARL 当前结果状态

当前没有可报告的 PEARL 性能结果。

2026-08-11 完成理论校验后，旧 checkpoint、训练日志、评估指标、机制 pilot 和由其生成的汇总结论已全部删除。旧产物使用的 context 聚合、replay 取样和评估口径与当前实现不一致，不能迁移或与当前结果合并。

当前目录只保留 `posterior_adaptation/taskbooks/` 与 `posterior_adaptation/casebooks/`，它们是冻结的任务/案例输入，不是性能证据。后续只有满足以下条件的产物才能进入本目录：

- checkpoint schema 为 `pearl_checkpoint`，且方法契约为 `transition_product_recent_context`；
- architecture metadata 明确记录 `context_aggregation`；
- 训练使用 recent context buffer、context/RL episode 隔离和无放回 meta-batch；
- 同时报告 posterior-mean deterministic 与 posterior-sampled query；
- 分开报告 `meta_test_template`（ID，已知逻辑类型）和 `meta_test_logical`（OOD，未见逻辑类型）；
- 通过当前契约测试、输入哈希与无梯度评估审计。

实现与重跑说明见 [`../../archives/pearl_learning/README.md`](../../archives/pearl_learning/README.md)。
