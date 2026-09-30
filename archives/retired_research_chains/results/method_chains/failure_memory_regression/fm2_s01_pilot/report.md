# FM²-FBT S01 development pilot

冻结候选池中的目标真实 failure：23/128。

| Method | D@5 | D@10 | D@20 | D@30 | D@50 | Recall@50 | HitRate@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FBRT-Memory-Exploit-v3 | 5.000 | 10.000 | 19.000 | 22.500 | 23.000 | 1.000 | 0.460 |
| FM2-FBT | 5.000 | 10.000 | 19.300 | 22.700 | 23.000 | 1.000 | 0.460 |
| FM2-NoDFE | 5.000 | 10.000 | 19.000 | 22.000 | 23.000 | 1.000 | 0.460 |
| FailureDistance-v2 | 4.200 | 7.300 | 14.900 | 22.200 | 23.000 | 1.000 | 0.460 |
| HistoryRank-UCB-v2 | 5.000 | 10.000 | 20.000 | 23.000 | 23.000 | 1.000 | 0.460 |
| Random | 0.700 | 1.100 | 2.700 | 4.400 | 8.200 | 0.357 | 0.164 |

FM2-FBT 在本 S01 开发库中未达到预先规定的清晰优势判据。

最强既有基线（先按 D@50，平局再按 D@20/D@30）：HistoryRank-UCB-v2。
FM2-NoDFE 的 D@50 为 23.0；FM2-FBT 为 23.0。
DFE 相对 NoDFE：D@20 +0.30，D@30 +0.70，D@50 +0.00。
固定候选的历史检索权重在目标反馈后的平均 L1 变化：0.0013876993907615542。
FM2-FBT 每次会话平均：Global 查询 32.0 次、发现 13.8 个；Local 查询 18.0 次、发现 9.2 个。
首次 failure 位次、发现 5 个 failure 所需查询数和重复标准差见 evaluation/repeats.csv。

## 诊断与边界

该库只有 23 个目标 failure，多种定向方法在 50 次预算内都已找全，D@50 出现任务天花板；这不能证明 FM² 一般性弱于基线，也不能支持其更优。
固定候选的检索变化很小，优先审计 Component 2 的目标证据门控和直接 support 分支；DFE 只略微提前发现，Component 3 在本库未提高 D@50。
后续若继续开发，应另冻结新的 target/pool 协议；本库不得作为独立 confirmation。

## 信息隔离与成本

泄漏审计：{'source_target_excluded': True, 'model_checkpoint_matches_pre_target_protocol': True, 'model_checkpoint_predates_target_bank': True, 'all_queries_in_frozen_manifest': True, 'all_sessions_have_50_distinct_queries': True, 'protocol_amendments': 'results\\method_chains\\failure_memory_regression\\fm2_s01_pilot\\protocol_amendments.jsonl'}。
物理执行：source bank 768 次，source 契约冒烟测试 8 次，target bank 128 次；三种子训练累计约 16.0 秒。
10 次逻辑重放共用一份物理目标响应库，不是独立车辆试验。
Attention 权重仅作检索诊断，不作因果解释。
目标结果仅经计费的 TargetOracle.query 进入选择器。
