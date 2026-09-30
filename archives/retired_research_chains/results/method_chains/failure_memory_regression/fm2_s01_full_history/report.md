# FM²-FBT S01 development pilot

冻结候选池中的目标真实 failure：408/2048。

| Method | D@5 | D@10 | D@20 | D@30 | D@50 | Recall@50 | HitRate@50 | Unfound truth@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FBRT-Memory-Exploit-v3 | 2.000 | 7.000 | 17.000 | 26.000 | 44.000 | 0.108 | 0.880 | 364.000 |
| FM2-FBT | 2.000 | 7.000 | 17.000 | 27.000 | 36.000 | 0.088 | 0.720 | 372.000 |
| FM2-NoDFE | 1.000 | 3.000 | 8.000 | 18.000 | 35.000 | 0.086 | 0.700 | 373.000 |
| FailureDistance-v2 | 3.000 | 7.000 | 16.000 | 24.000 | 40.000 | 0.098 | 0.800 | 368.000 |
| HistoryRank-UCB-v2 | 5.000 | 10.000 | 20.000 | 30.000 | 50.000 | 0.123 | 1.000 | 358.000 |
| Random | 1.000 | 1.000 | 3.000 | 6.000 | 7.000 | 0.017 | 0.140 | 401.000 |

FM2-FBT 在本 S01 开发库中未达到预先规定的清晰优势判据。

最强既有基线（先按 D@50，平局再按 D@20/D@30）：HistoryRank-UCB-v2。
FM2-NoDFE 的 D@50 为 35.0；FM2-FBT 为 36.0。
DFE 相对 NoDFE：D@20 +9.00，D@30 +9.00，D@50 +1.00。
固定候选的历史检索权重在目标反馈后的平均 L1 变化：0.0008922757697291672。
FM2-FBT 每次会话平均：Global 查询 13.0 次、发现 3.0 个；Local 查询 37.0 次、发现 33.0 个。
首次 failure 位次和发现 5 个 failure 所需查询数见 evaluation/repeats.csv。本轮每方法只有一个选择种子，不估计跨种子方差。

## 诊断与边界

该库有 408 个目标 failure；50 次预算内最多可发现 50 个。结果仅支持当前固定库和目标版本的开发判断。
Component 2 的目标证据作用可结合固定候选检索权重变化及 NoDFE 曲线判断；Component 3 的作用以 FBT 与 NoDFE 的各预算截点差值判断。
完整历史库与目标真值按相同场景 ID 对齐后，`idm_ref` 的 263 个失效和 `merge_brake2` 的 355 个失效全部也发生在目标上；六个 source 的逐项重合情况见 evaluation/source_target_overlap.csv。这与 HistoryRank 在 50 次查询中全部命中的结果一致。它说明本任务的历史版本对目标有很强的同场景预测信息，但不能单凭重合率解释内部失效机制。
六个 source 共记录 1,741 个确认失效，FM² 的历史记忆最多保留 64 张 PatternCard。逐场景证据的压缩可能是 FM² 落后于直接历史排序的原因之一；这只是根据当前结果提出的诊断，尚未通过独立消融验证。本轮不依据目标真值重新调节卡片数量或模型。
本库是开发续测，不能作为独立 confirmation。

## 信息隔离与成本

泄漏审计：{'source_target_excluded': True, 'model_checkpoint_matches_pre_target_protocol': True, 'model_checkpoint_predates_target_bank': True, 'all_queries_in_frozen_manifest': True, 'all_sessions_have_50_distinct_queries': True, 'protocol_amendments': 'results\\method_chains\\failure_memory_regression\\fm2_s01_full_history\\protocol_amendments.jsonl'}。
物理执行：历史响应库 12288 次（本轮新测 11520 次），source 契约冒烟测试 8 次，target bank 2048 次（本轮新测 1920 次）；1 个训练种子累计约 6.4 秒。
每方法 1 次逻辑重放共用一份物理目标响应库，不是独立车辆试验。
Attention 权重仅作检索诊断，不作因果解释。
目标结果仅经计费的 TargetOracle.query 进入选择器。
