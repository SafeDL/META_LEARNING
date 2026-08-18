# Gate 3 Vanilla PEARL 因果链报告（Round 1）

日期：2026-08-18。该记录是机制门禁证据，不是论文性能结果。

## 配置与协议

- 任务：物理完全匹配的 `adversary_first` / `sut_first` 对；仅冻结的 conflict-entry order 不同。
- 算法：Vanilla PEARL（Dense Actor，unit-normal prior，Scenario Encoder / MoE OFF），latent dim 5。
- 训练：2 tasks × 20k 环境步 × seed 1，meta-batch 2。只验证机制，不追求性能。
- 案例簿：`order_boundary_fewshot_v1` —— train 6（Gate 1 筛后可行子集）+ support 4 + query 4；跨 Task matched、跨 split disjoint。
- strict 指标：`logical_order_spatiotemporal_near_miss_v3`，calibration 与 Gate 1/1B/2 同一 manifest。
- 训练入口现已对 v2/v3 统一强制 `--critical-thresholds`（`resolve_calibration`）。

证据目录：

```text
配置      pearl_learning/configs/merge_method_flow_gate3_vanilla_pearl.yaml
案例簿    results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/mechanism_assets/casebooks/
训练      results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/mechanism_gate/gate3_vanilla_pearl_20k_seed1/
audit     results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/gate_3_causal_audit/gate3_causal_audit.json
判定      results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/gate_3_causal_audit/gate3_causal_chain_gate.json
```

## Causal audit 结果（K = 1 / 2 / 4 全部稳定）

| Stage | 链条 | 指标 | 结果 |
| --- | --- | --- | --- |
| A | context → posterior | \|\|μ_correct − μ_wrong\|\|₂ | **≈ 7.7（pass）**；vs prior ≈ 50-56 |
| B | posterior → action | mean \|\|a_correct − a_wrong\|\|₂ | **≈ 0.030（fail）**；对照 a_correct vs a_prior ≈ 0.61 |
| C | action → outcome | correct vs wrong VCSR | **0.00 vs 0.00（fail）** |

## 结论

```text
Gate 3 Round 1 status = fail；passed_stages = [stage_a]
```

- **Stage A 通过**：support evidence → posterior 的链路成立，Context Encoder 确实被训练（critic 梯度 norm 3.3 → 8586），posterior 远离 unit-normal prior（mean L2 0.2 → 50.3），evidence 精度主导 posterior（precision ratio 25 → 13596）。
- **Stage B 失败**：Actor 对 z 非零有强响应（correct vs prior ≈ 0.61），但在 correct/wrong 区分方向上 latent-insensitive（≈ 0.03，仅 1/20 尺度）。
- **Stage C 失败**：20k policy 在任何 context 下 query VCSR 均为 0 —— adversary_first query 100% invalid、sut_first query 100% target collision。outcome 环节当前不可评价。

## 下一步

按修订计划第 9 节的阶段判定规则：Stage A 已通过，**不改 Context Encoder**；失败位于 Stage B，才有资格研究 latent conditioning strength / Actor architecture / Critic latent dependence。Stage B 修复前不进入 Gate 4、不恢复真实物理 Merge Task、不扩大训练预算（20k 以上）或做 multi-seed 正式评估。
