# Merge Meta-RL Method Flow 当前状态

> 更新日期：2026-08-18。该文件是当前机制门禁链的唯一状态源。
> 历史 Gate 报告（`merge_meta_rl_method_flow_mechanism_gate_report.md`、`merge_meta_rl_method_flow_gate_1b_arrival_controller_addendum.md` 等）已被最新 screened 实验取代：它们的结论（Gate 1B fail / Gate 2 forbidden）不再适用，内容仍保留在 git 历史（commit `ce8a020`）中，状态视为 **SUPERSEDED / HISTORICAL**。
> 当前执行计划：`docs/merge_meta_rl_latest_minimal_revision_plan.md`。

## Gate 状态总览

```text
Gate 0   配置与工程链路                  pass
Gate 1   Task-Policy Conflict          pass
Gate 1B  Quick Single-Task SAC         pass
Gate 2   Context Identifiability       pass
Gate 3   Vanilla PEARL causal chain    fail (round 1, 20k seed 1)
Gate 4   Structure-Aware / Transfer    blocked by Gate 3
```

## 共同机制资产

两个 mechanism Task 是物理完全匹配的 `lane_drop_24` 派生对，仅冻结的 hidden conflict-entry order 不同：

```text
meta_train_lane_drop_24__logical_order_adversary_first
meta_train_lane_drop_24__logical_order_sut_first
```

```text
taskbook           results/pearl_learning/merge_method_flow_logical_order_interpolated/taskbooks
taskbook_hash      a2b4dbf32ec3eebbd8bc1b02ae8b09b37b05126dbd16d32c2ec22696be7226f7
calibration        results/pearl_learning/merge_method_flow_pilot/v2_assets/calibration/critical_thresholds.json
calibration_hash   8efed1246441c49b2c347fe23c6e7c5ebc1085b2c9410b10251e8b10426c4482
config             pearl_learning/configs/merge_method_flow_logical_order_screened_arrival_controller.yaml
config_hash        0759799161bc50a6136399179a83c3fbd5c08c0aa1aa275a0e355782c3afc0a5
casebook profile   order_boundary_screened_v1 (6 matched conditions)
casebook_hashes    adversary_first: dc1f75d2b16e1f0e9f735c640a3021c15939b7a5c0589f1e136e2e658b601613
                   sut_first:        d531e1a6e05632d7e64a37c2832c27c50dd7d4f296ffb52e5d8dee82bac1cf65
```

## Gate 0 — 配置与工程链路：pass

PEARL/SAC 训练与评估链路、run manifest、checkpoint 合约、案例簿 schema 全部可用。

## Gate 1 — Task-Policy Conflict：pass

- 结果路径：`results/pearl_learning/merge_method_flow_logical_order_screened_arrival_controller/gate_1_policy_conflict/policy_conflict_gate.json`
- 100% matched conditions 的最优固定探针发生改变（`matched_case_winner_change_rate = 1.0`）：
  - `adversary_first` → `P2_strong_accelerate`
  - `sut_first` → `P0_coast`
- 两个任务存在稳定、可量化的策略冲突。

## Gate 1B — Quick Single-Task SAC：pass

- 结果路径：`results/pearl_learning/merge_method_flow_logical_order_screened_arrival_controller/gate_1b_single_task_sac_10k/single_task_sac_transfer_gate.json`
- 每任务独立 SAC 10,000 环境步，2x2 transfer matrix 对角优势：
  - `adversary_first` 自任务 VCSR 优势 `+1.0`（最小要求 `+0.125`）
  - `sut_first` 自任务 VCSR 优势 `+0.5`（最小要求 `+0.125`）
- 单任务 SAC 能够学习 task-specific 策略。

## Gate 2 — Context Identifiability：pass

- 结果路径：`results/pearl_learning/merge_method_flow_logical_order_screened_arrival_controller/gate_2_context_identifiability/context_identifiability_gate.json`
- 固定 probing policy（`P6_arrival_gap_heuristic`），transition-only 特征，排除 task_id / geometry_id / descriptor / case_id：
  - held-out accuracy = 1.0（要求 ≥ 0.80）
  - stable energy distance：pass
- 仅凭 transition 数据即可区分两个任务。`next_allowed_stage = gate_3_vanilla_pearl`。

## Gate 3 — Vanilla PEARL causal chain：fail（round 1, 20k seed 1）

目标因果链：

```text
support evidence  ->  q(z|C_correct) != q(z|C_wrong)  ->  pi(a|s,z_correct) != pi(a|s,z_wrong)  ->  J(T,C_correct) > J(T,C_wrong)
```

### 预算与配置

```text
Tasks: 2 (matched logical-order pair)
Seed: 1
Environment steps: 20k
Actor: Dense
Prior: Unit Normal
Scenario Encoder: OFF
MoE: OFF
```

- 配置：`pearl_learning/configs/merge_method_flow_gate3_vanilla_pearl.yaml`
- 案例簿：`results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/mechanism_assets/casebooks/`
  - profile `order_boundary_fewshot_v1`：`train_pool` 6（Gate 1 筛后可行子集）+ `validation_support` 4 + `validation_query` 4，跨 Task matched、跨 split disjoint。
- 训练结果：`results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/mechanism_gate/gate3_vanilla_pearl_20k_seed1/`
- causal audit：`pearl_learning/scripts/audit_gate3_vanilla_pearl_mechanism.py`，对 `prior / correct / wrong / zero` 四种 latent 做 no-gradient intervention。
- 判定文件：`results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/gate_3_causal_audit/gate3_causal_chain_gate.json`

### Round 1 判定结果

```text
Stage A  context -> posterior     PASS
Stage B  posterior -> action      FAIL
Stage C  action -> outcome        FAIL
Status                            fail；passed_stages = [stage_a]
```

关键数值（K=1/2/4 全部稳定）：

```text
Stage A  ||mu_correct - mu_wrong||_2   ≈ 7.7（两任务一致；vs prior ≈ 50-56）
Stage B  mean ||a_correct - a_wrong||_2 ≈ 0.030
         mean ||a_correct - a_prior||_2 ≈ 0.61（对照）
Stage C  VCSR: correct = wrong = prior = 0.00（全部 query cases）
         adversary_first query: 100% invalid
         sut_first query:        100% target collision
```

### 失败定位

- **Stage A 通过**：context evidence 确实改变 posterior（μ_correct ≠ μ_wrong），问题不在 support evidence / Context Encoder 优化 / posterior collapse。
- **Stage B 失败**：两个 posterior 相距 ≈7.7，但确定性动作只差 ≈0.03（约 correct-vs-prior 尺度的 1/20）。Actor 在 correct/wrong 区分方向上 latent-insensitive。
- **Stage C 未达成**：当前 20k policy 在任何 context 下 query VCSR 均为 0（adversary_first 全部 invalid、sut_first 全部 target collision），outcome 环节尚不可评价。20k 预算按计划只用于机制验证，不追求最终性能。

### 下一步（按计划第 9 节）

Stage A 已通过，因此**不修改 Context Encoder**；失败位于 Stage B，当前才有资格研究：

```text
latent conditioning strength / Actor architecture / Critic latent dependence
```

在 Stage B 修复并在 matched query cases 上重新跑出完整链条之前，不进入 Gate 4，不恢复真实物理 Merge Task，不扩大训练预算。

训练诊断日志（`training_updates.jsonl` 新增字段）：

```text
context_encoder_critic_gradient_norm     3.3 -> 8586（encoder 确实被 Bellman 目标优化）
posterior_prior_mean_l2                  0.2 -> 50.3（posterior 远离 unit-normal prior）
evidence_to_prior_precision_ratio        25 -> 13596（evidence 精度主导 posterior）
```

## Gate 4 — Structure-Aware Prior / Transferability / MoE：blocked by Gate 3

Gate 3 通过前不恢复真实物理 Merge Task 变体、不运行 Structure-Aware PEARL、不引入 MoE/GNN/Transformer，也不扩大训练预算（20k 以上）或多 seed 正式评估。

## 冻结参数（Gate 3 期间不得修改）

```text
arrival target scale / collision penalty / collision-risk barrier
IDM target speed / screened case definition / mechanism action interface
strict VCSR definition / Actor-Critic latent 接口 / actor_z.detach()
Product-of-Gaussians 实现 / latent dim / KL beta / SAC 学习率 / reward
```
