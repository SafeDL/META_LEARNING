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
Gate 3   Vanilla PEARL causal chain    fail (round 1 & round 2, 20k seed 1)
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

## Gate 3 — Vanilla PEARL causal chain：fail（Round 1 & Round 2）

目标因果链：

```text
support evidence  ->  q(z|C_correct) != q(z|C_wrong)  ->  pi(a|s,z_correct) != pi(a|s,z_wrong)  ->  J(T,C_correct) > J(T,C_wrong)
```

### 预算与配置

```text
Tasks: 2 (matched logical-order pair)
Seed: 1
Environment steps: 20k per round（每轮只允许一个 20k 实验）
Round 1: Dense Actor + Dense Critic（concat）
Round 2: Dense Actor + Latent-FiLM Critic（分支 B，只改 Critic）
Prior: Unit Normal；Scenario Encoder: OFF；MoE: OFF
```

- Round 1 配置：`pearl_learning/configs/merge_method_flow_gate3_vanilla_pearl.yaml`
- Round 2 配置：`pearl_learning/configs/merge_method_flow_gate3_film_critic.yaml`
- 案例簿（Round 2 起）：`results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/mechanism_assets_screened/casebooks/`
  - profile `order_boundary_fewshot_screened_v1`：train 6（Gate 1 筛后可行子集）+ support 4 + query 4（query 经 Gate 1B 单任务 SAC oracle 筛选，见下）。
- causal audit：`pearl_learning/scripts/audit_gate3_vanilla_pearl_mechanism.py`（prior / correct / wrong / zero 四种 latent，no-gradient intervention + 四组 Stage-B 诊断）。
- 判定：顺序门禁 A→B→C，决策 shot 固定 K=4（`gate3_causal_chain_verdict()`）。

### 判定逻辑（Round 2 起）

```text
Stage A pass 才评价 Stage B；Stage B pass 且 oracle feasibility pass 才评价 Stage C。
Stage C 在 Stage B 未通过时记 blocked_by_stage_b（不是 fail）。
```

### Query Oracle 与 Stage C casebook 修复

Round 1 的 4 个 query conditions 未做可行性筛选。oracle 审计（Gate 1B 两个单任务 SAC 的 2x2 矩阵，0 训练步）显示：

```text
unscreened query: sut_first 对角优势 = -0.25（应为正），sut_first 可实现 VCSR 0/4 -> feasibility fail
```

用 8 个候选条件做 oracle 筛选（`screen_gate3_query_candidates.py`），冻结 4 个聚合门禁通过的 query 条件：

```text
screened query:   对角优势 adversary_first +0.5 / sut_first +0.75，可实现 2/4 与 3/4 -> feasibility pass
```

### Round 1（Dense Critic）判定结果

```text
Stage A  context -> posterior     PASS
Stage B  posterior -> action      FAIL
Stage C  action -> outcome        blocked_by_stage_b
Status                            fail；passed_stages = [stage_a]
```

Stage-B 四组诊断（失败定位依据）：

```text
latent geometry  cos(z_c, z_w) = 0.9972，R_sep = 0.148（posterior 几乎共线，task 信息只是幅值差）
actor pre-tanh   D_raw ≈ 0.09 -> D_tanh ≈ 0.03（tanh 压缩约 3 倍，非纯饱和）
latent interp    action 沿 z_c->z_w 方向基本水平（max step ≈ 0.008，局部 null-space）
critic Q-grid    argmax_c = argmax_w = +1.0 边界，D_Q = 0.0；Q 曲线沿动作网格 100% 单调递增
                 -> 情况 2：Critic 本身对 correct/wrong latent 不敏感（不是 Actor 的问题）
```

### Round 2（Latent-FiLM Critic，分支 B）判定结果

```text
Stage A  context -> posterior     PASS
Stage B  posterior -> action      FAIL（action L2 0.011 / 0.003，比 Round 1 更小）
Stage C  action -> outcome        blocked_by_stage_b（oracle feasibility: pass）
Status                            fail；passed_stages = [stage_a]
R_policy（vs 单任务 SAC 策略差异） 0.011 / 0.003（约 1%）
```

Round 2 的关键变化与未变项：

```text
积极     policy 质量大幅提升：adversary_first correct context VCSR 4/4（Round 1 为 0）
         q_loss 117k -> 2.6k；critic_latent_gradient_norm 2.6 -> 285（Q 对 z 的依赖确实建立）
         sut_first 的 prior(z=0) VCSR 3/4 —— prior 与 correct-posterior 的行为已分化
未变     Q-grid 仍 D_Q = 0.0：81 点网格上 argmax 仍全在 +1.0 边界，无内部峰
         correct/wrong posterior 更共线（cos 0.9997-0.99995，R_sep 0.068-0.141）
         correct vs wrong 动作几乎相同，插值曲线水平
```

### 失败定位（两轮汇总）

- **Stage A 名义通过但分离质量差**：L2 ≈ 3.6-7.7（> 0.5 阈值）但 cos ≈ 0.997+，task 方向只占 posterior 位移的 7-15%。encoder 学到的是"有无 context"的公共方向，不是 task-discriminative 方向。
- **Stage B 两轮均失败**：Round 1 证明 Critic 不敏感（D_Q=0，Q 单调）→ 分支 B 只改 Critic 后（20k）Q 仍未在 correct/wrong 方向分化。按分支 B 判据（Q audit 显示 a_c^Q* ≠ a_w^Q* 后 Actor 才可能自然分化），task-dependent Q 尚未形成。
- **policy 学会的是"有 context vs 无 context"的行为切换，而不是"哪个 task 的 context"**：adversary_first 上 correct 4/4 而 prior 0/4；sut_first 上 prior 3/4 而 correct 0/4。

### 下一步（待定，未实施）

两个 20k 实验（concat 与 FiLM Critic）均未在 correct/wrong 方向建立 Q 分化。下一轮候选方向（需要先决定、只选一个）：

```text
1) 加强 posterior 的 task-discriminative 分离质量（R_sep 目前仅 0.07-0.14，两 posterior 共线）
2) FiLM Critic 继续（更长预算/更强调 z 的调制路径），或 FiLM Actor + FiLM Critic 组合
3) 重新审视 support evidence 构造（context 是否包含足够的 task 区分信息）
```

不进入 Gate 4、不恢复真实物理 Merge Task、不做多模型并行搜索。

训练诊断日志（`training_updates.jsonl` 新增字段）：

```text
context_encoder_critic_gradient_norm     3.3 -> 8586（Round 1）/ 11.5 -> 382（Round 2）
posterior_prior_mean_l2                  0.2 -> 50.3 / 0.2 -> 48.8
evidence_to_prior_precision_ratio        25 -> 13596 / 25 -> 40056
critic_latent_gradient_norm（Round 2）   2.6 -> 285（Q 对 z 的梯度依赖建立）
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
