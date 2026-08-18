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
Gate 3   Vanilla PEARL causal chain    fail (round 3: Stage A pass, Stage B fail)
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
- **注意（Stage-A 审计后补充）**：Gate 2 看到的是**完整 trajectory summary + 原始 reward**，与 PEARL 训练实际输入（随机 8 transitions/episode、r/200）不同，见下。

## Gate 3 — Vanilla PEARL causal chain：fail（Round 3 后 Stage A 首次通过 v3）

目标因果链：

```text
support evidence  ->  q(z|C_correct) != q(z|C_wrong)  ->  pi(a|s,z_correct) != pi(a|s,z_wrong)  ->  J(T,C_correct) > J(T,C_wrong)
```

### 预算与配置

```text
Tasks: 2 (matched logical-order pair)
Seed: 1
Environment steps: 20k per round（每轮只允许一个 20k 实验）
Round 1: Dense Actor + Dense Critic（concat），随机 context 采样
Round 2: Dense Actor + Latent-FiLM Critic（分支 B，只改 Critic）
Round 3: Round 1 全部设置 + 唯一变量 terminal_stratified_v1 context 采样（分支 A）
Prior: Unit Normal；Scenario Encoder: OFF；MoE: OFF
```

- Round 1 配置：`pearl_learning/configs/merge_method_flow_gate3_vanilla_pearl.yaml`
- Round 2 配置：`pearl_learning/configs/merge_method_flow_gate3_film_critic.yaml`
- Round 3 配置：`pearl_learning/configs/merge_method_flow_gate3_context_sampling.yaml`
- 案例簿：`results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/mechanism_assets_screened/casebooks/`
  - profile `order_boundary_fewshot_screened_v1`：train 6（Gate 1 筛后可行子集）+ support 4 + query 4（query 经 Gate 1B 单任务 SAC oracle 筛选）。
- causal audit：`pearl_learning/scripts/audit_gate3_vanilla_pearl_mechanism.py`（prior / correct / wrong / zero 四种 latent，no-gradient intervention + 四组 Stage-B 诊断）。
- 判定：顺序门禁 A→B→C，决策 shot 固定 K=4（`gate3_causal_chain_verdict()`）。

### Stage A 判据（v3，2026-08-18 升级）

Round 2 暴露出旧判据过弱：D_cw = 3.6-7.7 通过 0.5 下限，但两 posterior 几乎共线（cos 0.9997-0.99995、R_sep 0.068-0.141），绝对 L2 主要来自跨 Task 共享的 common context shift。因此 Stage A 正式升级为「Context → **Task-discriminative** Posterior」，两个 Task 都必须同时满足：

```text
D_cw  = ||mu_c - mu_w||_2                                   >= 0.5
R_sep = D_cw / (0.5 * (||mu_c - mu_p||_2 + ||mu_w - mu_p||_2) + eps)  >= 0.25
```

其中 mu_p 是 prior mean（unit-normal prior 下为 0；定义相对 prior 显式写出，Structure-Aware Prior 非零时仍成立）。cosine 只作诊断指标，不作为门槛（Task 完全可以通过 latent 幅值编码）。verdict schema 升级为 `gate3_vanilla_pearl_causal_chain_gate_v3`，写入新文件 `gate3_causal_chain_gate_v3.json`，不覆盖历史 v2 文件；Round 1/2 用同一 audit JSON 离线重算（`pearl_learning/scripts/recompute_gate3_verdict_v3.py`），0 训练步。

### v3 离线重判（Round 1 / Round 2）

```text
Round 1 (screened)   Stage A FAIL (R_sep 0.128/0.130)   B/C blocked_by_stage_a   R_policy 0.030/0.029
Round 2 (FiLM Critic) Stage A FAIL (R_sep 0.141/0.068)   B/C blocked_by_stage_a   R_policy 0.011/0.003
```

Round 2 的正确结论因此不再是「Stage A PASS、Stage B FAIL」，而是：

> **Context 使 posterior 显著变化（posterior_prior_mean_l2 ≈ 49），但主要形成跨 Task 共享的 common direction，尚未形成充分的 task-discriminative posterior separation，因此 FiLM Critic 分支走早了一步，B/C 无资格判定。**

### Stage-A 定位审计（0 训练步，Round 1 冻结 checkpoint）

脚本：`pearl_learning/scripts/audit_gate3_stage_a_diagnostics.py`；结果：`results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/gate_3_stage_a_diagnostics/gate3_stage_a_diagnostics.json`。支持轨迹用冻结 checkpoint 的 prior policy 采集一次，三个子实验只重新切分/探测，参数 hash 前后一致：

**A. Exact-PEARL-input probe**（[o, a, r/200, o', done]，按训练 sampler 抽样；逐字段与 `agent.context_tensor` 字节级一致）

```text
Gate 2（full summary + 原始 reward，P6 策略）        held-out accuracy = 1.0
exact PEARL input + 随机采样（训练实际输入）          transition 0.125-0.156，episode-majority 0.0   -> 不可辨识
exact PEARL input + linspace（旧 audit 协议）        transition 0.219-0.391，episode-majority 0-0.25
```

**B. Context sampling ablation**（同一批采集轨迹，冻结 Encoder，只换 transition 选择）

```text
scheme                adversary_first R_sep (D_cw)     sut_first R_sep (D_cw)
random（训练现状）      0.091 (4.39)                    0.030 (1.49；draw 最差 0.006/0.276)
terminal_stratified   0.158 (8.21)  +74%               0.127 (6.71)  +4x
conflict_window       0.124 (6.67)                     0.093 (5.12)
```

**C. Channel ablation**（随机采样 rows 上做 logistic probe）

```text
full 0.41 / without_reward(dynamics only) 0.42-0.45 / without_dynamics(reward+done only) 0.20-0.30
```

结论：task 信息主要存在于 dynamics 通道的**尾部（terminal/冲突附近）transitions**；训练采样器每 episode 只随机抽 8 条、且不保证包含尾部，task-discriminative evidence 被公共 transition 稀释——问题在输入侧，不在 Encoder。按预注册决策树进入**分支 A：只修 context sampler**。

### 分支 A：统一 canonical sampler + Round 3（已完成，Stage A 首次 pass）

- 新增 `pearl.replay.select_context_rows()` 为唯一 within-episode 选择函数，schemes：`random`（历史默认）/ `terminal_stratified_v1`（每 episode 固定含 1 条 terminal transition + 其余随机，短 episode 才放回）。
- **training（`sample_episode_balanced`）、few-shot evaluation（`_fixed_episode_context_block`）、causal audit（`_context_block`）现在调用同一个函数**，并由 `pearl.context_transition_sampling` 统一控制；audit provenance 记录 `context_sampling_scheme`。
- Round 3 = Round 1 Vanilla PEARL + 唯一变量 `context_transition_sampling: terminal_stratified_v1`（Dense Critic，20k，seed 1，casebook/KL/reward/prior 全部不变），输出目录 `results/pearl_learning/merge_method_flow_gate3_context_sampling/`。

### Round 3 判定结果（v3，决策 shot K=4）

```text
Stage A  context -> task-discriminative posterior   PASS（两轮后首次）
Stage B  posterior -> action                        FAIL（action L2 0.026/0.026）
Stage C  action -> outcome                          blocked_by_stage_b（oracle feasibility: pass）
Status                                              fail；passed_stages = [stage_a]
R_policy（vs 单任务 SAC 策略差异）                    0.026 / 0.026（约 2.6%）
```

采样修复在 posterior 层面完全生效：

| 指标 | Round 2（FiLM） | Round 3（Dense + 新 sampler） |
| --- | --- | --- |
| R_sep（两 task） | 0.141 / 0.068 | **0.349 / 0.346** |
| D_cw | 7.19 / 3.55 | **17.43 / 17.23** |
| cos(z_c, z_w) | 0.99974 / 0.99995 | **0.98492 / 0.98532** |
| ‖μ_c‖ vs ‖μ_w‖ | 幅值差为主 | **57.5 vs 42.3（adversary_first；sut_first 对称互换）** |

correct-posterior 沿 task-specific 方向离开 prior 更远（‖μ_c‖=57.5 vs ‖μ_w‖=42.3，两 task 对称），cos 从 0.9997 降到 0.985 —— 不再只是幅值差，方向也分化了。Stage B 仍失败：

```text
action L2 0.0258/0.0255（阈值 0.1）；pre-tanh D_raw 0.044 -> tanh 0.026（饱和 0.57-0.61，非纯饱和）
插值曲线仍水平（max pairwise 0.0105）
Q-grid 仍 D_Q = 0.0：correct/wrong 的 argmax 全在 +1.0 边界，actor regret 0.41/0.20
VCSR：adversary_first 全 0；sut_first prior 0.75 / correct 0.50 / wrong 0.50
```

预注册决策树判定：**R_sep ↑ 达成，但 Q*(z_c) ≠ Q*(z_w) 未出现（Q-grid 仍不分化）→ FiLM Actor 不满足授权条件。** 下一步选项见「下一步」。

### Query Oracle 与 Stage C casebook 修复（冻结，不再改动）

unscreened query 曾被 oracle 审计否决（sut_first 对角优势 −0.25、可实现 0/4）。8 候选经两个单任务 SAC 筛选后冻结 `order_boundary_fewshot_screened_v1`：

```text
screened query:   对角优势 adversary_first +0.5 / sut_first +0.75，可实现 2/4 与 3/4 -> feasibility pass
```

筛选脚本明确标记 `uses_test_or_ood = false`。**不再筛 query**（避免"为了让 PEARL 成功而挑 case"）。

### Split 级 casebook provenance（新增防线）

- checkpoint 现在记录 `casebook_split_hashes`（train_pool / validation_support / validation_query 分 split hash）。
- audit 强制：`train_pool` 与 `validation_support` 必须与 checkpoint 逐位一致；`validation_query` 允许不同，但必须同时携带 `query_screening_manifest_hash`（schema `gate3_query_candidate_screening_v1`、`uses_test_or_ood = false`）且 oracle feasibility = pass，否则审计直接拒绝。
- 旧 checkpoint（无 split hashes）回退到 whole-book 对比并记录 `checkpoint predates split-level casebook hashing`。

### Round 1（Dense Critic，v2 判据下的历史记录）

```text
Stage A  context -> posterior     PASS（v2 判据；v3 重判为 fail，R_sep 0.128/0.130）
Stage B  posterior -> action      FAIL
Stage C  action -> outcome        blocked_by_stage_b
```

Stage-B 四组诊断（失败定位依据）：

```text
latent geometry  cos(z_c, z_w) = 0.9972，R_sep = 0.148（posterior 几乎共线，task 信息只是幅值差）
actor pre-tanh   D_raw ≈ 0.09 -> D_tanh ≈ 0.03（tanh 压缩约 3 倍，非纯饱和）
latent interp    action 沿 z_c->z_w 方向基本水平（max step ≈ 0.008，局部 null-space）
critic Q-grid    argmax_c = argmax_w = +1.0 边界，D_Q = 0.0；Q 曲线沿动作网格 100% 单调递增
                 -> 情况 2：Critic 本身对 correct/wrong latent 不敏感（不是 Actor 的问题）
```

### Round 2（Latent-FiLM Critic，分支 B，v2 判据下的历史记录）

```text
v2 判定：Stage A PASS；Stage B FAIL（action L2 0.011/0.003）；Stage C blocked_by_stage_b
v3 重判：Stage A FAIL（R_sep 0.141/0.068）；B/C blocked_by_stage_a
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

### 失败定位（v3 后）

- **Stage A 真正失败**：posterior 位移 ≈ 49（posterior_prior_mean_l2）但 task 方向只占 7-15%（R_sep），Encoder 学到的是"有无 context"的公共方向。
- **根因在输入侧**：Gate 2 的 1.0 用的是 summary+原始 reward；PEARL 真实输入（随机 8 transitions、r/200）在 probe 下 task 信息 < chance；换 terminal 分层采样后冻结 Encoder 的 R_sep 提升 2-4x。
- **policy 学会的是"有 context vs 无 context"的行为切换，而不是"哪个 task 的 context"**。

### 下一步（待定，需要先决定、只选一个）

Round 3 后 Stage A 已真正通过（posterior 方向分化），失败点回到 Stage B 的 Critic：
Dense concat Critic 在 20k 内仍没有对 task-discriminative z 方向形成 Q 分化（Q-grid D_Q=0.0）。
候选方向（均未实施）：

```text
1) FiLM Critic 重跑（分支 B 前提现已满足）：Round 2 失败的原因是 posterior 不分化
   （z_c/z_w 同向）；现在 z_c/z_w 方向已分化（cos 0.985、D_cw 17.4、R_sep 0.35），
   重跑 Round 3 + Latent-FiLM Critic（单变量 = Critic 结构），20k seed 1
2) 0 步 per-transition Gaussian evidence precision 审计：terminal/task-sensitive transition
   贡献多少 precision，common evidence 是否在 PoG 中仍压倒 task-specific evidence
3) 停止并复盘（写 Round 1-3 完整机制结论）
```

不进入 Gate 4、不恢复真实物理 Merge Task、不做多模型并行搜索、不扩大预算。FiLM Actor 需先出现 Q*(z_c) ≠ Q*(z_w)（当前未满足）。

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
query casebook（order_boundary_fewshot_screened_v1，已冻结不再筛）
```
