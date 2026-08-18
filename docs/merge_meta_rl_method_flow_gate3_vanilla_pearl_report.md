# Gate 3 Vanilla PEARL 因果链报告（Round 1 / Round 2 / Round 3）

日期：2026-08-18。该记录是机制门禁证据，不是论文性能结果。

## 配置与协议

- 任务：物理完全匹配的 `adversary_first` / `sut_first` 对；仅冻结的 conflict-entry order 不同。
- 算法：Vanilla PEARL（Dense Actor，unit-normal prior，Scenario Encoder / MoE OFF），latent dim 5。
- 训练：每轮 2 tasks × 20k 环境步 × seed 1，meta-batch 2。只验证机制，不追求性能。
- 案例簿：`order_boundary_fewshot_screened_v1` —— train 6（Gate 1 筛后可行子集）+ support 4 + query 4（query 经 oracle 筛选）。
- strict 指标：`logical_order_spatiotemporal_near_miss_v3`，calibration 与 Gate 1/1B/2 同一 manifest。
- 训练入口对 v2/v3 统一强制 `--critical-thresholds`（`resolve_calibration`）。
- 判定：顺序门禁 A→B→C，决策 shot 固定 K=4，verdict schema v3（`gate3_causal_chain_gate_v3.json`）。

证据目录：

```text
Round 1 配置   pearl_learning/configs/merge_method_flow_gate3_vanilla_pearl.yaml
Round 2 配置   pearl_learning/configs/merge_method_flow_gate3_film_critic.yaml
Round 3 配置   pearl_learning/configs/merge_method_flow_gate3_context_sampling.yaml
案例簿         results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/mechanism_assets_screened/casebooks/
Round 1 训练   results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/mechanism_gate/gate3_vanilla_pearl_20k_seed1/
Round 2 训练   results/pearl_learning/merge_method_flow_gate3_film_critic/mechanism_gate/gate3_film_critic_20k_seed1/
Round 3 训练   results/pearl_learning/merge_method_flow_gate3_context_sampling/mechanism_gate/gate3_context_sampling_20k_seed1/
oracle 审计    results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/gate_3_query_oracle_screened/
Stage-A 诊断   results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/gate_3_stage_a_diagnostics/
```

## Stage A 判据升级（v3）

Round 2 之后发现旧 Stage A 判据（D_cw ≥ 0.5）不充分：D_cw = 3.6-7.7 通过，但 cos(z_c,z_w) = 0.9997-0.99995、R_sep = 0.068-0.141 —— 绝对 L2 主要来自跨 Task 共享的 common context shift。Stage A 正式升级为「Context → Task-discriminative Posterior」，两个 Task 都必须同时满足：

```text
D_cw  = ||mu_c - mu_w||_2 >= 0.5
R_sep = D_cw / (0.5 * (||mu_c - mu_p||_2 + ||mu_w - mu_p||_2) + eps) >= 0.25
```

mu_p = prior mean（unit-normal 下为 0；显式相对 prior 的定义在 Structure-Aware Prior 非零时仍成立）。cosine 只作诊断。Round 1/2 的 v3 verdict 由 `recompute_gate3_verdict_v3.py` 用同一 audit JSON 离线重算（0 训练步），写入 `gate3_causal_chain_gate_v3.json`，历史 v2 文件不覆盖。

## v3 离线重判

| 轮次 | Stage A | Stage B | Stage C | R_policy |
| --- | --- | --- | --- | --- |
| Round 1 (Dense, screened) | **FAIL**（R_sep 0.128/0.130） | blocked_by_stage_a | blocked_by_stage_a | 0.030 / 0.029 |
| Round 2 (FiLM Critic) | **FAIL**（R_sep 0.141/0.068） | blocked_by_stage_a | blocked_by_stage_a | 0.011 / 0.003 |

Round 2 的正确结论（替代旧的「A PASS、B FAIL」）：

> **Context 使 posterior 显著变化（posterior_prior_mean_l2 ≈ 49），但主要形成跨 Task 共享的 common direction，尚未形成充分的 task-discriminative posterior separation —— FiLM Critic 分支走早了一步，B/C 无资格判定。**

## Stage-A 定位审计（0 训练步）

脚本 `audit_gate3_stage_a_diagnostics.py`，基于 Round 1 冻结 checkpoint；支持轨迹用冻结 prior policy 采集一次，参数 hash 前后一致，三个子实验：

**A. Exact-PEARL-input probe**（[o, a, r/200, o', done]，按训练 sampler 抽样，与 `agent.context_tensor` 逐字节一致）

```text
Gate 2（full summary + 原始 reward，P6 策略）          held-out accuracy = 1.0
exact PEARL input + random（训练实际输入）             transition 0.125-0.156，episode-majority 0.0  -> 不可辨识
exact PEARL input + linspace（旧 audit 协议）          transition 0.219-0.391，episode-majority 0-0.25
```

**B. Sampling ablation**（同一批轨迹，冻结 Encoder，只换 transition 选择；3 draws）

```text
scheme                adversary_first R_sep (D_cw)      sut_first R_sep (D_cw)
random（训练现状）      0.091 (4.39)                     0.030 (1.49；最差 draw 0.006/0.276)
terminal_stratified   0.158 (8.21)                     0.127 (6.71)
conflict_window       0.124 (6.67)                     0.093 (5.12)
```

**C. Channel ablation**（random rows 上 probe）：full 0.41 / without_reward 0.42-0.45 / without_dynamics 0.20-0.30 —— task 信息主要在 dynamics 通道的尾部 transitions。

**结论：输入侧丢失 Task 信息（Gate 2 数据 ≠ 训练数据 ≠ 旧 audit 数据），问题还没到 Encoder。按预注册决策树进入分支 A：只修 context sampler。**

## Round 3（分支 A，已完成）：Stage A 首次通过 v3

Round 1 Vanilla PEARL + 唯一变量 `pearl.context_transition_sampling: terminal_stratified_v1`（每 episode 固定 1 条 terminal + 7 随机；Dense Critic；20k；seed 1；casebook/KL/reward/prior 不变）。

配套工程改动：

- `pearl_learning/src/replay.py::select_context_rows()` 为唯一 within-episode 选择函数；training / few-shot evaluation / causal audit 全部调用同一函数，由 `pearl.context_transition_sampling` 统一控制。
- checkpoint 新增 `casebook_split_hashes`；audit 强制 train_pool/validation_support 与 checkpoint 逐位一致，query 改动必须携带 screening manifest + oracle feasibility pass（`verify_casebook_split_provenance`）。
- 停止条件：先只看 R_sep(K=4)；< 0.25 立即停止。结果 R_sep = 0.349/0.346 ≥ 0.25，继续 B/C 判定。

### Round 3 判定（v3，K=4）

```text
Stage A  PASS   R_sep 0.349/0.346；D_cw 17.43/17.23；cos 0.98492/0.98532
Stage B  FAIL   action L2 0.0258/0.0255（阈值 0.1）；Q-grid 仍 D_Q=0.0
Stage C  blocked_by_stage_b（oracle feasibility: pass）
R_policy 0.026 / 0.026
```

采样修复在 posterior 层面完全生效：R_sep 0.07-0.14 → 0.35，D_cw 3.6-7.7 → 17.4，cos 0.9997 → 0.985；correct-posterior 沿 task-specific 方向离开 prior 更远（‖μ_c‖=57.5 vs ‖μ_w‖=42.3，两 task 对称互换）——不再是幅值差，方向也分化了。Stage B 仍失败：pre-tanh 0.044 → tanh 0.026（非纯饱和），插值水平，Q-grid argmax 全在 +1.0 边界（correct/wrong 一致），actor regret 0.41/0.20。VCSR：adversary_first 全 0；sut_first prior 0.75 / correct 0.50。

预注册决策树判定：**R_sep ↑ 达成，但 Q*(z_c) ≠ Q*(z_w) 未出现 → FiLM Actor 不满足授权条件。**

## Round 1 / Round 2 历史记录（v2 判据）

Round 1：Stage-B 诊断 cos 0.9972、pre-tanh D_raw 0.09→tanh 0.03、插值水平、Q-grid D_Q=0.0 且 Q 100% 单调 → Critic 不敏感（情况 2）。Round 2（FiLM Critic）：adversary_first correct VCSR 4/4、q_loss 117k→2.6k、critic_latent_gradient_norm 2.6→285，但 Q-grid 仍 D_Q=0.0、posterior 更共线（cos 0.9997+）。详见 `merge_meta_rl_method_flow_current_status.md`。

## 下一步（待定，只选一个）

```text
1) FiLM Critic 重跑（分支 B 前提现已满足）：Round 2 失败因 posterior 不分化；现在
   z_c/z_w 方向已分化，重跑 Round 3 + Latent-FiLM Critic（单变量 = Critic 结构）
2) 0 步 per-transition Gaussian evidence precision 审计（terminal 贡献多少 precision）
3) 停止并复盘
```

FiLM Actor 需先出现 Q*(z_c) ≠ Q*(z_w)（当前未满足）。不进入 Gate 4、不恢复真实物理 Merge Task、不做多模型并行搜索、不扩大预算。
