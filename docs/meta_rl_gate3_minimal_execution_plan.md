# Meta-RL Gate 3 最小闭环执行计划（v4）

> 目的：在不扩大方法、数据规模或训练预算的前提下，完成
> `support → posterior → Critic → Actor → query outcome` 的最小机制闭环定位。
>
> 本文是新的执行规划；不覆盖历史 v2/v3 Gate 文件，也不替换
> `merge_meta_rl_method_flow_current_status.md` 或此前的修订计划。

## 1. 当前基线与范围

截至 Round 3，唯一已被严格证明的链路是：

```text
support trajectory
  → terminal_stratified_v1 context sampling
  → task-discriminative posterior
```

Round 3 在 K=4 的两个 mechanism Task 上均通过 Stage A：

```text
R_sep = 0.349 / 0.346
D_cw  = 17.43 / 17.23
```

但现有 Q-grid 的最优动作仍相同（`D_Q=0`），Actor correct/wrong
action L2 仅为 `0.0258 / 0.0255`。因此当前不能宣称完整 few-shot
policy adaptation 成立。

本轮只允许两类工作：

1. 将 Gate 3 判定升级为可区分 Critic 与 Actor 的 v4，并以 0 环境步重判 Round 1–3。
2. 仅运行一次 Round 3 sampler + Latent-FiLM Critic 的 20k、seed 1 机制实验。

以下内容保持冻结：Task 定义、IDM、arrival controller、reward、screened
casebook/query、Context Encoder、PoG、latent dim、KL beta、学习率、Dense
Actor、Scenario Encoder、Structure-Aware Prior、MoE 及训练预算。

## 2. Gate 3 v4 判定契约

新增 verdict schema：

```text
gate3_vanilla_pearl_causal_chain_gate_v4
```

新增输出文件：

```text
gate3_causal_chain_gate_v4.json
```

v4 必须严格按下列顺序判定，决策 shot 固定为 K=4：

| 阶段 | JSON 键 | 通过标准 | 未通过时后续状态 |
| --- | --- | --- | --- |
| A | `stage_a_context_to_posterior` | 两个 Task 均满足 `D_cw ≥ 0.5` 且 `R_sep ≥ 0.25` | `blocked_by_stage_a` |
| B_Q | `stage_b_q_posterior_to_critic_action_preference` | 两个 Task 均满足 `D_Q^action ≥ 0.10` | `blocked_by_stage_b_q` |
| B_π | `stage_b_pi_critic_to_actor_action` | 两个 Task均满足 deterministic correct/wrong action L2 `≥ 0.10` | `blocked_by_stage_b_pi` |
| C | `stage_c_actor_to_outcome` | oracle feasibility PASS，且至少一个 Task 满足 correct-context strict VCSR 严格大于 wrong-context 且 correct VCSR 大于零 | `blocked_by_query_feasibility` 或对应上游阻塞 |

其中：

\[
a_c^*(s)=\arg\max_a Q(s,a,z_c),\qquad
a_w^*(s)=\arg\max_a Q(s,a,z_w)
\]

\[
D_Q^{action}=\mathbb E_{s\in\mathcal S_{audit}}
\left[\left\|a_c^*(s)-a_w^*(s)\right\|_2\right]
\]

`D_Q^action` 直接使用既有 Stage-B diagnostics 的
`critic_q_grid.argmax_action_distance_mean`。审计固定沿用当前 41 点动作网格和
state bank；`0.10` 等于两个网格步长，必须在两个 Task 上同时达到，不能按新结果事后调低。

Stage C 的 paired return 保留为诊断字段，不作为额外通过门槛。`passed_stages`
固定使用 `stage_a`、`stage_b_q`、`stage_b_pi`、`stage_c`；所有阶段通过后，才将
`next_allowed_stage` 设为 Gate 4 的规划资格。

### 实现边界

- 保留现有 `gate3_causal_chain_verdict()` 及 `recompute_gate3_verdict_v3.py` 的 v3 行为。
- 在 `pearl_learning/scripts/audit_gate3_vanilla_pearl_mechanism.py` 中新增独立的
  `gate3_causal_chain_verdict_v4()`，由新的在线 audit 输出 v4 文件。
- 新增 `pearl_learning/scripts/recompute_gate3_verdict_v4.py`；它只读取已有 audit suite，
  写入 v4 verdict，绝不覆盖 v2/v3 文件。

## 3. 第一步：0 环境步离线重判

使用现有 `gate3_causal_audit.json` 和已冻结 oracle audit 生成 v4 结果。该步骤不得加载
环境、不得生成 rollout、不得更新 checkpoint；每份 v4 文件应记录原 audit hash、oracle hash、
原 provenance、`environment_steps: 0` 与 `training_updates: 0`。

```powershell
conda run -n metadrive python -m pearl_learning.scripts.recompute_gate3_verdict_v4 `
  --suite results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/gate_3_causal_audit_screened/gate3_causal_audit.json `
  --suite results/pearl_learning/merge_method_flow_gate3_film_critic/gate_3_causal_audit/gate3_causal_audit.json `
  --suite results/pearl_learning/merge_method_flow_gate3_context_sampling/gate_3_causal_audit/gate3_causal_audit.json `
  --oracle-audit results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/gate_3_query_oracle_screened/gate3_query_oracle_audit.json
```

预期结果如下：

| 轮次 | Stage A | Stage B_Q | Stage B_π | Stage C |
| --- | --- | --- | --- | --- |
| Round 1（screened Dense） | FAIL | `blocked_by_stage_a` | `blocked_by_stage_a` | `blocked_by_stage_a` |
| Round 2（FiLM Critic） | FAIL | `blocked_by_stage_a` | `blocked_by_stage_a` | `blocked_by_stage_a` |
| Round 3（terminal sampler + Dense Critic） | PASS | FAIL，`D_Q=0` | `blocked_by_stage_b_q` | `blocked_by_stage_b_q` |

若离线结果偏离此表，先检查 v4 的输入字段映射和历史 audit provenance，不启动新训练。

## 4. 第二步：唯一允许的新训练

新增配置：

```text
pearl_learning/configs/merge_method_flow_gate3_context_sampling_film_critic.yaml
```

该配置继承 `merge_method_flow_gate3_context_sampling.yaml`，且只覆盖：

```yaml
project:
  output_root: results/pearl_learning/merge_method_flow_gate3_context_sampling_film_critic

experiment:
  method_variant: vanilla_gate3_context_sampling_film_critic

networks:
  critic_architecture: latent_film_dense
```

训练前，解析新配置和 Round 3 配置并比较 training-relevant sections；除了
`networks.critic_architecture` 外不得有差异。`project`、`experiment.method_variant` 等
provenance/output 字段可以不同。确认 CUDA 可用后，运行唯一一次训练：

```powershell
conda run -n metadrive python -c "import torch; assert torch.cuda.is_available(), 'CUDA is required'; print(torch.cuda.get_device_name(0))"

conda run -n metadrive python -m pearl_learning.scripts.train_pearl `
  --config pearl_learning/configs/merge_method_flow_gate3_context_sampling_film_critic.yaml `
  --taskbook results/pearl_learning/merge_method_flow_logical_order_interpolated/taskbooks `
  --casebook-root results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/mechanism_assets_screened `
  --critical-thresholds results/pearl_learning/merge_method_flow_pilot/v2_assets/calibration/critical_thresholds.json `
  --seed 1 `
  --max-env-steps 20000 `
  --run-name gate3_context_sampling_film_critic_20k_seed1 `
  --mechanism-gate
```

期望训练目录：

```text
results/pearl_learning/merge_method_flow_gate3_context_sampling_film_critic/
  mechanism_gate/gate3_context_sampling_film_critic_20k_seed1/
```

训练脚本以 episode 边界停止，`training_summary.json` 的实际环境步数可略高于 20k；
这不代表扩大预算，命令行预算仍固定为 `--max-env-steps 20000`。

训练完成后使用同一 casebook、taskbook、calibration 和 screened oracle 做 v4 audit：

```powershell
conda run -n metadrive python -m pearl_learning.scripts.audit_gate3_vanilla_pearl_mechanism `
  --config pearl_learning/configs/merge_method_flow_gate3_context_sampling_film_critic.yaml `
  --checkpoint results/pearl_learning/merge_method_flow_gate3_context_sampling_film_critic/mechanism_gate/gate3_context_sampling_film_critic_20k_seed1/best_model.pt `
  --taskbook results/pearl_learning/merge_method_flow_logical_order_interpolated/taskbooks `
  --casebook-root results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/mechanism_assets_screened `
  --critical-thresholds results/pearl_learning/merge_method_flow_pilot/v2_assets/calibration/critical_thresholds.json `
  --output results/pearl_learning/merge_method_flow_gate3_context_sampling_film_critic/gate_3_causal_audit `
  --task-id meta_train_lane_drop_24__logical_order_adversary_first `
  --task-id meta_train_lane_drop_24__logical_order_sut_first `
  --shots 1 2 4 `
  --oracle-audit results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/gate_3_query_oracle_screened/gate3_query_oracle_audit.json
```

## 5. 结果决策与停止条件

只根据 v4 中第一个失败阶段决策，不做并行试验：

| 首个失败阶段 | 本轮结论 | 下一步授权 | 当前禁止项 |
| --- | --- | --- | --- |
| A | FiLM Critic 的梯度破坏了 task-discriminative representation，B_Q/B_π/C 均不可判定 | 单独定位 representation/gradient regression | 判断 Actor、修改 replay、增加预算 |
| B_Q | posterior 已分化，但 Critic 尚未产生 task-dependent action preference | 从 checkpoint 的 `trainer_state.buffers` 做 0 环境步 RL replay signal 审计 | 直接改 replay sampler 或 Actor |
| B_π | Critic 已分化，但 Dense Actor 未利用价值偏好 | 下一轮才允许单变量 Latent-FiLM Actor 方案 | MoE Actor、多变量改动 |
| C | support→posterior→Critic→Actor 已成立，但未形成 query 收益 | 单独审查 actor adaptation magnitude、query 难度和策略质量 | 自动扩大预算或重筛 query |
| 无失败 | Vanilla PEARL 的最小 few-shot adaptation 闭环初步成立 | 可规划 Gate 4 | 本轮直接运行 Gate 4 |

当 B_Q 失败时的 replay audit 必须是 0 环境步、只读 checkpoint 的 replay buffers，至少报告：

```text
terminal transition 占比
conflict-near transition 占比
task-sensitive TD/reward/dynamics signal 的分布
context episode 排除后 RL batch 的上述占比
```

实现入口为 `pearl_learning/scripts/audit_gate3_critic_replay_signal.py`。它以 checkpoint
自带的 `config_resolved.json` 和 `trainer_state.buffers` 为唯一输入，使用独立的确定性 RNG
重复训练时的 context 抽样及 context-episode 排除规则；不创建环境、不执行梯度更新。由于单条
replay transition 没有 counterfactual task label，`task_sensitive_proxy_rate` 明确只定义为
`terminal OR public-dynamics conflict-near`，同时保留各 stratum 的 reward、arrival-gap、
distance-to-conflict 和 TTC 分布，不能将该 proxy rate 误报为 task identifiability。

```powershell
conda run -n metadrive python -m pearl_learning.scripts.audit_gate3_critic_replay_signal `
  --checkpoint results/pearl_learning/merge_method_flow_gate3_context_sampling_film_critic/mechanism_gate/gate3_context_sampling_film_critic_20k_seed1/best_model.pt `
  --output results/pearl_learning/merge_method_flow_gate3_context_sampling_film_critic/gate_3_causal_audit/gate3_critic_replay_signal_audit.json `
  --sampled-batches 128
```

在该审计完成前，不允许改变 RL replay distribution。

## 6. 验证清单

在任何训练前完成以下验证：

1. 扩展 `pearl_learning/tests/test_method_flow_v2.py`：覆盖 A/B_Q/B_π/C 顺序阻塞、两个 Task
   同时通过要求、`D_Q=0.10` 边界、oracle 阻塞及完整 PASS。
2. 验证 v3 函数、schema、`recompute_gate3_verdict_v3.py` 及既有 v3 JSON 不变。
3. 验证新配置相对 Round 3 的训练相关差异仅为 `critic_architecture`。
4. 验证 v4 重判只新增 `gate3_causal_chain_gate_v4.json`，不改写 audit suite。

```powershell
conda run -n metadrive python -m unittest pearl_learning.tests.test_method_flow_v2
```

只有测试全部通过、配置差异检查通过、离线 v4 重判与预期一致后，才允许启动第 4 节中的单次 GPU 训练。
