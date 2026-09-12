# DIVA-Former 第一阶段训练与验收需求

**文档用途**：作为 `SafeDL/META_LEARNING` 下一阶段的直接执行目标与停止规则。  
**研究主线**：Belief-Space Regression Testing + DIVA-Former + Counterfactual Test-Utility Distillation。  
**适用仓库**：`https://github.com/SafeDL/META_LEARNING`  
**设计基线快照**：`main`，参考提交 `02342c06a69c4cdd5de30bf02e82dd502c481cae`（2026-09-11）。  
**日期**：2026-09-12。  
**阶段性质**：**source-only、offline-first、零新增 target rollout 的机制验证阶段**。  

---

## 0. 一句话目标

第一阶段不追求直接在 `validation/test SUT` 上证明最终 failure-mining 收益，而是首先回答三个必须成立的问题：

1. **Belief inference 是否成立**：一个小型 AI 模型能否只根据少量 `(scenario, outcome)` context，在不输入 SUT identity、不做在线梯度更新的情况下恢复 held-out SUT 的 vulnerability profile；
2. **Counterfactual test utility 是否有意义**：历史 source bank 是否能够离线产生“如果下一次测试这个场景，最终固定预算内能多发现多少 failure”的非平凡监督信号；
3. **Utility distillation 是否有效**：DIVA-Former 能否把昂贵的离线 belief-space lookahead teacher 蒸馏成一次前向传播的测试决策器，并在 source-only LOSO 中优于或至少稳定不劣于当前 DIVA heuristic。

只有三点同时成立，才允许进入真正的 unseen validation SUT。

---

# 1. 当前代码基础与本阶段定位

当前 DIVA v2 已经提供本阶段所需的完整物理与统计基础，不允许为了 DIVA-Former 随意重写已有 scenario/failure contract。

当前可直接复用的核心资产包括：

- `mvr/metadrive/diva/types.py`
  - `DivaCutInDesign`
  - `DivaObservation`
  - `score`
  - `vulnerability_response`
- `mvr/metadrive/diva/response.py`
  - 连续 vulnerability learning signal
  - formal score 与 learning response 已解耦
- `mvr/metadrive/diva/source_bank.py`
  - 对齐的 source-SUT × common-anchor response bank
- `mvr/metadrive/diva/factorization.py`
  - source vulnerability 的低秩结构
- `mvr/metadrive/diva/prior.py`
  - `LowRankVulnerabilityPrior`
- `mvr/metadrive/diva/posterior.py`
  - `LatentVulnerabilityPosterior`
- `mvr/metadrive/diva/acquisition.py`
  - 当前 `diagnostic_scores()` 与 `mining_scores()`
- `mvr/metadrive/diva/miner.py`
  - 当前 DIVA 的在线选择状态机
- `mvr/metadrive/scripts/evaluate_diva_source_loso.py`
  - source-only LOSO protocol
- `mvr/metadrive/scripts/evaluate_diva_cutin.py`
  - target 预算协议和物理执行入口
- `mvr/metadrive/evaluation/diva_protocol.py`
  - fixed-budget ledger
- `mvr/metadrive/configs/diva_cutin.yaml`
  - 当前冻结的 Cut-in source domain 与 B=20 协议

当前 source v2 已经得到：

- 256 个 source observations；
- formal valid rate = 1.0；
- formal event rate ≈ 8.20%；
- rank-2 cumulative explained variance ≈ 0.977；
- source-only G0 通过；
- source-only G1 通过；
- K=4 diagnostic support 的 NDCG@8 ≈ 0.968；
- diagnostic vs random K=4 的 paired NDCG 95% CI 为正。

**因此本阶段不再重复证明“low-rank DIVA 是否可行”，而是把它降级为：**

> **解析 teacher + strong baseline + synthetic-task prior generator。**

最终学生模型才是 DIVA-Former。

---

# 2. 第一阶段明确不做什么

为避免重新陷入大规模训练但机制不可归因的问题，本阶段明确禁止：

1. **禁止访问 `idm_fast_small_gap` validation SUT 的任何 rollout**；
2. **禁止访问 `idm_late_response` test SUT 的任何 rollout**；
3. 禁止重新训练 SAC / PPO / PEARL；
4. 禁止修改正式 failure oracle：
   - valid collision = `1.0`
   - valid critical near-miss = `0.5`
   - otherwise = `0.0`
5. 禁止为了让 Transformer 结果更好而改变 Cut-in 物理场景合同；
6. 禁止把 `sut_ref`、controller profile 参数或显式 SUT identity 输入 DIVA-Former；
7. 禁止在 held-out source fold 上做模型选择、early stopping 或超参数调节；
8. 禁止第一阶段直接加入大模型、视觉输入或 CARLA；
9. 禁止以 training loss、latent 可视化、attention 图作为通过标准；
10. 禁止因为某个 gate 失败而直接增加网络层数或训练时长。

**第一阶段新增 MetaDrive 调用预算：0。**

允许使用的真实数据只有已经冻结的 source bank：

`results/metadrive/diva/cutin_g01/source_observations_v2.jsonl`

---

# 3. 新问题形式化：Belief-Space Regression Testing

## 3.1 状态

在第 `t` 次测试后，当前信息为：

\[
D_t=\{(x_i,y_i,r_i)\}_{i=1}^{t},
\]

其中：

- `x_i`：concrete/logical scenario；
- `y_i`：连续 `vulnerability_response`；
- `r_i`：formal score ∈ {0, 0.5, 1}。

新方法不再把状态定义为“哪些 case 还没跑”，而定义为：

> **根据当前少量测试结果，对 unseen SUT vulnerability function 的 belief。**

DIVA-Former 使用 context set `D_t` 隐式表达这一 belief。

## 3.2 动作

动作是从未测试候选池中选择一个 scenario：

\[
x_t \in \mathcal X \setminus D_t.
\]

## 3.3 reward

第一阶段保留两个并行 reward。

### Formal reward

\[
r_t \in \{0,0.5,1\}.
\]

这是最终 failure-mining 的唯一正式成功标准。

### Dense auxiliary reward

\[
v_t = vulnerability\_response(x_t)\in[0,1].
\]

它只用于 belief learning、counterfactual teacher 的 dense auxiliary target 与 representation learning。

**不得用 `v_t` 替换最终 formal reward。**

## 3.4 有限预算目标

总预算仍冻结为：

\[
B=20.
\]

最终目标：

\[
\max_\pi \mathbb E_\pi
\left[
\sum_{t=1}^{B} r_t
\right].
\]

关键变化：不再预先指定 K 个 support，再进入 mining。

每一步统一为：

\[
x_t=\arg\max_x Q(x\mid D_t,B-t).
\]

同一个场景在剩余预算不同的时候可以有完全不同的测试价值。

---

# 4. 第一阶段新增代码结构

不得破坏 `mvr/metadrive/diva/` 当前 v2 baseline。建议新增独立包：

```text
mvr/metadrive/diva_ai/
    __init__.py
    state.py
    scenario_encoder.py
    outcome_encoder.py
    belief_encoder.py
    vulnerability_decoder.py
    value_head.py
    model.py

    counterfactual_teacher.py
    counterfactual_dataset.py
    prior_task_generator.py

    losses.py
    policy.py
    metrics.py
```

新增脚本：

```text
mvr/metadrive/scripts/build_diva_counterfactual_dataset.py
mvr/metadrive/scripts/pretrain_diva_former.py
mvr/metadrive/scripts/train_diva_former.py
mvr/metadrive/scripts/evaluate_diva_former_source_loso.py
mvr/metadrive/scripts/validate_diva_former_stage1.py
```

新增测试：

```text
mvr/tests/test_diva_ai_context_invariance.py
mvr/tests/test_diva_ai_no_sut_identity.py
mvr/tests/test_diva_counterfactual_teacher.py
mvr/tests/test_diva_counterfactual_dataset.py
mvr/tests/test_diva_former_forward.py
mvr/tests/test_diva_former_budget_conditioning.py
mvr/tests/test_diva_former_policy_budget.py
mvr/tests/test_diva_former_source_loso_contract.py
```

建议单独新建配置：

```text
mvr/metadrive/configs/diva_former_stage1.yaml
```

不要把 AI 实验超参数写回冻结的 `diva_cutin.yaml`。

---

# 5. DIVA-Former 第一版模型设计

第一版目标不是追求大模型，而是证明：

> **in-context belief inference + budget-conditioned decision value 可以成立。**

建议参数量控制在 **3M 以下**。

## 5.1 Scenario Encoder

输入保持当前 Cut-in 物理定义：

\[
x=[initial\_gap, ego\_speed, relative\_speed, cutin\_start, cutin\_path\_length].
\]

所有连续参数统一映射至 `[0,1]`。

离散 `candidate_index ∈ {0,1}` 使用 embedding。

推荐：

```text
continuous 5-d
      ↓
MLP(5 → 64 → 128)

candidate_index
      ↓
Embedding(2 → 16)
      ↓
Linear(16 → 128)

两者相加/拼接后投影
      ↓
scenario token: 128-d
```

**禁止输入 `sut_ref`。**

## 5.2 Outcome Encoder

一次已经执行的测试应编码：

```text
vulnerability_response
formal_score
event class
posterior_eligible
```

event class 建议固定：

```text
0 = completed_noncritical
1 = valid_critical_near_miss
2 = valid_target_collision
```

最终生成 128-d outcome token。

Context token：

\[
c_i=f_{\text{context}}(e_{x_i},e_{y_i}).
\]

通俗理解：**“给 Agent 出了什么测试题 + Agent 怎么回答”。**

## 5.3 Belief Encoder

输入：

\[
\{c_1,\ldots,c_t\}.
\]

推荐第一版：

- Transformer Encoder layers = 4；
- `d_model = 128`；
- heads = 4；
- FFN width = 256；
- dropout = 0.1；
- **不使用普通位置编码**；
- 增加一个 learned `[BELIEF]` token；
- 最大 context length = 20。

必须满足：同一组 context 改变输入顺序，不应改变最终预测。

输出：

```text
belief_token: 128-d
context_memory: t × 128
```

`belief_token` 可以被解释为当前新 SUT 的隐式 safety fingerprint。

## 5.4 Candidate Cross-Attention

候选场景不与全部 8192 candidates 做 self-attention。

正确结构：

```text
context <= 20 tokens
candidate batch = 512 / 1024
```

每个 candidate query 只 cross-attend context memory。

复杂度约：

\[
O(N_{candidate}\times N_{context})
\]

而不是：

\[
O(N_{candidate}^2).
\]

这样 Stage 2 时可以直接扩展到当前 8192 candidate pool。

## 5.5 Vulnerability Head

第一版输出：

```text
predicted vulnerability mean
predicted vulnerability uncertainty
```

推荐用 bounded distribution；可采用 Beta head：

\[
\alpha(x),\beta(x)>0.
\]

训练时对 0 和 1 使用 epsilon clamp，例如：

```text
eps = 1e-4
y_beta = y * (1 - 2*eps) + eps
```

至少必须可以输出：

\[
\mu_v(x),\sigma_v^2(x).
\]

## 5.6 Formal Event Head

额外输出：

\[
P(normal),P(near\ miss),P(collision).
\]

Immediate formal utility：

\[
U_{immediate}(x)=0.5P(near\ miss)+P(collision).
\]

事件类别不平衡时允许使用 class weight 或 focal loss，但权重必须仅由 training fold 统计产生。

## 5.7 Decision Value Head

这是第一阶段最重要的新增模块。

输入至少包括：

```text
candidate representation
belief token
predicted immediate formal utility
vulnerability uncertainty
remaining_budget / total_budget
evaluability
novelty
```

输出两个 value：

```text
Q_formal
Q_vulnerability
```

其中 `Q_formal` 表示：如果现在测试该 candidate，从当前直到 B 次预算耗尽，teacher 预计能取得的累计 formal critical score。

`Q_vulnerability` 是 dense auxiliary value。

**在线选景只允许以 `Q_formal` 为主目标；`Q_vulnerability` 只做辅助训练或稳定 tie-breaking。**

---

# 6. Counterfactual Test-Utility Teacher

## 6.1 为什么需要 teacher

真实测试只能执行一个 next scenario，因此真实系统无法告诉模型：

> “如果刚才选择另一个测试，后面 19 次会不会更好？”

但 source bank 对 64 个 common anchors 有完整已知 response，因此可以离线构造大量反事实：

```text
如果下一步选 A → 最终收益多少
如果下一步选 B → 最终收益多少
如果下一步选 C → 最终收益多少
```

这就是 Counterfactual Test-Utility Distillation 的监督来源。

## 6.2 Teacher 的状态

定义：

```text
TeacherState:
    history_design_ids
    history_vulnerability_responses
    history_formal_scores
    analytic_posterior
    selected_ids
    failure_archive
    remaining_budget
```

需要新增可复制/可重放的 belief state，禁止直接依赖不可复制的 `DivaMiner` mutable instance。

建议 `mvr/metadrive/diva_ai/state.py` 提供：

```python
clone()
observe(...)
available_mask(...)
```

## 6.3 Teacher 的单步 Q

对当前 state `s` 和候选 `x`：

1. 从历史 source bank 查询 `x` 的真实 response；
2. 得到 immediate formal reward `r(x)`；
3. 使用 `vulnerability_response(x)` 更新现有解析 posterior；
4. 得到新 belief state `s'`；
5. 从 `s'` 开始模拟剩余预算。

定义：

\[
Q_T^F(s,x)=r(x)+\max_{\pi\in\Pi}G_F(\pi,s',b-1),
\]

以及：

\[
Q_T^V(s,x)=v(x)+\max_{\pi\in\Pi}G_V(\pi,s',b-1).
\]

## 6.4 Teacher continuation portfolio

第一版不训练 RL teacher。

固定使用以下 3 个 continuation policy。

### P0 — Mine Now

直接使用当前 `mining_scores()` 直到预算结束。

### P1 — Diagnose Once Then Mine

下一步使用 `diagnostic_scores()` 1 次，再全部 mining。

### P2 — Diagnose Twice Then Mine

最多额外诊断 2 次，再全部 mining。

Teacher 取三种 continuation 中真实累计收益最大的一个。

因此 teacher 并不声称是全局最优 POMDP solver，但比单一 heuristic 强，并且明确表达：

> “现在继续诊断还是直接挖掘”的 downstream value。

## 6.5 必须满足的 teacher 恒等式

当 `b=1` 时，没有未来。因此必须严格满足：

\[
Q_T^F(s,x)=r(x),
\]

\[
Q_T^V(s,x)=v(x).
\]

这是 mandatory unit test。

---

# 7. Counterfactual Dataset 构造协议

## 7.1 必须采用 source LOSO

四个 source：

```text
idm_cautious
idm_defensive
idm_normal
idm_assertive
```

每个 fold：

```text
3 source SUT → train/teacher/prior
1 source SUT → held-out evaluation only
```

held-out SUT 的任何 response 不允许进入：

- prior；
- synthetic-task generator；
- teacher training dataset；
- early stopping；
- threshold tuning。

## 7.2 Context 状态采样

每个 training SUT 生成不少于：

```text
2,000 partial states / fold / training SUT
```

即每 fold 至少 6,000 states。

context size 从：

```text
t ∈ {0, 1, 2, 4, 8, 12}
```

采样。

remaining budget 从：

```text
b ∈ {1, 2, 4, 8, 12, 20-t}
```

中合法采样。

history 中 scenario 不得重复。

## 7.3 Candidate actions

Stage 1 teacher 在已有 64 common anchors 上计算。

在每个 state 中，优先对**全部未测试 anchors**计算 Q，而不是只计算当前 heuristic 的 top-k。

若 CPU 时间超预算，可降级为：

```text
all high-risk top 16
+ all diagnostic top 16
+ random 16
```

但最终每 state 至少 32 个 candidate labels。

## 7.4 数据记录格式

每条 state-action label 至少包含：

```json
{
  "fold": 0,
  "source_sut": "...",
  "history": [],
  "candidate_design_id": "...",
  "remaining_budget": 12,
  "teacher_q_formal": 4.5,
  "teacher_q_vulnerability": 10.81,
  "teacher_best_continuation": "diagnose1_then_mine",
  "immediate_formal_score": 0.0,
  "immediate_vulnerability_response": 0.61
}
```

`source_sut` 只作为 provenance 和分组字段，不允许送入模型。

---

# 8. Prior-Fitted Pretraining：第一阶段采用轻量版本

四个真实 source SUT 太少，不适合直接从头训练 Transformer。

因此当前 `LowRankVulnerabilityPrior` 继续保留，但角色改为：

> **synthetic vulnerability task generator。**

每个 LOSO fold 的 synthetic generator 只能由该 fold 的 3 个 training source 拟合。

## 8.1 Synthetic task

从低秩 latent prior 中采样：

\[
z^{(j)}\sim N(0,I)
\]

并通过当前 basis GP 得到：

\[
f^{(j)}(x)=\mu(x)+b(x)^Tz^{(j)}+\epsilon.
\]

第一阶段每 fold 建议生成：

```text
10,000 synthetic vulnerability tasks
```

只用于训练：

- scenario/context representation；
- in-context vulnerability inference；
- predictive uncertainty。

**第一阶段不要求 synthetic task 直接生成可靠 formal collision label。**

Formal Event Head 主要使用真实 training-source observations 训练。

这样避免人为制造“伪碰撞”。

---

# 9. 训练流程

## Phase S1-P0 — Contract Freeze

在任何 AI 训练前：

1. 运行完整 `mvr/tests`；
2. 保存 Git commit、`diva_cutin.yaml` hash、`source_observations_v2.jsonl` hash、`source_loso_g1_v2.json` hash；
3. 建立 `diva_former_stage1.yaml` 与 Stage 1 manifest；
4. validation/test SUT 必须处于 sealed 状态。

若 contract hash 变化，Stage 1 所有结果必须作废重新执行。

## Phase S1-P1 — Teacher Viability

先不训练神经网络。

构建 counterfactual teacher dataset，并验证 belief-space downstream value 本身是否比当前 heuristic 有额外信息。

如果 teacher 自己不能带来 fixed-budget 改善：

**立即停止 DIVA-Former，不训练 Transformer。**

## Phase S1-P2 — Belief Model Pretraining

训练 Scenario Encoder、Outcome Encoder、Belief Encoder、Vulnerability Decoder。

损失：

\[
\mathcal L_{belief}=\mathcal L_{vuln-NLL}+\lambda_e\mathcal L_{event}.
\]

推荐初始：

```yaml
lambda_event: 0.5
```

synthetic prior tasks 主要优化 vulnerability NLL。

真实 training-source data 同时优化 vulnerability + event classification。

## Phase S1-P3 — Counterfactual Utility Distillation

加载 P2 checkpoint。

训练 Decision Value Head，并允许小学习率联合微调 belief encoder。

总损失：

\[
\mathcal L=\mathcal L_{belief}+\lambda_F\mathcal L_{Q_F}+\lambda_V\mathcal L_{Q_V}+\lambda_R\mathcal L_{rank}.
\]

建议第一版：

```yaml
lambda_formal_q: 1.0
lambda_vulnerability_q: 0.25
lambda_rank: 1.0
```

Q regression 使用 normalized Q 后的 Huber loss。

Ranking loss 对 teacher utility 差距足够大的 pair 使用 pairwise ranking loss。

建议：

```yaml
pair_margin_formal: 0.25
pair_margin_vulnerability: 0.10
```

## Phase S1-P4 — Source-Only LOSO Final Gate

每 fold 训练完全独立模型。

最终建议：

```text
4 LOSO folds × 3 network random seeds
```

共 12 个小模型。

held-out fold 只运行一次最终评估。

任何针对 held-out 结果的超参数修改，都必须产生新实验版本并重新执行四个 fold。

---

# 10. 推荐初始模型配置

```yaml
schema: diva_former_stage1_v1
base_config: mvr/metadrive/configs/diva_cutin.yaml

model:
  d_model: 128
  context_layers: 4
  cross_attention_layers: 2
  num_heads: 4
  ffn_dim: 256
  dropout: 0.10
  candidate_embedding_dim: 16
  max_context: 20
  max_parameters: 3000000

pretraining:
  synthetic_tasks_per_fold: 10000
  batch_size: 64
  max_steps: 30000
  learning_rate: 0.0003
  weight_decay: 0.0001

utility_distillation:
  states_per_training_sut: 2000
  batch_size: 32
  max_steps: 20000
  learning_rate: 0.0001
  lambda_formal_q: 1.0
  lambda_vulnerability_q: 0.25
  lambda_rank: 1.0

teacher:
  total_budget: 20
  continuation_policies:
    - mine_now
    - diagnose1_then_mine
    - diagnose2_then_mine

evaluation:
  network_seeds: [2601, 2602, 2603]
  total_budget: 20
  ndcg_k: 8
```

第一阶段原则：**不进行大范围超参数搜索。**

---

# 11. 验收 Gate 总览

第一阶段采用 **5 道 gate**。任一 gate 失败，不进入下一道。

---

# G1-0：数据、协议与无泄漏 Gate

### 必须全部满足

- [ ] `pytest mvr/tests -q` 全部通过；
- [ ] 新增 DIVA-AI tests 全部通过；
- [ ] 训练脚本不实例化 MetaDrive environment；
- [ ] Stage 1 新增 simulator calls = 0；
- [ ] 训练数据中不存在 `idm_fast_small_gap` 与 `idm_late_response`；
- [ ] 模型 input schema 中不存在 `sut_ref`、profile id 或 IDM controller 参数；
- [ ] held-out source 在当前 fold 不进入 prior fit、synthetic generator、teacher dataset、optimizer 或 early stopping；
- [ ] existing DIVA v2 artifacts 不被覆盖；
- [ ] config / input / checkpoint 均记录 SHA256；
- [ ] `B=20` budget ledger 不允许越界。

### 决策

任何一项失败：**Stage 1 = INVALID，而不是模型性能失败。**

---

# G1-1：Counterfactual Teacher Viability Gate

这是最关键的“先证明新 idea 值得训练 AI”的门。

## 单元正确性

- [ ] 当 `remaining_budget = 1`，`Q_T^F=r(x)`，数值误差 `< 1e-10`；
- [ ] 当 `remaining_budget = 1`，`Q_T^V=v(x)`，数值误差 `< 1e-10`；
- [ ] teacher portfolio return 不得低于其三个 continuation policy 中任意一个；
- [ ] candidate 已选过时，teacher 不得再次选择；
- [ ] rollout 每次 observation 后必须更新 analytic posterior。

## 信号非退化

在 `b >= 4` 的 sampled states 中：

- [ ] 至少 50% state 的 `Q_vulnerability` max-min ≥ `0.10 × b`；
- [ ] 对包含 formal positive event 的 training states，至少 30% state 的 `Q_formal` 存在 ≥ `0.5` 的 action gap。

若做不到，说明“选哪个测试对未来几乎没有差别”，则 Counterfactual Utility 不是当前数据下的有效创新点。

## Teacher 下游收益

在 source-only training-fold internal evaluation 中，Teacher policy 的 `FormalScore@20` 必须：

- [ ] 不低于 current DIVA fixed-K=4；
- [ ] 不低于 mine-now；
- [ ] 聚合绝对收益至少比 best single heuristic 高 `0.5` formal score，或相对提升 ≥ 5%。

若 best baseline 接近当前 panel 的理论上限，则允许使用 teacher regret 相对 oracle 上限显著下降作为替代证明，但必须在报告中解释 ceiling effect。

### 失败停止规则

**Teacher 本身没有 downstream gain → 不训练 DIVA-Former。**

---

# G1-2：Belief / In-Context Inference Gate

评估对象：held-out source SUT。

K 仍只用于分析：

```text
K = 0 / 1 / 2 / 4
```

不是最终 policy 的固定 support 数。

## 必须报告

- RMSE；
- vulnerability NLL；
- NDCG@8；
- Top-8 Recall；
- Spearman；
- event Brier；
- AUROC（仅 event-bearing fold）；
- calibration curve；
- predictive uncertainty。

## 验收要求

以当前 analytic DIVA source-LOSO 作为 reference。

### K=4

- [ ] DIVA-Former NDCG@8 ≥ `0.95 × analytic DIVA NDCG@8`；
- [ ] Top-8 Recall ≥ `0.75`；
- [ ] RMSE ≤ `1.10 × analytic DIVA RMSE`。

根据当前 v2 结果，参考量级约：

```text
analytic K4 NDCG@8 ≈ 0.968
analytic K4 RMSE   ≈ 0.125
analytic K4 Top8 Recall ≈ 0.844
```

第一版 student 并不要求立即超过解析模型，但必须基本恢复其 few-shot inference 能力。

### Few-shot gain

- [ ] `NDCG@8(K=4) - NDCG@8(K=0) >= 0.04`；
- [ ] `RMSE(K=4) < RMSE(K=0)`。

### Context permutation invariance

对同一 context 随机打乱 100 次：

- [ ] vulnerability mean 最大绝对变化 `< 1e-5`；
- [ ] event probability 最大绝对变化 `< 1e-5`；
- [ ] value head 最大绝对变化 `< 1e-5`。

### Context usage sanity test

将 `(scenario, outcome)` 随机错误配对后：

- [ ] held-out NDCG@8 至少下降 `0.03`，或 RMSE 至少恶化 5%。

否则说明模型可能根本没有学习 scenario-response-belief 关系。

---

# G1-3：Utility Distillation Gate

这里只检查 student 是否真的学会了 teacher 的测试价值，而不是仅学习 risk prediction。

对 held-out source 的 counterfactual states：

## Formal Q

- [ ] pairwise action-ranking accuracy ≥ `0.70`；
- [ ] teacher-action ranking NDCG@10 ≥ `0.85`；
- [ ] normalized top-1 regret ≤ `0.15`。

其中：

\[
regret=\frac{Q_T(x_T^*)-Q_T(x_S^*)}{\max Q_T-\min Q_T+\epsilon}.
\]

## Dense Q

- [ ] `Q_vulnerability` pairwise ranking accuracy ≥ `0.75`。

## Budget conditioning

### b = 1

- [ ] `Q_formal` MAE ≤ `0.05`；
- [ ] student top action 与 teacher immediate-reward top-3 的一致率 ≥ 90%。

### budget-aware behavior

只在 teacher 的最优 action 确实随预算改变的 states 中统计：

- [ ] student 对 `b=1` 与 `b>=8` 的 teacher action preference agreement ≥ 60%。

此测试用于防止模型完全忽略 `remaining_budget`。

---

# G1-4：Belief-Space Sequential Policy Gate

这是 Stage 1 的最终方法 Gate。

## 评估方式

每个 LOSO fold：

```text
held-out source SUT
candidate pool = 64 common source anchors
total budget B = 20
start context = empty
```

DIVA-Former 每一步：

```text
encode history
→ score all untested candidates
→ argmax Q_formal
→ lookup held-out real observation
→ append context
→ next step
```

**不得在线梯度更新。**

## 必须比较的 baseline

1. Random；
2. Frozen Source Mean；
3. Current DIVA mine-now / online_diva_k0；
4. Current DIVA fixed K=4 diagnostic → mining；
5. Highest Shared Risk；
6. Counterfactual Teacher；
7. DIVA-Former。

Target-only GP 不作为 Stage 1 必须 baseline，但可以保留为附加对照。

## 必须报告的 policy metrics

### Primary

`FormalScore@20`

### Secondary

- Failure Count@20；
- Collision Count@20；
- Near-Miss Count@20；
- `CumulativeVulnerability@20`；
- First-Failure Budget；
- cumulative formal-score curve `R@1...R@20`；
- unique scenario count；
- duplicate selection = 0；
- average decision latency。

## 特殊处理：cautious fold

当前 source bank 中 `idm_cautious` 没有 formal positive event。

因此 formal failure-discovery gain 在该 fold没有可辨识 headroom；仍必须报告 vulnerability cumulative utility、ranking、calibration 与 model stability。

不得为了让正式收益“看起来更好”而移除 cautious fold。

formal failure-yield 的主比较在 event-bearing held-out folds：

```text
idm_defensive
idm_normal
idm_assertive
```

## Final policy acceptance

设：

```text
R_B = best non-teacher baseline FormalScore@20
R_T = teacher FormalScore@20
R_S = student FormalScore@20
```

在 event-bearing folds 聚合。

### Teacher 必须先有 headroom

- [ ] `R_T > R_B`。

若 teacher 都无法比当前 DIVA 好，不允许以 Transformer 的偶然收益强行通过。

### Student 必须恢复 teacher gain

定义：

\[
Recovery=\frac{R_S-R_B}{R_T-R_B}.
\]

要求：

- [ ] `Recovery >= 0.70`；
- [ ] `R_S >= R_B`；
- [ ] 三个 event-bearing fold 中至少 2 个 fold：DIVA-Former ≥ best baseline；
- [ ] `First-Failure Budget` 中位数不得差于 best baseline。

### Dense mechanism metric

在全部 4 folds：

- [ ] DIVA-Former `CumulativeVulnerability@20` 至少比 best non-teacher baseline 聚合提升 3%；
- [ ] 若未提升，则必须有 formal-score 明确提升，否则 Stage 1 不通过。

### 模型随机种子稳定性

4 folds × 3 network seeds：

- [ ] 不允许只有 1 个 seed 获得主要收益；
- [ ] 至少 2/3 network seeds 满足 `R_S >= R_B`。

---

# G1-5：工程与算力 Gate

## 模型规模

- [ ] trainable parameters ≤ 3M；
- [ ] Stage 1 不允许因为性能差而突破 5M。

## 部署行为

- [ ] target adaptation 无 gradient update；
- [ ] 单次 context update 只需要 forward；
- [ ] 8192 candidates 可以 batch scoring；
- [ ] GPU 显存目标 < 4 GB；
- [ ] 当前项目 GPU 上，8192 candidates 单步决策目标 < 2 s；
- [ ] 若没有 GPU，可 CPU fallback，结果必须数值一致至合理浮点容差。

## 训练成本

第一阶段硬上限：

```text
new simulator calls: 0
total GPU training: <= 24 GPU-hours
recommended target: <= 12 GPU-hours
```

若 12 个 LOSO-seed 模型无法在该预算内训练，优先减少 synthetic task 数量 / training steps，而不是扩大算力。

---

# 12. 训练 checkpoint 选择规则

严禁使用 held-out source 的 final metrics 选择 checkpoint。

每个 fold：

```text
3 training SUT
    ↓
training-state split
    ↓
train / internal-dev
```

checkpoint priority：

1. dev vulnerability NLL；
2. dev teacher action NDCG；
3. dev Q ranking accuracy。

最终只加载一次 frozen best checkpoint 到 held-out source。

---

# 13. 建议新增的关键单元测试

## `test_diva_ai_no_sut_identity.py`

断言所有 model forward input 中不存在：

```text
sut_ref
profile_id
distance_wanted
time_headway
speed_ratio
acceleration_factor
deceleration_factor
```

## `test_diva_ai_context_invariance.py`

同一 context 不同排列：

```text
prediction A == prediction B
```

容差：`atol <= 1e-5`。

## `test_diva_counterfactual_teacher.py`

覆盖：

- B=1 identity；
- selected case 不可重复；
- posterior 必须随 counterfactual response 更新；
- teacher portfolio return ≥ each continuation；
- formal 和 vulnerability return 分开累计。

## `test_diva_former_budget_conditioning.py`

构造 teacher 明确表现出：

```text
b=1 → risk-first
b=10 → information-first
```

的 synthetic state。

DIVA-Former 必须能够使用 `remaining_budget` 字段，至少 forward 输出不同。

## `test_diva_former_policy_budget.py`

严格执行：

```text
B=20
exactly 20 distinct selections
no duplicate
no budget overrun
```

---

# 14. Stage 1 结果目录要求

所有新产物单独保存：

```text
results/metadrive/diva_former/cutin_g01/stage1/
```

建议：

```text
manifest.json

teacher/
    fold_0.jsonl
    fold_1.jsonl
    fold_2.jsonl
    fold_3.jsonl
    teacher_gate.json

datasets/
    counterfactual_fold_0.jsonl
    counterfactual_fold_1.jsonl
    counterfactual_fold_2.jsonl
    counterfactual_fold_3.jsonl

checkpoints/
    fold_0_seed_2601.pt
    ...
    fold_3_seed_2603.pt

evaluation/
    belief_metrics.json
    utility_metrics.json
    policy_metrics.json
    ablations.json

stage1_gate.json
```

大数据训练集若体积太大，可以不提交 Git，但 manifest 必须保存 generator config、source hashes、random seed、row count、SHA256 与 rebuild command。

---

# 15. `stage1_gate.json` 最终必须包含

```json
{
  "schema": "diva_former_stage1_gate_v1",
  "source_only": true,
  "new_simulator_calls": 0,
  "g1_0_contract": {"pass": true},
  "g1_1_teacher": {"pass": true},
  "g1_2_belief": {"pass": true},
  "g1_3_distillation": {"pass": true},
  "g1_4_policy": {"pass": true},
  "g1_5_engineering": {"pass": true},
  "decision": "continue_to_unseen_validation_sut"
}
```

如果任一 gate 失败：

```json
"decision": "stop_before_validation_sut"
```

---

# 16. 第一阶段最重要的消融实验

只做能够回答机制归因的问题。

## A1 — No Counterfactual Utility

DIVA-Former 不训练 Value Head，测试时只按预测 risk 最大化。

回答：AI 的收益是不是仅来自更强的 vulnerability predictor？

## A2 — Information Gain Heuristic

保留 Transformer belief model，但决策使用当前：

```text
IG × boundary × validity
```

回答：Counterfactual downstream utility 是否真的优于传统 information gain？

## A3 — No Budget Conditioning

Value Head 不输入 remaining budget。

回答：belief-space formulation 是否真的需要显式预算？

## A4 — No In-Context Outcome

只给 scenario，不给 target outcome。

回答：方法是否真正利用新 SUT 的少量反馈？

## A5 — Analytic DIVA

当前 low-rank + posterior + heuristic。

回答：AI 模型是否比现有解析方案带来实质新能力？

---

# 17. 第一阶段成功后才允许做什么

只有 `stage1_gate.json` 全通过，才允许进入：

## Stage 2 — Unseen Validation SUT

目标：

```text
idm_fast_small_gap
```

此时才进行真正 simulator B=20 fixed-budget evaluation。

Stage 2 重点回答：

> source-only counterfactual distillation 是否真正迁移到从未训练过的新 SUT。

Stage 2 前仍然禁止打开：

```text
idm_late_response
```

test SUT。

---

# 18. 第一阶段失败后的处理规则

## Teacher Gate 失败

说明 belief-space downstream decision 本身没有比现有 heuristic 提供足够 headroom。

行动：

- 不训练 Transformer；
- 检查 teacher horizon、source task diversity、formal reward sparsity；
- 不允许增加网络复杂度解决 teacher 问题。

## Belief Gate 失败

说明 AI in-context model 连当前解析 posterior 的基本 few-shot inference 都学不会。

行动优先检查：

- set invariance；
- normalization；
- synthetic prior mismatch；
- event imbalance；
- context masking。

不进入 Value Head。

## Utility Distillation Gate 失败

说明 teacher 有价值，但 student 没学会。

行动优先级：

1. action-ranking loss；
2. Q normalization；
3. teacher data coverage；
4. budget embedding；
5. candidate representation。

最后才考虑扩大模型。

## Policy Gate 失败

如果 belief prediction 好、Q imitation 好、真实 sequential policy 差，说明 one-step Q distillation 与闭环执行之间存在 distribution shift。

下一版优先：

- DAgger-style teacher relabeling；
- student-policy induced states；
- iterative counterfactual dataset augmentation。

**不要第一反应回到 RL。**

---

# 19. Definition of Done

只有以下全部满足，才能宣称“DIVA-Former 第一阶段训练完成”：

- [ ] 当前 DIVA v2 physics/failure contract 完整保留；
- [ ] zero new target/simulator calls；
- [ ] source LOSO 完整隔离；
- [ ] Counterfactual Teacher 本身产生可验证 downstream gain；
- [ ] DIVA-Former 能进行 permutation-invariant in-context vulnerability inference；
- [ ] few-shot context 明显改善 held-out SUT prediction；
- [ ] Value Head 能恢复 teacher action ranking；
- [ ] 模型对 remaining budget 有真实响应；
- [ ] sequential B=20 policy 恢复至少 70% teacher gain；
- [ ] 不依赖固定 K；
- [ ] 不使用 SUT identity；
- [ ] 不做 target-time gradient update；
- [ ] 训练成本在单卡可接受范围内；
- [ ] 所有结果、checkpoint、dataset manifest 可重建；
- [ ] `stage1_gate.json` 判定 `continue_to_unseen_validation_sut`。

---

# 20. 推荐执行顺序

严格按以下顺序：

```text
① 冻结当前 DIVA v2
        ↓
② 新建 diva_ai/，只实现 state + teacher
        ↓
③ 构造 counterfactual teacher，先过 G1-1
        ↓
④ 实现 Scenario / Outcome / Belief Encoder
        ↓
⑤ 只训练 vulnerability inference，先过 G1-2
        ↓
⑥ 加入 Value Head
        ↓
⑦ 反事实 Utility Distillation，过 G1-3
        ↓
⑧ 实现统一 belief-space policy
        ↓
⑨ source-only LOSO B=20，过 G1-4
        ↓
⑩ 工程验收，生成 stage1_gate.json
        ↓
⑪ 才允许访问 validation SUT
```

---

# 21. 最终研究判断标准

第一阶段真正要证明的不是：

> “Transformer 比 GP 的 RMSE 更小。”

而是下面这条完整证据链：

\[
\boxed{
\text{少量 target observations}
\rightarrow
\text{形成可迁移的 safety belief}
\rightarrow
\text{反事实估计每个测试的 downstream value}
\rightarrow
\text{AI 学会 budget-aware test decision}
\rightarrow
\text{在相同 B=20 下获得更高 failure-discovery utility}
}
\]

如果这条链条不能在 source-only LOSO 中成立：

> **停止扩大 DIVA-Former。**

如果成立：

> 才进入 unseen validation SUT，并开始检验真正的跨 SUT regression-testing claim。

---

## 附录 A：当前 DIVA 与 DIVA-Former 的角色变化

| 当前组件 | 当前角色 | DIVA-Former Stage 1 角色 |
|---|---|---|
| Low-rank factorization | 最终 prior | teacher prior + synthetic task generator + baseline |
| Gaussian posterior | target adaptation | analytic teacher belief + baseline |
| diagnostic_scores | support acquisition | IG baseline + teacher portfolio component |
| mining_scores | failure mining | greedy baseline + teacher portfolio component |
| fixed K=1/2/4 | 正式 protocol | 机制分析条件，不再是最终 policy |
| SourceBank | prior data | population data + counterfactual world model |
| DIVA Miner | 最终算法 | teacher / baseline |
| DIVA-Former | 无 | 最终 learned belief-space policy |

---

## 附录 B：最低实现优先级

如果时间有限，只实现以下 MVP：

```text
MVP-1
Counterfactual teacher on 64 real anchors
→ 必须证明 teacher > current heuristic

MVP-2
Small set-transformer belief model
→ 必须近似 current analytic posterior 的 held-out ranking

MVP-3
Value head + pairwise utility distillation
→ 必须学习 budget-aware action ranking

MVP-4
Unified B=20 student policy
→ 必须不使用固定 K
```

PFN-style 更大规模 synthetic pretraining、prior trust gate、controller-family OOD 均放到后续阶段，不得阻塞本阶段核心验证。

---

## 最重要的停止规则

> 如果 Counterfactual Teacher 本身不能在 source-only fixed-budget replay 中优于当前 DIVA heuristic，或者 DIVA-Former 无法把 teacher 的 downstream utility 转化为闭环 B=20 收益，则停止增加模型复杂度；优先否定或修正 belief-space / counterfactual-utility 假设，而不是增加训练规模。
