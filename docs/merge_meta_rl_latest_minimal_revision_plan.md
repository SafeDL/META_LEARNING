# Merge Meta-RL 当前代码与结果：最小必要修改方案

> 适用仓库：`SafeDL/META_LEARNING`  
> 目标：基于当前最新代码与已有 Gate 结果，只规划**必要修改**，不扩大实验规模，不提前引入 Structure-Aware Prior、MoE、GNN/Transformer 或大规模正式训练。  
> 当前阶段原则：**先打通 `support → posterior → policy → task-specific outcome` 的因果链，再继续方法扩展。**

---

## 1. 当前状态判断

当前代码已经不再停留在“Task 不可辨识”的阶段。根据最新机制实验：

```text
Gate 0  配置与工程链路                  pass
Gate 1  Task-Policy Conflict          pass
Gate 1B Quick Single-Task SAC         pass
Gate 2  Context Identifiability       pass
Gate 3  Vanilla PEARL causal chain    pending
```

最新 screened arrival-controller 机制实验已经表明：

- 两个任务存在稳定的策略冲突；
- 单任务 SAC 能够学习 task-specific 策略；
- cross-task transfer matrix 存在明显对角优势；
- 使用完全相同 probing policy 时，仅根据 transition 数据即可区分两个任务；
- 因此，当前没有必要继续修改 Task 定义、IDM、reward、arrival controller 或 PEARL 网络容量。

当前真正缺失的是：

\[
C_{\mathrm{correct}} \neq C_{\mathrm{wrong}}
\]

是否能够进一步导致：

\[
q(z|C_{\mathrm{correct}})
\neq q(z|C_{\mathrm{wrong}})
\]

再进一步导致：

\[
\pi(a|s,z_{\mathrm{correct}})
\neq
\pi(a|s,z_{\mathrm{wrong}})
\]

以及最终：

\[
J(T,C_{\mathrm{correct}})
>
J(T,C_{\mathrm{wrong}})
\]

因此下一步只需完成 **Gate 3：Vanilla PEARL causal mechanism**。

---

## 2. 当前不要修改的部分

### 2.1 Actor / Critic latent 接口

当前 Dense Actor 已明确使用：

```text
actor input = observation + latent z
```

Critic 使用：

```text
critic input = observation + action + latent z
```

因此不存在“latent 没有接入 policy”的明显实现错误。

### 2.2 `actor_z.detach()`

当前 Actor 更新阶段：

```python
actor_z = expanded_z.detach()
```

应当保留。

当前设计目标是：

```text
critic / Bellman objective
    ↓
Context Encoder

Actor
    ↓
利用 z
但不通过 actor loss 更新 Context Encoder
```

现阶段不要为了“增强 latent 使用”而删除 `detach()`。

### 2.3 Product-of-Gaussians

当前 `ContextEncoder` 已正确实现：

```text
transition evidence
    ↓
Gaussian factors
    ↓
Product-of-Gaussians
```

Structure-Aware 版本中，静态 prior 也是作为**一个单独的 Gaussian prior factor**加入，而不是重复拼接到每个 transition。

该部分当前不需要修改。

### 2.4 暂时禁止的扩展

Gate 3 通过之前，不新增：

```text
GNN
Transformer
MoE
disentangled auxiliary task supervision
larger latent dimension
larger Context Encoder
KL tuning
learning-rate sweep
200k+ PEARL training
multi-seed formal evaluation
```

这些修改会破坏当前已经建立起来的机制定位链。

---

## 3. P0：修复 strict-v3 calibration 入口

这是当前最重要的代码正确性修改。

### 3.1 当前问题

当前 mechanism task 使用：

```text
logical_order_spatiotemporal_near_miss_v3
```

底层 `apply_calibration_manifest()` 已经支持：

```text
spatiotemporal_near_miss_v2
logical_order_spatiotemporal_near_miss_v3
```

但：

```text
pearl_learning/scripts/train_pearl.py
pearl_learning/scripts/evaluate_fewshot.py
```

当前仅在：

```python
schema == "spatiotemporal_near_miss_v2"
```

时才加载 calibration manifest。

这意味着：

```text
Gate 1B / Gate 2
```

与：

```text
Gate 3 PEARL
```

可能使用不同 threshold contract。

这是必须修复的问题。

### 3.2 建议修改

不要再写：

```python
if schema == "spatiotemporal_near_miss_v2":
```

建议增加统一函数：

```python
requires_calibration(schema)
```

或直接：

```python
if is_strict_near_miss_schema(schema):
```

使下面两类指标都必须：

```text
--critical-thresholds
```

并执行：

```python
apply_calibration_manifest(...)
```

#### 必改文件

```text
pearl_learning/scripts/train_pearl.py
pearl_learning/scripts/evaluate_fewshot.py
```

#### 建议增加测试

```text
v2 + no manifest       -> fail
v3 + no manifest       -> fail
v2 + valid manifest    -> pass
v3 + valid manifest    -> pass
wrong manifest hash    -> fail
```

---

## 4. P0：扩展 Mechanism Casebook，形成真正的 PEARL train/support/query

### 4.1 当前问题

当前：

```text
build_mechanism_casebook.py
```

一次只生成一个 split。

例如：

```text
train_pool          有数据
validation_support  空
validation_query    空
test_support        空
test_query          空
```

这对于 Gate 1 / Gate 1B / Gate 2 是足够的，但无法用于真正的 few-shot Gate 3。

### 4.2 最小修改目标

当前仅需要很小规模的数据：

```text
train_pool : 6 matched conditions
support    : 4 matched conditions
query      : 4 matched conditions
```

每个 Task 总计：

```text
14 cases
```

即可，不需要恢复完整 benchmark casebook。

### 4.3 两个 Task 之间仍然必须 matched

对同一个 condition \(x_j\)，两个 task 必须保持：

```text
same case_seed
same SUT initial speed
same adversary initial speed
same spawn
same arrival gap
same distance-to-conflict
same relative speed
```

即：

\[
x_j^{A}=x_j^{B}
\]

这样 task difference 才来自隐藏逻辑目标，而不是物理初始化差异。

### 4.4 split 之间必须严格不重叠

要求：

\[
\mathcal X_{\mathrm{train}}
\cap
\mathcal X_{\mathrm{support}}
=
\varnothing
\]

\[
\mathcal X_{\mathrm{train}}
\cap
\mathcal X_{\mathrm{query}}
=
\varnothing
\]

\[
\mathcal X_{\mathrm{support}}
\cap
\mathcal X_{\mathrm{query}}
=
\varnothing
\]

不能直接使用普通 `validate_casebook_disjoint()`，因为普通 validator 禁止不同 Task 复用相同 seed；但 mechanism experiment 恰恰要求相同 matched condition 跨 Task 共享物理 seed。

因此建议新增：

```python
validate_mechanism_split_disjointness()
```

规则为：

```text
Across tasks:
    matched conditions CAN share physical seed

Across splits:
    condition IDs / seeds MUST be disjoint
```

---

## 5. P0：增加 Gate 3 专用 Vanilla PEARL 配置

当前不要直接复用：

```text
merge_method_flow_pilot.yaml
```

建议新增：

```text
pearl_learning/configs/merge_method_flow_gate3_vanilla_pearl.yaml
```

### 5.1 建议配置核心

```yaml
extends: merge_method_flow_logical_order_screened_arrival_controller.yaml

experiment:
  method_variant: vanilla_gate3

scenario_representation:
  enabled: false

scenario_prior:
  mode: unit_normal

networks:
  actor_architecture: dense

posterior_routed_moe:
  enabled: false

method_flow_pilot:
  task_ids:
    meta_train:
      - lane_drop_24__logical_order_adversary_first
      - lane_drop_24__logical_order_sut_first
    meta_validation: []

meta_training:
  meta_batch_size: 2
  total_environment_steps: 20000

evaluation:
  shots: [0, 1, 2, 4]
```

同时继续继承已经通过 Gate 1B 的：

```text
fixed IDM
1-D mechanism action
target_arrival_gap controller
logical_order_spatiotemporal_near_miss_v3
collision barrier
current frozen reward
```

### 5.2 不要重新修改

下面参数全部冻结：

```text
arrival target scale
collision penalty
collision-risk barrier
IDM target speed
screened case definition
mechanism action interface
strict VCSR definition
```

Gate 3 应只验证 PEARL。

---

## 6. P1：增加 mechanism-specific latent causal audit

当前 `audit_latent_context_interventions.py` 主要面向普通 logical-scenario pair，且使用普通 Casebook schema。

不建议大改原脚本。

新增：

```text
pearl_learning/scripts/audit_gate3_vanilla_pearl_mechanism.py
```

即可。

### 6.1 必测四种 latent

对于相同 query state / query case：

```text
z_prior
z_correct
z_wrong
z_zero
```

其中最重要的是：

```text
z_correct vs z_wrong
```

### 6.2 Latent-level diagnostics

输出：

\[
D_{\mathrm{cw}}
=
\|\mu_{\mathrm{correct}}
-
\mu_{\mathrm{wrong}}\|_2
\]

\[
D_{\mathrm{cp}}
=
\|\mu_{\mathrm{correct}}
-
\mu_{\mathrm{prior}}\|_2
\]

\[
D_{\mathrm{wp}}
=
\|\mu_{\mathrm{wrong}}
-
\mu_{\mathrm{prior}}\|_2
\]

同时输出：

```text
posterior variance
posterior log variance
```

### 6.3 Fixed-state action audit

建立相同 state bank \(S\)：

\[
S=\{s_1,\ldots,s_N\}
\]

计算：

\[
D_a^{cw}
=
\frac{1}{N}
\sum_s
\|
\pi(s,z_{\mathrm{correct}})
-
\pi(s,z_{\mathrm{wrong}})
\|_2
\]

以及：

```text
D_a(correct, prior)
D_a(zero, prior)
```

### 6.4 Query-level causal outcome

在完全相同 query cases 上比较：

```text
correct context
wrong context
prior
zero latent
```

输出：

```text
VCSR
episode return
target collision rate
invalid rate
episodes/environment-steps to first valid critical
action trajectory L2
arrival-gap trajectory
```

这里最关键的不是单纯：

```text
K=4 > K=0
```

而是：

```text
correct context > wrong context
```

---

## 7. P1：增加三个训练诊断日志

当前 Gate 3 如果失败，需要能够定位失败发生在哪一段：

```text
support -> encoder
encoder -> posterior
posterior -> actor
actor -> environment outcome
```

因此建议只增加日志，不修改 loss。

### 7.1 Context Encoder 的 Critic 梯度

新增：

```text
context_encoder_critic_gradient_norm
```

定义：

\[
G_C
=
\|
\nabla_{\phi}L_Q
\|
\]

注意：`context_encoder_actor_gradient_norm` 由于 `actor_z.detach()` 的设计，本来就应该接近 0，因此不能判断 encoder 有没有真正被训练。

### 7.2 Posterior-prior displacement

新增：

```text
posterior_prior_mean_l2
```

即：

\[
D_{\mathrm{post}}
=
\|\mu_q-\mu_p\|_2
\]

对于 Vanilla PEARL：

\[
\mu_p=0
\]

### 7.3 Evidence-to-prior precision ratio

新增：

```text
evidence_to_prior_precision_ratio
```

例如：

\[
R_{\Lambda}
=
\frac{
\operatorname{mean}(\Lambda_{\mathrm{evidence}})
}{
\operatorname{mean}(\Lambda_{\mathrm{prior}})+\epsilon
}
\]

用于观察 context evidence 是否真正改变 posterior。

---

## 8. Gate 3 的唯一训练预算

第一轮：

```text
Tasks: 2
Seed: 1
Environment steps: 20k
Actor: Dense
Prior: Unit Normal
Scenario Encoder: OFF
MoE: OFF
```

禁止直接增加到：

```text
100k
200k
500k
multi-seed
```

20k 的目的不是获得最终性能，而是验证机制。

---

## 9. Gate 3 应如何判定

### Stage A：Context 是否改变 posterior？

要求至少看到：

\[
\mu_{\mathrm{correct}}
\neq
\mu_{\mathrm{wrong}}
\]

如果：

```text
correct posterior ≈ wrong posterior
```

则问题位于：

```text
support evidence
Context Encoder optimization
posterior collapse
```

此时不要修改 Actor。

### Stage B：Posterior 是否改变 Actor？

如果 posterior 已分离，但：

\[
a_{\mathrm{correct}}
\approx
a_{\mathrm{wrong}}
\]

说明：

```text
Actor is latent-insensitive
```

这时才有资格研究 latent conditioning strength / Actor architecture / Critic latent dependence。

### Stage C：动作变化是否改变 outcome？

如果：

\[
a_{\mathrm{correct}}
\neq
a_{\mathrm{wrong}}
\]

但 VCSR / return / collision 不变，则问题位于：

```text
control interface
reward-objective alignment
query case sensitivity
```

而不是 Context Encoder。

### Stage D：Gate 3 成功

希望最终得到：

\[
C_{\mathrm{correct}}
\neq
C_{\mathrm{wrong}}
\]

\[
\Downarrow
\]

\[
q(z|C_{\mathrm{correct}})
\neq
q(z|C_{\mathrm{wrong}})
\]

\[
\Downarrow
\]

\[
\pi(a|s,z_{\mathrm{correct}})
\neq
\pi(a|s,z_{\mathrm{wrong}})
\]

\[
\Downarrow
\]

\[
J(T,C_{\mathrm{correct}})
>
J(T,C_{\mathrm{wrong}})
\]

只要在部分 matched query cases 上稳定出现这一链条，当前 mechanism gate 即可认为通过。

---

## 10. 当前不要运行 Structure-Aware PEARL

当前两个 mechanism Task：

```text
same map
same route
same geometry
same IDM
same initial matched case
```

只改变隐藏 logical-order objective。

但当前 `ScenarioEncoder` 只读取静态场景结构，因此对于这两个任务：

\[
e_A=e_B
\]

是合理结果。

所以当前比较：

```text
Vanilla PEARL
vs
Structure-Aware PEARL
```

没有方法学意义。

Structure-Aware Prior 应等待重新回到：

```text
lane-drop
bottleneck
Y-merge
geometry variation
```

等真正存在静态结构差异的 Task 后再测试。

---

## 11. 当前不要继续扩大 Context Identifiability 数据

最新 Gate 2 已经表明：

```text
fixed probing policy
transition-only feature
excluded task_id / geometry_id / descriptor / case_id
held-out accuracy = 1.0
```

当前样本量显然不足以作为论文统计结果，但作为 mechanism engineering gate 已经足够。

因此现在不需要：

```text
更多 trajectory
更复杂 classifier
DeepSets
Transformer classifier
```

---

## 12. 当前 mechanism pair 的学术定位

当前两个 Task 的区别主要是：

```text
hidden desired conflict-entry order
```

即 reward / success semantics 不同。

PEARL context 包含：

```text
(s, a, r, s')
```

因此 task 可以通过 reward response 被识别。这不是 leakage。

但它只证明：

\[
\boxed{
\text{PEARL-style reward-varying task can be identified from few-shot interaction}
}
\]

它还没有证明：

\[
\boxed{
\text{different physical Merge logical scenarios can be few-shot adapted}
}
\]

当前 Gate 3 的目标只是证明：

```text
support -> posterior -> policy -> outcome
```

这条 PEARL 因果链能够在当前自动驾驶场景挖掘实现中成立。

---

## 13. 必要的实验状态文档清理

当前仓库旧机制报告仍记录：

```text
Gate 1B fail
Gate 2 forbidden
PEARL forbidden
```

但最新 screened experiment 已经：

```text
Gate 1B pass
Gate 2 pass
```

建议：

### 保留旧报告

不要删除历史结果，只增加头部：

```text
STATUS: SUPERSEDED / HISTORICAL
```

### 新增当前状态文件

```text
docs/merge_meta_rl_method_flow_current_status.md
```

明确记录：

```text
Gate 0   pass
Gate 1   pass
Gate 1B  pass
Gate 2   pass
Gate 3   pending
Gate 4   blocked by Gate 3
```

并列出每个 Gate 对应的：

```text
config hash
taskbook hash
casebook hash
result path
commit SHA
```

---

## 14. 最小修改清单

### 必须修改

```text
[1] train_pearl.py
    strict v2/v3 统一 calibration 入口

[2] evaluate_fewshot.py
    strict v2/v3 统一 calibration 入口

[3] mechanism_casebook.py / build_mechanism_casebook.py
    支持 train/support/query 三个 disjoint mechanism splits

[4] 新增 validate_mechanism_split_disjointness()
    保证跨 Task matched，跨 split disjoint

[5] 新增 merge_method_flow_gate3_vanilla_pearl.yaml
    只运行两个 mechanism Task 的 Vanilla PEARL

[6] 新增 audit_gate3_vanilla_pearl_mechanism.py
    correct / wrong / prior / zero latent causal intervention
```

### 建议修改

```text
[7] pearl_agent.py
    只增加：
    - context_encoder_critic_gradient_norm
    - posterior_prior_mean_l2
    - evidence_to_prior_precision_ratio

[8] docs
    标记旧 Gate 报告 superseded
    新增 current_status.md
```

### 当前不要修改

```text
Actor architecture
Context Encoder architecture
latent dimension
KL beta
SAC learning rate
reward
IDM
arrival controller
screened mechanism case definition
Scenario Encoder
Structure-Aware Prior
MoE
GNN
Transformer
Relatedness
```

---

## 15. 最小执行顺序

```text
Step 1
修复 v3 calibration CLI

Step 2
生成 6 train + 4 support + 4 query
matched mechanism cases

Step 3
验证：
cross-task matched
cross-split disjoint

Step 4
运行 Vanilla PEARL
2 tasks × 20k × 1 seed

Step 5
查看训练诊断：
gradient
posterior-prior displacement
evidence precision

Step 6
运行 causal intervention：
prior
correct
wrong
zero

Step 7
判断失败位置：
context -> posterior
posterior -> action
action -> outcome

Step 8
只有 Gate 3 成功后，
再恢复真实物理 Merge Task
并重新测试 Structure-Aware Prior
```

---

## 16. 当前阶段最终目标

这一轮不要求证明最终论文的：

```text
cross-topology generalization
OOD logical scenario adaptation
Structure-Aware superiority
MoE expert specialization
```

只要求获得一个干净的因果证据：

\[
\boxed{
\text{少量 support interaction}
\rightarrow
\text{不同 task posterior}
\rightarrow
\text{不同 adversarial policy}
\rightarrow
\text{更适合当前 task 的失败场景搜索结果}
}
\]

只要这一链条在当前极简 mechanism pair 中成立，方法流就可以认为真正跑通。

随后才进入下一阶段：

\[
\boxed{
\text{physical Merge task variation}
\rightarrow
\text{Scenario Encoder}
\rightarrow
\text{Structure-Aware Prior}
\rightarrow
\text{Transferability}
\rightarrow
\text{MoE}
}
\]

---

## 17. 一句话结论

当前代码不需要继续“增强 PEARL”。

真正必要的工作只有三类：

\[
\boxed{
\text{v3 calibration 对齐}
+
\text{few-shot mechanism Casebook split}
+
\text{Gate 3 causal audit}
}
\]

再补少量诊断日志即可。

在这三项完成之前，不建议增加训练预算、模型容量或新的方法模块。
