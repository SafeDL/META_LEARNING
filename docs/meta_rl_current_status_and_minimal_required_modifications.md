# 当前 Meta-RL 结果判断与最小必要修改目标

gpu在系统以下的环境中：
conda activate metadrive

## 1. 结论

**当前还不能认为“元强化学习方法已经初步完整跑通”。**

更准确的表述是：

> **PEARL 的元任务推断（meta-inference）子链路已经初步跑通，但“根据少量 support 交互识别任务 → 改变策略 → 在 query 上获得 task-specific 收益”的完整元强化学习适应链路尚未闭环。**

当前已经证明：

\[
C_{\mathrm{correct}} \neq C_{\mathrm{wrong}}
\Longrightarrow
q(z\mid C_{\mathrm{correct}})
\neq
q(z\mid C_{\mathrm{wrong}})
\]

但尚未证明：

\[
q(z\mid C_{\mathrm{correct}})
\neq
q(z\mid C_{\mathrm{wrong}})
\Longrightarrow
\pi(a\mid s,z_{\mathrm{correct}})
\neq
\pi(a\mid s,z_{\mathrm{wrong}})
\]

更没有完成：

\[
J(T,C_{\mathrm{correct}})
>
J(T,C_{\mathrm{wrong}})
\]

因此，从论文方法论角度，当前状态应定义为：

**“Meta-task inference 已成立，meta-policy adaptation 尚未成立。”**

---

## 2. 当前已经跑通的部分

当前 Gate 状态为：

| Gate | 目标 | 当前结果 |
|---|---|---|
| Gate 0 | 配置、训练、checkpoint、评估工程链路 | PASS |
| Gate 1 | 两个 Task 是否存在策略冲突 | PASS |
| Gate 1B | 单任务 SAC 是否能学出 task-specific 策略 | PASS |
| Gate 2 | 仅凭 transition 是否可识别 Task | PASS |
| Gate 3 Stage A | support context 是否形成 task-discriminative posterior | **PASS** |
| Gate 3 Stage B | posterior 是否导致不同动作策略 | **FAIL** |
| Gate 3 Stage C | 不同策略是否导致不同 query outcome | BLOCKED |
| Gate 4 | Structure-Aware / Transfer / MoE | BLOCKED |

Round 3 最关键的正面结果是：

```text
R_sep: 0.349 / 0.346
D_cw : 17.43 / 17.23
cos(z_correct, z_wrong): 0.98492 / 0.98532
```

这说明 `terminal_stratified_v1` 修复后，Context Encoder 已经不再只学习“有没有 context”的公共方向，而是开始形成两个 Task 的可区分 posterior。

因此以下链路已经成立：

```text
support trajectory
        ↓
context sampling
        ↓
Context Encoder
        ↓
task-discriminative posterior z
```

这是一个重要进展，说明 PEARL 的核心“从少量交互推断隐藏 Task”的机制已经开始工作。

---

## 3. 为什么还不能称为完整跑通 Meta-RL

Round 3 在 Stage B 仍然失败：

```text
correct/wrong action L2:
0.0258 / 0.0255

当前门槛:
0.1
```

更关键的是 Critic 诊断：

```text
Q-grid D_Q = 0.0
correct latent argmax action = +1.0
wrong latent argmax action   = +1.0
```

即：

\[
z_{\mathrm{correct}}
\neq
z_{\mathrm{wrong}}
\]

虽然已经成立，但 Critic 并没有因此形成：

\[
\arg\max_a Q(s,a,z_{\mathrm{correct}})
\neq
\arg\max_a Q(s,a,z_{\mathrm{wrong}})
\]

于是 Actor 也几乎不发生变化。

当前真正断裂的位置应理解为：

```text
support
   ↓
task-discriminative posterior       PASS
   ↓
task-dependent Q action preference  NOT ESTABLISHED
   ↓
task-dependent Actor                FAIL
   ↓
task-dependent query outcome        NOT TESTABLE YET
```

因此目前不能用“few-shot adaptation 已经成功”描述结果。

---

# 4. 最小必要修改目标

以下只列当前必要或条件性必要的修改，不扩展方法复杂度。

## P0：把 Gate 3 审计链拆成 Critic 与 Actor 两个独立环节

### 当前问题

当前 Gate 3 的逻辑是：

```text
Stage A: Context → Posterior
Stage B: Posterior → Actor Action
Stage C: Action → Outcome
```

但是 PEARL-SAC 实际的策略学习机制更接近：

```text
Posterior z
    ↓
Critic Q(s,a,z)
    ↓
Actor π(a|s,z)
```

当前审计实际上已经计算：

```text
critic_argmax_action_distance
actor_regret
Q-grid
```

但这些只作为 diagnostic，没有成为独立 Gate。

### 必要修改

将 Gate 3 升级为：

```text
Stage A
Context → task-discriminative posterior

Stage B_Q
Posterior → task-dependent Critic action preference

Stage B_π
Posterior/Critic → task-dependent Actor action

Stage C
Actor → task-dependent query outcome
```

建议保留历史 v3 结果，新建：

```text
gate3_vanilla_pearl_causal_chain_gate_v4
```

不要覆盖已有 v3 文件。

### 建议判据

Stage A 保持现有标准不变：

\[
D_{cw}\ge 0.5
\]

且：

\[
R_{\mathrm{sep}}\ge 0.25
\]

Stage \(B_Q\) 使用：

\[
a_c^*(s)=\arg\max_aQ(s,a,z_c)
\]

\[
a_w^*(s)=\arg\max_aQ(s,a,z_w)
\]

然后计算：

\[
D_Q^{action}
=
\mathbb{E}_s
[
\|a_c^*(s)-a_w^*(s)\|
]
\]

重点检查 **latent 是否改变动作价值排序**，而不是只看 Q 数值是否整体平移。

Stage \(B_\pi\) 继续使用当前 deterministic action L2 判据。

Stage C 保持当前 strict VCSR / paired query outcome 判据。

### 需要修改的文件

```text
pearl_learning/scripts/audit_gate3_vanilla_pearl_mechanism.py
```

这一修改不需要重新训练，可先用现有 Round 1–3 结果离线重判。

---

## P1：只跑一次“Round 3 sampler + Latent-FiLM Critic”

这是当前最必要的新训练实验。

### 原因

Round 2 已经使用过 Latent-FiLM Critic，并且证明：

```text
critic_latent_gradient_norm:
2.6 → 285
```

说明 FiLM Critic 确实建立了更强的 latent-conditioning 路径。

但 Round 2 当时的 posterior 并不满足当前 Stage A 标准：

```text
R_sep = 0.141 / 0.068
```

因此 Round 2 不能用于回答：

> “当真正 task-discriminative 的 z 已经形成后，FiLM Critic 能否利用它？”

Round 3 已经解决了这个前提：

```text
R_sep = 0.349 / 0.346
```

因此下一轮应严格只改变 Critic：

```text
Round 3:
terminal_stratified_v1
+
Dense Actor
+
Dense Critic

          ↓ only one variable

terminal_stratified_v1
+
Dense Actor
+
Latent-FiLM Critic
```

### 最小实现

新增一个配置即可，例如：

```text
pearl_learning/configs/
merge_method_flow_gate3_context_sampling_film_critic.yaml
```

继承：

```text
merge_method_flow_gate3_context_sampling.yaml
```

仅增加：

```yaml
networks:
  critic_architecture: latent_film_dense
```

### 保持全部冻结

```text
Tasks               unchanged
Casebook             unchanged
Query                unchanged
Reward               unchanged
Context sampler      terminal_stratified_v1
Context Encoder      unchanged
Actor                dense
Latent dimension     unchanged
KL beta              unchanged
Learning rate        unchanged
Prior                Unit Normal
Scenario Encoder     OFF
MoE                  OFF
Seed                 1
Budget               20k environment steps
```

---

# 5. 新一轮结果的决策规则

## 情况 A：Stage A 再次失败

如果 FiLM Critic 训练后：

\[
R_{\mathrm{sep}}<0.25
\]

则停止后续判断。

这说明新的 Critic 梯度改变了 Encoder 学到的 posterior，导致 task-discriminative representation 再次退化。

此时不能判断 Actor。

---

## 情况 B：Stage A PASS，但 Stage \(B_Q\) FAIL

即：

```text
posterior 已经区分 Task
但 Q-grid 最优动作仍完全一致
```

则说明当前主要问题仍在 Critic 学习信号，而不是 Actor。

### 下一步只做 0 环境步诊断

检查 Critic 的 RL replay batch 中：

```text
terminal transition 占比
conflict-near transition 占比
task-sensitive transition 占比
```

当前 Context sampler 已保证每个 support episode 包含 terminal transition，但 Critic RL batch 仍然主要从 replay 中均匀抽取非-context episode transitions。

如果 task-specific TD / reward / dynamics 信号主要集中在冲突末段，那么 Critic 可能仍被大量 common transitions 稀释。

**此诊断必须先做，不应直接修改 replay sampler。**

---

## 情况 C：Stage A PASS，Stage \(B_Q\) PASS，但 Stage \(B_\pi\) FAIL

这时才可以确认：

> Critic 已经学到 task-dependent action preference，但 Actor 没有利用这种差异。

只有在这种情况下，才授权修改 Actor。

最低优先级方案应是：

```text
Dense Actor
    ↓
Latent-FiLM Actor
```

或者增加明确的 latent skip-conditioning。

当前阶段不要直接引入 MoE Actor。

---

## 情况 D：Stage A、\(B_Q\)、\(B_\pi\) 全部 PASS，但 Stage C FAIL

这时才说明：

```text
support → posterior → Q → policy
```

已经跑通，但策略差异还没有转化为 query-level critical scenario mining 收益。

此时再检查：

```text
Actor adaptation magnitude
query case difficulty
20k budget是否只限制策略质量
```

在此之前不扩大训练预算。

---

# 6. 什么时候才能称为“初步跑通 Meta-RL”

建议使用一个明确的最低标准。

至少需要同时满足：

### 条件 1：Meta-inference 成立

\[
q(z|C_{\mathrm{correct}})
\neq
q(z|C_{\mathrm{wrong}})
\]

并满足 task-discriminative Stage A。

**当前已满足。**

### 条件 2：Meta-policy adaptation 成立

\[
\pi(a|s,z_{\mathrm{correct}})
\neq
\pi(a|s,z_{\mathrm{wrong}})
\]

而且差异应达到预注册的最低作用量，不只是数值噪声。

**当前未满足。**

### 条件 3：Few-shot query benefit 成立

正确 support context 应在 held-out query cases 上优于错误 context，例如：

\[
VCSR_{\mathrm{correct}}
>
VCSR_{\mathrm{wrong}}
\]

或获得明确的 paired return / criticality advantage。

**当前因为 Stage B 失败而尚未成立。**

只有完成这三步，才建议正式表述：

> **“Vanilla PEARL has been preliminarily validated for few-shot adaptation in the safety-critical scenario mining task.”**

当前更合适的表述是：

> **“The task-inference component of PEARL has been validated, while the policy-adaptation pathway remains incomplete.”**

---

# 7. 当前明确不要修改的部分

在上面的最小机制链跑通前，不建议修改：

```text
Task 定义
IDM 参数体系
arrival controller
reward
query casebook
Context Encoder 容量
Product-of-Gaussians
latent dimension
KL beta
SAC learning rate
Scenario Encoder
Structure-Aware Prior
Transferability module
MoE
GNN
Transformer
大规模多 seed 实验
20k 以上预算
```

尤其不要同时改变：

```text
Critic architecture
+
Critic replay distribution
+
Actor architecture
```

否则无法判断究竟是哪一个环节修复了机制。

---

# 8. 推荐的最短执行顺序

```text
Step 1
0-step:
升级 Gate 3 audit 为 A → B_Q → B_π → C
并离线重判现有 Round 1–3

        ↓

Step 2
20k / seed 1:
terminal_stratified_v1
+
Latent-FiLM Critic

        ↓

Step 3
看第一个失败 Gate

A fail
→ representation/gradient regression 定位

B_Q fail
→ 0-step Critic replay signal audit

B_Q pass, B_π fail
→ 允许最小 Latent-FiLM Actor

B_π pass, C fail
→ 再检查训练预算/策略质量/query-level效果

A+B_Q+B_π+C pass
→ Vanilla PEARL 初步跑通
→ 才进入 Gate 4
```

---

# 9. 最终判断

当前结果并不是失败的 Meta-RL 实验。

相反，它已经证明了一个非常关键的中间机制：

\[
\boxed{
\text{few-shot support evidence}
\rightarrow
\text{task-discriminative latent posterior}
}
\]

但是它还没有证明元强化学习最核心的最终作用：

\[
\boxed{
\text{task inference}
\rightarrow
\text{task-specific policy adaptation}
\rightarrow
\text{better query outcome}
}
\]

因此当前最合理的状态定义是：

> **“Vanilla PEARL 已跑通元任务识别环节，但完整的少样本策略适应链尚未跑通。”**

接下来不需要扩大方法或数据规模，只需要继续沿当前已经定位出的 **Critic → Actor → Outcome** 链路完成最小闭环。

---

## 10. 依据的当前仓库文件

```text
docs/merge_meta_rl_method_flow_current_status.md
pearl_learning/scripts/audit_gate3_vanilla_pearl_mechanism.py
pearl_learning/configs/merge_method_flow_gate3_context_sampling.yaml
pearl_learning/configs/merge_method_flow_gate3_film_critic.yaml
pearl_learning/src/networks.py
pearl_learning/src/pearl_agent.py
pearl_learning/src/pearl_trainer.py
pearl_learning/src/replay.py
```
