# Merge 类元强化学习方法：近期讨论后的最小必要修改方案

GPU环境在系统的
conda activate metadrive

> 当前目标：**不是继续死磕 hidden-order mechanism pair，而是以逻辑场景设计为核心，尽快跑通“逻辑场景 → Meta-RL → 少样本适应 → 危险场景挖掘”的完整方法链。**  
> 本文只保留当前阶段必要的修改，不扩大到正式论文级全量实验。

---

## 1. 当前最新状态

最新一轮已经完成 `latent_film_gamma_only`：

- 训练约 20k 环境步；
- `Stage A: Context → task-discriminative posterior` 已通过；
- `z_correct` 与 `z_wrong` 已明显分离；
- `correct/wrong action L2 ≈ 0.145`，说明 latent 已经能显著改变 Actor 输出；
- 但当前 v4 Gate 中：
  - `critic_argmax_action_distance = 0.0`
  - `Stage B_Q = FAIL`
  - `Stage B_pi` 因 B_Q 被阻塞；
  - `Stage C` 也被阻塞。

因此，当前状态不应再描述为：

```text
context / latent 完全不起作用
```

更准确的结论是：

\[
\boxed{
C_{\text{correct}}\neq C_{\text{wrong}}
\rightarrow
z_{\text{correct}}\neq z_{\text{wrong}}
\rightarrow
\pi(s,z_{\text{correct}})
\neq
\pi(s,z_{\text{wrong}})
}
\]

已经出现。

当前真正尚未验证的是：

\[
\boxed{
\pi(s,z_{\text{correct}})
\rightarrow
\text{更正确的 task-specific 场景挖掘结果}
}
\]

也就是 **policy → outcome** 这一段。

---

# 2. 对当前 Gate 3 判定逻辑的必要修正

## 2.1 `B_Q` 不再作为硬门禁

当前 Gate v4 的链条是：

\[
A\rightarrow B_Q\rightarrow B_\pi\rightarrow C
\]

其中 `B_Q` 要求：

\[
|\arg\max_a Q(s,a,z_c)-\arg\max_a Q(s,a,z_w)|\ge 0.1
\]

但 SAC 的 Actor 优化目标是：

\[
\mathcal L_\pi
=
\mathbb E[
\alpha\log\pi(a|s,z)-Q(s,a,z)
]
\]

并不要求：

\[
\pi(s,z)=\arg\max_a Q(s,a,z)
\]

因此：

```text
correct/wrong latent 下的 Q-grid hard argmax 必须不同
```

不应该作为 PEARL/SAC 成立的必要条件。

### 修改意见

将 Gate 3 改为：

\[
\boxed{
A\rightarrow B_\pi\rightarrow C
}
\]

其中：

- **Stage A**：context 是否产生 task-discriminative posterior；
- **Stage Bπ**：posterior 是否改变 policy；
- **Stage C**：correct context 是否带来更好的 task-specific 场景挖掘结果。

`B_Q` 保留，但改成：

```text
diagnostic_only
```

用于诊断 Critic，而不是阻塞 Actor / outcome 判定。

---

## 2.2 继续保留的 Gate 指标

### Stage A

保持现有：

\[
D_z=\|\mu_c-\mu_w\|_2
\]

以及 prior-relative separation：

\[
R_{\mathrm{sep}}
=
\frac{
\|\mu_c-\mu_w\|_2
}{
0.5(
\|\mu_c-\mu_p\|_2+
\|\mu_w-\mu_p\|_2
)+\epsilon
}
\]

不需要再改 Context Encoder。

### Stage Bπ

直接使用：

\[
D_\pi
=
\frac{1}{N}
\sum_s
\|\pi(s,z_c)-\pi(s,z_w)\|_2
\]

当前 Gamma-only 已经约为：

```text
adversary_first ≈ 0.145
sut_first       ≈ 0.145
```

原阈值 0.1 下，这一段已经具备 PASS 的证据。

### Stage C

真正需要继续判断：

\[
J(T,z_c)>J(T,z_w)
\]

建议优先看：

```text
VCSR
valid critical rate
episode return
invalid rate
target collision rate
episodes / env-steps to first valid critical
```

---

# 3. 不再继续修改 Critic 网络结构

已经测试：

```text
Dense Critic
Latent-FiLM Critic
Gamma-only FiLM Critic
```

继续增加：

```text
Cross-attention Critic
Bilinear Critic
Hypernetwork Critic
MoE Critic
```

在当前阶段没有足够必要性。

Gamma-only 已经说明：

\[
z\rightarrow \pi
\]

不是死通路，因此后续不应继续把主要精力放在 Critic 架构搜索上。

---

# 4. 当前 hidden logical-order pair 的重新定位

现有两个 mechanism Task：

```text
lane_drop_24 + adversary_first
lane_drop_24 + sut_first
```

具有：

```text
相同地图
相同几何
相同路线
相同 IDM
matched initial cases
相同 observation
```

主要区别只来自隐藏的：

```text
conflict-entry order / reward semantics
```

这会造成一个非常困难的 Meta-RL 压力测试：

- 大部分 transition 在两个 Task 中很相似；
- 真正 task-sensitive 的信息主要集中在 terminal / conflict-near 区域；
- replay audit 中这类 transition 仅约占 6%；
- PEARL 必须从少量后期交互证据中推断隐藏规则。

因此该 pair 不应继续作为“完整方法必须通过的主逻辑场景”。

### 新定位

将其保留为：

\[
\boxed{\text{PEARL mechanism stress test}}
\]

其用途是证明：

```text
context sampling 会影响 posterior identifiability
C → z 可以成立
z → policy 可以成立
稀疏 task-sensitive transition 会显著增加 Meta-RL 难度
```

但不再要求它承担：

```text
完整逻辑场景迁移
Structure-Aware Prior
Transferability
MoE
```

这些主方法验证。

---

# 5. 主实验需要重新定义“逻辑场景”

## 5.1 新原则

主实验中的 Meta-RL Task 应满足：

\[
\boxed{
\text{Task difference 来自物理 / 逻辑场景，
而不是来自不同 reward semantics}
}
\]

即所有 Task 统一：

```text
固定 IDM
统一 reward
统一 critical scenario 定义
统一 action semantics
统一 observation schema
统一 horizon
统一 SAC / PEARL 算法
```

变化的是：

\[
\lambda_i=\text{Merge 逻辑场景参数}
\]

因此任务应表示为：

\[
T_i=T(\lambda_i,p(x|\lambda_i),R,\Omega)
\]

其中：

- \(\lambda_i\)：Task-level 逻辑场景参数；
- \(x_{ij}\)：Episode-level 初始条件；
- \(a_{ijt}\)：Step-level adversarial action；
- \(R,\Omega\)：所有 Task 共享的 reward / 协议。

---

# 6. 不需要开发新地图，优先复用已有 Merge 物理任务

当前仓库已经具备：

```text
lane_drop_24
lane_drop_32
lane_drop_40
lane_drop_48

bottleneck_24
bottleneck_32
bottleneck_40
bottleneck_48

y_merge_*
```

第一阶段直接复用，不新建 MetaDrive 地图。

---

# 7. 最小主实验任务设计

## 7.1 Meta-train

建议使用 4 个 Task：

| Task | logical subtype | merge length |
|---|---|---:|
| T1 | lane-drop | 24 m |
| T2 | lane-drop | 32 m |
| T3 | bottleneck | 24 m |
| T4 | bottleneck | 32 m |

即：

```text
lane_drop_24
lane_drop_32
bottleneck_24
bottleneck_32
```

## 7.2 Unseen validation

使用：

```text
lane_drop_40
bottleneck_40
```

即：

| Task | logical subtype | merge length |
|---|---|---:|
| T5 | lane-drop | 40 m |
| T6 | bottleneck | 40 m |

第一轮暂时不要加入：

```text
lane_drop_48
bottleneck_48
y_merge
```

它们留作后续 test / OOD。

---

# 8. Task descriptor 的最小定义

第一版不要把 \(\lambda\) 设计得太复杂。

建议：

\[
\boxed{
\lambda_i=
[
\text{merge subtype},
L_{\mathrm{merge}}
]
}
\]

编码方式：

```text
merge subtype -> one-hot
merge length  -> normalized scalar
```

Scenario Encoder：

\[
e_i=E_\omega(\lambda_i)
\]

暂时继续使用小型 MLP：

```text
input
→ 32
→ 16
→ e (8-dim)
```

第一版不使用 GNN / Transformer。

后续只有在真正加入：

```text
lane graph connectivity
branch count
multi-conflict-zone topology
```

时，再考虑 GNN。

---

# 9. Episode-level Case 参数继续随机

一个 Task 不是一个固定初始状态。

保持：

\[
x_{ij}\sim p(x|\lambda_i)
\]

第一版可使用：

\[
x=
[
v_{\mathrm{SUT}}^0,
v_{\mathrm{adv}}^0,
p_{\mathrm{SUT}}^0,
p_{\mathrm{adv}}^0,
\Delta t_{\mathrm{arrival}}^0
]
\]

建议继续：

```text
SUT initial speed        U(10, 14) m/s
Adversary initial speed  U(10, 17) m/s
max initial arrival gap  4 s
background density       0
```

不同 Task 间尽量使用一致的 case distribution envelope：

\[
p(x|\lambda_i)\approx p(x|\lambda_j)
\]

避免产生：

```text
Task A = 高速 case
Task B = 低速 case
```

这种伪 task identity。

---

# 10. 第一轮 Casebook 数据量

当前目的只是跑通方法，不做正式统计。

建议每个 Task：

```text
train_pool           8
validation_support   4
validation_query     4
```

因此：

- 4 个 meta-train Task × 8 train cases；
- 2 个 validation Task × (4 support + 4 query)。

这已经足够做第一轮 Meta-RL 方法验证。

---

# 11. 在重新训练 PEARL 前，必须先做一次 Physical Task Heterogeneity Gate

这是下一阶段最重要的前置检查。

不要再次假设：

```text
不同几何 Task 一定需要不同策略
```

先验证：

\[
\pi_i^*\neq \pi_j^*
\]

## 最小方案

只挑两个代表性 Task：

```text
lane_drop_24
bottleneck_32
```

各训练一个极小 single-task SAC。

然后做 2×2 transfer matrix：

\[
M=
\begin{bmatrix}
J(\pi_A,T_A) & J(\pi_A,T_B)\\
J(\pi_B,T_A) & J(\pi_B,T_B)
\end{bmatrix}
\]

希望至少看到：

\[
J(\pi_A,T_A)>J(\pi_B,T_A)
\]

以及：

\[
J(\pi_B,T_B)>J(\pi_A,T_B)
\]

或至少出现稳定、明显的：

\[
D(\pi_A,\pi_B)>0
\]

### Gate 结果解释

如果 PASS：

\[
\boxed{\text{直接进入 Meta-RL}}
\]

如果 FAIL：

\[
\boxed{\text{先重新设计逻辑场景参数，不要调 PEARL}}
\]

这一步用于防止再次在本来就 policy-invariant 的场景上浪费 Meta-RL 训练时间。

---

# 12. 主实验中必须统一 reward

新的 Physical Merge Task 必须满足：

\[
\boxed{
R_A=R_B=R_C=\cdots=R
}
\]

不要再用：

```text
adversary_first
sut_first
```

作为主 Task identity。

也不要通过不同：

```text
target_contact rule
hidden speed relation
hidden entry order
```

来制造 Task 区别。

主任务应从：

\[
T=(M,R_i)
\]

回到：

\[
\boxed{
T_i=(M_i,R)
}
\]

这样才能清晰证明：

> Meta-RL 适应的是不同自动驾驶逻辑场景，而不是不同奖励函数。

---

# 13. Structure-Aware PEARL 在新任务定义下才真正有意义

新的方法链：

\[
\lambda_i
\xrightarrow{E_\omega}
e_i
\]

\[
e_i
\rightarrow
p_\eta(z|e_i)
\]

\[
(C_i,e_i)
\rightarrow
q_\phi(z|C_i,e_i)
\]

\[
(s_t,z_i)
\rightarrow
\pi_\theta(a_t|s_t,z_i)
\]

其中：

- \(e_i\)：已知逻辑场景结构；
- \(z_i\)：少量 support interaction 提供的经验条件信息。

直观上：

```text
已知：
“这是一个 40m lane-drop Merge”

        ↓

Scenario Encoder

        ↓

结构先验 e

        +

少量 support episodes

        ↓

posterior z

        ↓

快速形成适合该 Merge Task 的危险场景搜索策略
```

这比 hidden logical-order pair 更符合论文主线。

---

# 14. 新的最小方法实验顺序

## Phase 0：冻结 mechanism stress test

停止继续修改：

```text
hidden-order Task
Context Encoder
Critic architecture
Actor architecture
latent dimension
KL beta
```

保存其历史结果，作为 mechanism diagnostic。

---

## Phase 1：Physical Task Heterogeneity Gate

Tasks：

```text
lane_drop_24
bottleneck_32
```

实验：

```text
small single-task SAC
2 × 2 cross-transfer
1 seed
```

目标：

```text
确认 physical logical tasks 确实需要不同策略
```

---

## Phase 2：Vanilla PEARL 最小跑通

Meta-train：

```text
lane_drop_24
lane_drop_32
bottleneck_24
bottleneck_32
```

Validation：

```text
lane_drop_40
bottleneck_40
```

建议：

```text
K = [0, 1, 2, 4]
1 seed
约 100k–200k env steps 上限
validation 每 20k–25k
```

这里只验证：

\[
K>0
\]

是否相对：

\[
K=0
\]

出现稳定适应趋势。

---

## Phase 3：Structure-Aware PEARL

Vanilla PEARL 至少出现合理 adaptation trend 后，再启用：

```text
Scenario Encoder
task-conditioned prior
q(z | C, e)
```

比较：

```text
Vanilla PEARL
vs
Structure-Aware PEARL
```

重点看：

```text
K=0
K=1
K=2
```

因为 Structure-Aware Prior 的核心价值应该体现在小样本阶段。

---

## Phase 4：再做 Transferability

只有在：

```text
physical tasks
Vanilla PEARL
Structure-Aware PEARL
```

都工作后，再研究：

\[
d(e_i,e_j)
\]

与真实：

\[
M_{ij}=J(\pi_i,T_j)
\]

之间的关系。

第一版只做：

```text
Spearman
nearest-neighbor agreement
transfer matrix heatmap
```

不要立即训练深度 relatedness network。

---

## Phase 5：最后再考虑 MoE

只有当 cross-task transfer matrix 表明存在明显策略簇/冲突时，才有理由加入 MoE。

否则继续使用 Dense Actor。

---

# 15. 当前不应该做的事情

当前阶段明确暂缓：

```text
继续为 hidden-order pair 调 Critic
更大 latent
更深 Context Encoder
KL sweep
学习率 sweep
GNN
Transformer
MoE
relatedness network
multi-seed formal experiment
百万级 environment steps
```

---

# 16. 需要修改的代码位置

## P0：Gate 逻辑

### `pearl_learning/scripts/audit_gate3_vanilla_pearl_mechanism.py`

修改：

```text
B_Q:
hard gate
→
diagnostic_only
```

主 Gate 改为：

```text
Stage A  context → posterior
Stage B  posterior → actor action
Stage C  actor → outcome
```

---

## P0：主实验 Task 定义

### `pearl_learning/configs/merge_method_flow_pilot.yaml`

重新作为主实验入口，保持：

```text
lane_drop_24
lane_drop_32
bottleneck_24
bottleneck_32
→ train

lane_drop_40
bottleneck_40
→ validation
```

确认：

```text
allow_hidden_reward_rule_variants: false
```

必须真正生效。

---

## P0：Taskbook

### `pearl_learning/src/taskbook.py`

主实验构造时禁止：

```text
__logical_order_*
__rule_*
hidden target-contact reward variants
```

确保一个物理逻辑场景对应一个 Meta-RL Task。

---

## P0：Reward

### `pearl_learning/src/task_env.py`
### `pearl_learning/src/reward.py`

主 Physical Merge 配置下：

```text
不要读取 hidden logical order 来改变 reward
不要读取 hidden speed-relation 来改变 reward
```

主实验 reward 必须对所有 Task 完全一致。

---

## P1：新增 Physical Task Heterogeneity Gate

建议新增：

```text
pearl_learning/scripts/audit_physical_task_policy_heterogeneity.py
```

功能：

```text
train / load two small single-task SACs
evaluate 2×2 cross-transfer
fixed matched query cases
report VCSR / return / action distance
```

只用于在 PEARL 前确认 task-policy conflict。

---

## P1：Scenario Encoder

保留现有：

```text
pearl_learning/src/scenario_encoder.py
```

第一版只输入 Task-level \(\lambda\)，不要输入：

```text
case id
case seed
initial speed sample
support result
query result
reward label
```

---

# 17. 推荐的实验停止条件

## Physical Task Gate FAIL

如果：

```text
lane_drop_24
bottleneck_32
```

的 single-task SAC cross-transfer 几乎没有差异：

```text
停止 PEARL
→ 修改逻辑场景参数
```

不要调神经网络。

---

## Vanilla PEARL FAIL

如果 Physical Task Gate PASS，但：

```text
K=0/1/2/4
```

完全没有 adaptation：

先检查：

```text
support → posterior
posterior → policy
```

而不是立即加 Structure-Aware Prior。

---

## Structure-Aware FAIL

如果 Vanilla PEARL 能适应，而 Structure-Aware 没有改善：

才检查：

```text
task descriptor
scenario encoder
conditional prior
```

---

# 18. 最终推荐的主方法路线

```text
Functional Scenario: Merge
        │
        ▼
Physical Logical Tasks
        │
        ├── lane_drop_24
        ├── lane_drop_32
        ├── bottleneck_24
        └── bottleneck_32
        │
        ▼
Physical Task Heterogeneity Gate
        │
        ▼
Vanilla PEARL
        │
        ▼
Unseen Physical Tasks
        ├── lane_drop_40
        └── bottleneck_40
        │
        ▼
K-shot Adaptation
        │
        ▼
Critical Scenario Mining
        │
        ▼
Structure-Aware Prior
        │
        ▼
Transferability
        │
        ▼
MoE（仅在确有策略簇时）
```

---

# 19. 当前阶段最终结论

当前不再把“跑通方法”绑定在：

```text
同一 lane_drop_24
+
两个隐藏 logical-order reward task
```

上。

这组任务保留为机制压力测试即可。

主线改为：

\[
\boxed{
\text{Physical Merge Logical Scenario}
\rightarrow
\text{Meta-RL Task}
\rightarrow
\text{Few-shot Adaptation}
\rightarrow
\text{Critical Scenario Mining}
}
\]

推荐的最小任务集合：

\[
\boxed{
\text{Train: }
\{\text{lane-drop}_{24},
\text{lane-drop}_{32},
\text{bottleneck}_{24},
\text{bottleneck}_{32}\}
}
\]

\[
\boxed{
\text{Validation: }
\{\text{lane-drop}_{40},
\text{bottleneck}_{40}\}
}
\]

这一修改能够同时满足：

1. Task 间存在真实物理/逻辑差异；
2. reward 语义保持一致；
3. Scenario Encoder 有实际意义；
4. Structure-Aware Prior 有合理输入；
5. Transferability 可以有可解释的 task distance；
6. 与最终“自动驾驶逻辑场景挖掘”的论文问题一致；
7. 不需要重新开发 MetaDrive 地图；
8. 仍能保持很小的实验规模。

---

## 一句话执行建议

\[
\boxed{
\text{停止继续优化 hidden-order pair，先验证 physical-task policy heterogeneity；}
}
\]

\[
\boxed{
\text{一旦通过，直接进入 4-train + 2-validation 的 Physical Merge PEARL pilot。}
}
\]
