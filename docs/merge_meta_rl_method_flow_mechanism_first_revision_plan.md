# Merge 类场景 Meta-RL 方法流跑通 V3：机制优先的代码修改目标与最小实验 Gate

> 适用代码：`sac_scenario_mining/` 与 `pearl_learning/`  
> 本文档状态：**替代上一版“直接进入 150k~250k Method-Flow Pilot”的执行计划，作为当前阶段新的 Codex 目标文档。**  
> 当前固定 SUT：**IDMPolicy，Meta-RL Task 间不改变 IDM 行为参数。**  
> 当前阶段原则：**先证明方法论成立，再扩大 Task 数量、训练预算、baseline 数量和统计重复。禁止通过盲目堆实验量掩盖机制未成立的问题。**

---

# 0. 当前状态与本轮目标调整

上一版目标已经完成了大量工程性改造，并成功跑通：

```text
Task / Case / Action 三层契约
taskbook / casebook
critical metric calibration
20k smoke / 对齐训练
checkpoint
few-shot evaluation
latent-context causal audit
```

因此当前不再把“程序能跑完”作为主要问题。

现有 20k 结果表明：

```text
工程层面：通过
机制层面：未通过
```

更具体地说：

1. `z_prior / z_correct / z_wrong` 对同一 query 的动作与最终结果差异极小；
2. `z_correct` 与 `z_wrong` 的 posterior 距离很小，说明正确 support 与错误 support 没有形成足够不同的 task belief；
3. 将 latent 强制替换为 `z_zero` 时 Actor 会产生明显动作变化，说明 `z -> policy` 计算通路并非完全失效；
4. 因此当前最主要的问题更接近：

```text
support -> posterior 的 task-specific evidence 太弱
以及/或者
当前 Task 在给定 observation / reward 下本来就不需要不同策略
```

5. 当前 Structure-Aware 版本还存在一个潜在 shortcut：

```text
task descriptor -> scenario prior -> z
```

可能已经近似提供 Task ID，使 Context Encoder 更容易退化为：

```text
posterior ≈ prior
```

6. 当前 `logical_merge_dynamic_obs_v1` 已经包含到冲突点距离、速度、到达时间、TTC、相对速度等高度规范化的交互状态，因此不同 Merge Task 可能被转换成了近似相同的纵向控制问题。
7. 当前 v2 casebook 使用 task-specific calibrated threshold 构造相对难度，并通过 heuristic / zero / random 做筛选。这适合作为后续公平 benchmark，但**不适合作为当前“Task 是否真的需要不同策略”的机制验证数据**。
8. Vanilla 对照必须先完成配置溯源修复。任何名为 vanilla 的 run 必须硬性满足：

```text
scenario_representation.enabled = false
scenario_prior.mode = unit_normal
```

否则该 run 不得进入结果比较。

---

# 1. 新的核心问题：先证明 Meta-RL 在当前问题定义下“有必要”

本轮不能直接假设：

```text
不同 logical scenario Task
    =>
不同最优策略
    =>
PEARL 应该产生不同 posterior 和动作
```

真正需要先验证两个必要条件。

## 1.1 条件 A：Task 对策略是有意义的

需要验证：

```text
在相似/匹配的动态状态或初始交互条件下，
不同 Task 是否真的需要不同的 adversarial action strategy。
```

形式化地说，需要存在明显的：

```text
I(A* ; T | S) > 0
```

直观含义：

```text
仅仅知道当前动态状态还不够；
知道 Task 身份后，合理的最优动作会发生变化。
```

如果不同 Task 上同一种纵向策略始终最优，则：

```text
context-invariant policy
```

可能就是合理解，此时继续增加 PEARL 训练步数没有意义。

## 1.2 条件 B：Support context 能够辨认 Task

需要验证：

```text
I(T ; C) > 0
```

即：

```text
少量 support rollout 中的 transition 序列，
确实包含足够信息区分不同 Task。
```

如果独立于 PEARL 的简单分类器都不能从 support trajectory 分辨 Task，则不能要求 PEARL 的 Context Encoder 自动学出稳定 posterior separation。

## 1.3 只有 A 和 B 都成立，才进入 PEARL 机制 Gate

本轮新的主链条为：

```text
Gate 0  Reproducibility / config provenance
        ↓
Gate 1  Task-Policy Conflict
        ↓
Gate 2  Context Identifiability
        ↓
Gate 3  Vanilla PEARL causal mechanism
        ↓
Gate 4  Structure-aware prior
        ↓
Gate 5  小规模完整 Method-Flow Pilot
```

任何前置 Gate 未通过：

```text
立即停止后续大预算训练
```

优先修改：

```text
Task definition
case design
observation
reward signal
support protocol
```

而不是直接增加训练步数或网络容量。

---

# 2. 本轮实验设计原则

## 原则 1：先做“必要性实验”，再做“性能实验”

优先回答：

```text
Task 是否需要不同策略？
Support 是否包含 Task 信息？
Posterior 是否真正影响策略？
```

暂时不优先回答：

```text
最终 VCSR 能提高多少？
是否显著优于所有 baseline？
```

## 原则 2：每个 Gate 只改变一个核心因素

例如：

```text
Policy-conflict audit:
只改变 Task
固定 matched initial condition
固定 scripted policy set
```

```text
Context-identifiability audit:
固定 probing policy
只观察 Task -> trajectory information
```

```text
Vanilla PEARL:
unit-normal prior
无 Scenario Encoder
```

这样失败后可以明确定位原因。

## 原则 3：先使用 2 个“最容易产生差异”的 Task

当前不要一开始就恢复：

```text
4 meta-train
2 validation
2 meta-test
多 logical types
```

第一轮只使用两个物理差异最大的、当前代码已稳定支持的 Task，例如：

```text
T_A = lane_drop_merge, merge_length = 24 m
T_B = bottleneck_merge, merge_length = 32 m
```

如果这两个 Task 都无法形成策略冲突，那么扩大到更多近似 Task 没有意义。

## 原则 4：机制 casebook 与 benchmark casebook 分开

保留当前：

```text
logical_merge_casebook_v2
```

用于后续公平 benchmark。

新增独立：

```text
logical_merge_mechanism_casebook_v3
```

目的不是难度对齐，而是：

```text
最大化 Task-dependent decision boundary 的可见性。
```

两者不能混用。

## 原则 5：尽可能 matched physical cases

机制验证时，Task 间尽可能使用相同：

```text
SUT initial speed
adversary initial speed
arrival-time gap
distance-to-conflict
relative speed
```

只改变 Task 的结构/逻辑参数。

要回答的问题是：

```text
“相同交互条件，仅因 Task 不同，最合理的 adversarial action 是否改变？”
```

## 原则 6：先跑便宜策略，再训练 RL

先使用：

```text
scripted longitudinal policies
```

检查 Task 是否存在策略冲突。

只有 scripted audit 显示冲突后，才训练：

```text
quick single-task SAC
```

只有 single-task policy transfer matrix 有明显对角优势后，才进入 PEARL。

---

# 3. Gate 0：配置溯源与结果目录完整性

这是所有后续实验的前置条件。

## 3.1 必须修复 run provenance

每个 run 启动时必须保存：

```text
requested_config_path
source_config_sha256
resolved_config_sha256
git_commit_sha
taskbook_hash
casebook_hash
critical_threshold_hash
run_name
run_kind
training_seed
```

建议统一写入：

```text
run_manifest.json
```

## 3.2 Vanilla run 增加硬断言

当：

```text
run_name / run_kind 包含 vanilla
```

必须：

```python
assert scenario_representation.enabled is False
assert scenario_prior.mode == "unit_normal"
```

否则直接报错，不允许训练。

## 3.3 Structure-aware run 增加对应断言

```python
assert scenario_representation.enabled is True
assert scenario_prior.mode == "task_conditioned"
```

## 3.4 禁止无条件复用旧输出目录

默认：

```text
output directory exists
    =>
拒绝运行
```

只有显式：

```text
--resume
```

并且：

```text
resolved_config_sha256
taskbook_hash
casebook_hash
```

全部一致时才能继续。

## Gate 0 通过标准

必须自动化测试：

```text
vanilla config -> resolved vanilla config
structure config -> resolved structure config
两个 run 的 resolved hash 不同
错误配置/错误目录复用会触发失败
```

Gate 0 不通过时，禁止进入任何结果解释。

---

# 4. Mechanism Casebook V3：不再做 Task-specific 难度归一化

## 4.1 为什么必须新增机制 casebook

当前 v2 casebook 使用：

```text
target_gap = multiplier × task_specific_threshold
```

这会把不同 Task 主动归一化到近似相同的相对风险位置。

这对最终 benchmark 是合理的，但可能削弱：

```text
Task -> different action strategy
```

的可见性。

## 4.2 V3 casebook 的目标

机制 casebook 使用统一的绝对物理网格，例如：

```text
arrival_time_gap_s ∈ {-0.8, -0.4, 0.0, +0.4, +0.8}
relative_speed_mps ∈ {-2.0, 0.0, +2.0}
```

第一轮不需要全组合。

只选择约：

```text
8~16 个 matched initial conditions
```

覆盖：

```text
adversary earlier
near-simultaneous arrival
SUT earlier
```

三类边界区域即可。

## 4.3 不使用 heuristic-reachable 固定比例

机制 casebook 不强制：

```text
每个 Task 3/8 heuristic reachable
```

也不按 Task 独立调节达到同样成功率。

筛选只做基本可执行性：

```text
车辆不初始重叠
双方均在 conflict point 前
episode 可正常运行
存在潜在时序交互
```

不要筛选到：

```text
各 Task 结果分布看起来一样
```

## 4.4 建议代码实现

新增：

```text
pearl_learning/src/mechanism_casebook.py
```

或在现有 `casebook_v2.py` 中加入严格隔离的：

```text
generation_mode = benchmark | mechanism
```

更推荐独立文件，避免后续误用。

新增脚本：

```text
pearl_learning/scripts/build_mechanism_casebook.py
```

输出：

```text
results/.../mechanism_assets/casebooks/
```

manifest 必须写：

```text
schema = logical_merge_mechanism_casebook_v3
purpose = mechanism_identifiability
task_specific_risk_normalization = false
```

---

# 5. Gate 1：Task-Policy Conflict Audit

这是当前最重要的第一项实验。

## 5.1 第一阶段先不用 SAC

只使用 5~7 个极简单的纵向策略。

例如：

```text
P0: coast / zero longitudinal residual
P1: constant moderate acceleration
P2: constant strong acceleration
P3: constant moderate braking
P4: early accelerate -> late coast/brake
P5: early brake -> late accelerate
P6: TTC / arrival-gap heuristic policy
```

横向控制全部由 route tracker 负责。

## 5.2 本 Gate 建议临时使用 1D action

机制阶段可以增加配置：

```yaml
control:
  mechanism_longitudinal_only: true
```

Actor / scripted policy 只输出：

```text
a_longitudinal
```

steering 完全由已有 route-tracking controller 生成。

目的：

```text
降低动作空间
消除横向学习噪声
使 Task-dependent action 具有直接物理解释
```

正式完整实验再恢复 2D action。

## 5.3 运行方式

对两个 Task：

```text
T_A
T_B
```

使用完全相同的 matched mechanism cases：

```text
x_1 ... x_N
```

运行：

```text
policy P_k × Task T_i × case x_j
```

记录：

```text
return
valid critical
collision
invalid
min TTC
arrival gap trajectory
longitudinal action trajectory
```

## 5.4 主要分析对象

构造：

```text
J(T_i, P_k)
```

以及每个 matched case 上的策略排序：

```text
rank_i,j(P_k)
```

重点不是寻找“最好 scripted policy”，而是观察：

```text
T_A 与 T_B 是否偏好不同动作模式。
```

## 5.5 Gate 1 工程通过标准

以下标准只作为机制 sanity gate，不作为论文统计阈值。

至少满足其中两项：

```text
A. 两个 Task 的 aggregate best scripted policy 不相同；
B. ≥ 30% matched cases 的 top-2 policy 排序发生明显翻转；
C. 某个 policy 在 T_A 上改善关键指标、但在 T_B 上明显恶化；
D. longitudinal action strategy 的最优方向在部分 matched cases 上相反。
```

如果：

```text
同一个 scripted policy 在两个 Task 上几乎全局占优
```

则 Gate 1 失败。

## 5.6 Gate 1 失败后的动作

禁止训练 PEARL。

优先依次检查：

```text
1. Task 参数是否真的改变冲突机制；
2. observation 是否已经把 Task 差异完全 canonicalize；
3. reward 是否只依赖通用 TTC / distance，因此无需 Task-specific strategy；
4. mechanism cases 是否覆盖真正决策边界；
5. 固定 IDM 下是否缺少足够交互异质性。
```

只修改其中最小必要因素，再重跑 Gate 1。

---

# 6. Gate 1B：Quick Single-Task SAC Transfer Matrix

只有 scripted Gate 1 通过后执行。

## 6.1 目的

确认：

```text
不仅 scripted policy 存在冲突，
RL 学到的近似最优策略也具有 Task specificity。
```

## 6.2 训练预算

仅两个 Task：

```text
T_A
T_B
```

每个：

```text
10k~20k environment steps
1 seed
```

先从 10k 开始。

如果 loss / policy 仍明显未形成，再增加到 20k。

不要一开始跑 50k×多 seed。

## 6.3 构造 2×2 transfer matrix

训练：

```text
π_A on T_A
π_B on T_B
```

评估：

```text
M_AA = π_A on T_A
M_AB = π_A on T_B
M_BA = π_B on T_A
M_BB = π_B on T_B
```

使用相同机制 query cases。

## 6.4 Gate 1B 通过标准

希望看到：

```text
M_AA > M_BA
M_BB > M_AB
```

至少在：

```text
return
或 VCSR / critical search objective
```

中的一个主要目标上存在一致对角优势。

同时记录：

```text
π_A vs π_B longitudinal action trajectory difference
```

如果 cross-task policy 与 in-task policy 基本等价，则说明 Task specificity 仍然不足，不进入 PEARL。

---

# 7. Gate 2：Context Identifiability Audit

只有 Gate 1/1B 通过后执行。

## 7.1 目的

独立回答：

```text
support trajectories 是否包含 Task identity 信息？
```

不能用 PEARL 自己作为唯一证据，否则无法区分：

```text
数据不可辨识
vs
PEARL 没学会
```

## 7.2 固定 probing policy

使用一个统一的 probing policy：

```text
scripted heuristic
或
pooled / neutral longitudinal policy
```

两个 Task 必须完全相同。

不能：

```text
T_A 用 π_A
T_B 用 π_B
```

否则 policy 本身会泄漏 Task。

## 7.3 Support 数据量

第一轮：

```text
约 12~20 trajectories / Task
```

即可。

拆分为：

```text
small train split
held-out eval split
```

目标不是最终分类器性能，只是判断信息是否存在。

## 7.4 最小 probe

优先使用最简单模型：

```text
trajectory summary + logistic regression
```

或：

```text
tiny MLP / DeepSets
```

输入只能来自 transition：

```text
obs
action
reward
next_obs
termination
```

不得输入：

```text
task_id
geometry_id
descriptor
case_id
```

## 7.5 同时做非学习式统计

输出：

```text
support feature distribution distance
MMD / energy distance（任选一个简单实现）
```

以及：

```text
arrival-gap trajectory
TTC trajectory
reward trajectory
```

的 Task 间差异。

不要只依赖分类准确率一个数字。

## 7.6 Gate 2 工程通过标准

二分类的简单 probe 在 held-out trajectories 上达到：

```text
accuracy >= 0.80
```

可作为工程 sanity threshold。

同时至少一种 trajectory-level distance 显示稳定 Task separation。

注意：

```text
0.80 不是论文结论阈值，
只用于决定是否值得继续训练 PEARL。
```

## 7.7 Gate 2 失败后的动作

禁止通过：

```text
更大 Context Encoder
更多 latent dimension
更多 PEARL steps
```

直接解决。

优先修改：

```text
support probing policy
support case selection
reward density
observation 中与 Task response 相关的动态信息
```

然后重跑 Gate 2。

---

# 8. Gate 3：真正的 Vanilla PEARL 机制验证

Gate 1 与 Gate 2 均通过后，才开始 PEARL。

## 8.1 第一轮只跑 Vanilla

配置必须：

```yaml
scenario_representation:
  enabled: false

scenario_prior:
  mode: unit_normal
```

不使用：

```text
Scenario Encoder
task-conditioned prior
MoE
disentangled auxiliary supervision
```

原因：

```text
先证明 support -> posterior -> policy 这一条原始链路本身成立。
```

## 8.2 训练规模

仍然只用两个 Task。

初始预算：

```text
20k environment steps
1 seed
```

如果 20k 已出现明确机制信号，不立即扩大训练量。

如果 20k 完全没有 posterior separation，则先查机制，不直接增加到 200k。

## 8.3 必须新增/强化的训练日志

除现有：

```text
KL
posterior variance
actor loss
critic loss
```

新增：

```text
context_encoder_critic_gradient_norm
prior_precision_mean
evidence_precision_mean
evidence_to_prior_precision_ratio
posterior_prior_mean_l2
cross_task_posterior_mean_l2
```

对于当前 unit-normal prior：

```text
prior_precision
```

仍应可统一计算，方便以后和 structure prior 对比。

## 8.4 必须继续使用 fixed-state causal intervention

对相同 dynamic state bank，比较：

```text
z_prior
z_correct
z_wrong
z_zero
```

新增两个归一化机制指标。

### Context separation ratio

```text
R_ctx =
||mu_correct - mu_wrong||
/
(
0.5 * (
||mu_correct - mu_prior||
+
||mu_wrong - mu_prior||
)
+ eps
)
```

当前失败模式通常表现为：

```text
R_ctx << 1
```

### Action utilization ratio

```text
R_act =
mean ||a(z_correct) - a(z_wrong)||
/
(
mean ||a(z_zero) - a(z_prior)||
+ eps
)
```

它回答：

```text
真实 support 引起的 action shift，
占 Actor 可用 latent sensitivity 的多少比例。
```

## 8.5 Gate 3 工程通过标准

本轮使用相对门槛，不追求最终统计显著性。

至少满足：

```text
1. correct / wrong posterior separation 明显高于当前失败 run；
2. R_ctx >= 0.25；
3. R_act >= 0.05；
4. correct support 与 wrong support 在相同 query 上产生可重复的 action trajectory 差异；
5. 至少部分 matched query cases 的 return / critical outcome 随 correct-vs-wrong context 呈一致方向变化。
```

如果只出现：

```text
latent separation
```

但：

```text
action 不变化
```

则检查 Actor/Critic 对 latent 的利用。

如果：

```text
z_zero 会改变动作
但 correct/wrong 不变
```

则继续定位 Context Encoder / support evidence，而不是 Actor 架构。

---

# 9. Gate 4：Structure-Aware Prior 只在 Vanilla 机制成功后加入

## 9.1 不再一开始给 prior 完整 Task fingerprint

当前第一版 structure-aware prior 可能过强。

新的第一轮结构先验应使用：

```text
coarse descriptor
```

例如只保留：

```text
logical_type
lane-count relation（可选）
```

暂时不把：

```text
精确 merge_length
完整 route identity
高维 topology fingerprint
```

全部交给 prior。

## 9.2 目标关系

希望得到：

```text
coarse structure
    ->
better initial prior
    ->
support evidence 仍然能够明显更新 posterior
```

而不是：

```text
full task descriptor
    ->
prior ≈ task ID
    ->
posterior ≈ prior
    ->
context 失去作用
```

## 9.3 增加 prior-dominance audit

必须记录：

```text
||mu_post - mu_prior||
posterior variance reduction
evidence_to_prior_precision_ratio
correct_wrong posterior distance
```

并加入 warning：

```text
如果 evidence_to_prior_precision_ratio 长期接近 0
或
posterior_prior_mean_l2 长期接近 0
则标记 conditional-prior shortcut risk
```

## 9.4 Gate 4 通过标准

相对 Vanilla：

```text
K=0 / K=1 可以更好或至少不系统退化
```

同时必须满足：

```text
support 仍然产生可测 posterior shift
correct / wrong context 仍然可区分
R_ctx / R_act 不因结构 prior 加入而重新塌缩
```

如果 structure prior 提高 K=0，但让 support adaptation 消失，则不能视为方法机制成功。

---

# 10. Gate 5：小规模完整 Method-Flow Pilot

只有 Gate 0~4 全部通过后，再恢复完整方法流。

## 10.1 Task 数量

第一轮完整 pilot 可以恢复：

```text
meta-train:
lane_drop_24
lane_drop_32
bottleneck_24
bottleneck_32

meta-validation:
lane_drop_40
bottleneck_40

meta-test:
lane_drop_48
bottleneck_48
```

仍然不加入：

```text
on_ramp_srs
y_merge
```

## 10.2 方法数量

只比较：

```text
Pooled / no-context policy
Vanilla Dense PEARL
Structure-Aware Dense PEARL
```

不加入第四种以上主方法。

## 10.3 训练预算

不要直接恢复 150k~250k。

建议：

```text
第一轮 = 50k steps / method / 1 seed
```

检查：

```text
mechanism metrics
validation trend
few-shot adaptation
```

如果 50k 仍在稳定改善且 Gate 指标保持健康，再增加到：

```text
100k
```

只有确实需要时才进入：

```text
150k~250k
```

## 10.4 评估 shots

保持：

```text
K = [0, 1, 2, 4]
```

但第一轮 query 数仍保持小规模：

```text
6~8 query cases / Task
```

不要提前扩到几十个 query 或多 seed。

---

# 11. 当前代码修改清单

以下修改按优先级执行。

## P0：Run provenance 与配置硬约束

修改：

```text
pearl_learning/scripts/train_pearl.py
pearl_learning/src/checkpoint.py
pearl_learning/src/io.py
```

新增：

```text
run_manifest.json
config assertions
output-dir reuse guard
```

同时为：

```text
merge_method_flow_vanilla_pilot.yaml
merge_method_flow_pilot.yaml
```

增加明确 run kind。

## P1：新增机制实验配置

新增：

```text
pearl_learning/configs/merge_method_flow_mechanism.yaml
```

建议扩展已有 pilot config，但覆盖：

```yaml
method_flow:
  mode: mechanism

environment:
  action_dim: 1  # mechanism-only longitudinal control

scenario_representation:
  enabled: false

scenario_prior:
  mode: unit_normal

mechanism:
  task_ids:
    - lane_drop_24
    - bottleneck_32
  matched_case_count: 12
  task_specific_risk_normalization: false
```

实际字段名根据现有 config schema 调整，不要求机械照抄。

## P2：新增 Mechanism Casebook V3

新增：

```text
pearl_learning/src/mechanism_casebook.py
pearl_learning/scripts/build_mechanism_casebook.py
```

或者最小化实现为现有 casebook 模块中的独立路径，但必须有不同 schema/hash。

## P3：新增 Task-Policy Conflict Audit

优先复用现有：

```text
run_single_task_sac_diagnostic.py
audit_task_heterogeneity.py
```

不要重复开发已有功能。

建议新增或扩展：

```text
pearl_learning/scripts/audit_task_policy_conflict.py
```

输出：

```text
scripted_policy_matrix.json
scripted_policy_case_rankings.json
policy_conflict_gate.json
```

## P4：扩展 Quick Single-Task SAC 诊断

复用：

```text
run_single_task_sac_diagnostic.py
```

增加：

```text
--mechanism-casebook
--task-id
--steps 10000
```

并增加统一评估脚本生成：

```text
2x2_policy_transfer_matrix.json
```

不新增完整 baseline pipeline。

## P5：新增 Context Identifiability Audit

新增：

```text
pearl_learning/scripts/audit_context_identifiability.py
```

要求：

```text
固定 probing policy
trajectory-only features
无 descriptor / task label 输入泄漏
```

输出：

```text
context_probe_metrics.json
context_feature_distance.json
context_identifiability_gate.json
```

## P6：强化 Latent Context Causal Audit

现有：

```text
audit_latent_context_interventions.py
```

继续使用，不重写。

新增：

```text
R_ctx
R_act
posterior_prior_mean_l2
correct_wrong_posterior_l2
evidence_to_prior_precision_ratio
```

输出 schema 升级，例如：

```text
latent_context_causal_audit_suite_v2
```

## P7：训练日志补充 Encoder 梯度与证据强度

修改：

```text
pearl_learning/src/pearl_agent.py
pearl_learning/src/context_encoder.py
```

新增：

```text
context_encoder_critic_gradient_norm
prior_precision_mean
evidence_precision_mean
```

注意：

```text
context_encoder_actor_gradient_norm == 0
```

在当前 actor detach 设计下并不等价于 bug。

真正需要审计的是：

```text
critic / encoder phase 是否给 Context Encoder 足够 gradient。
```

## P8：Structure prior 改成 coarse-first

修改：

```text
pearl_learning/src/scenario_encoder.py
```

增加 descriptor profile：

```yaml
scenario_representation:
  descriptor_profile: coarse | full
```

当前 Gate 4 只允许：

```text
coarse
```

`full` 留到方法链验证成功以后。

---

# 12. 必须新增的单元/契约测试

建议扩展：

```text
pearl_learning/tests/test_method_flow_v2.py
pearl_learning/tests/test_scenario_prior.py
pearl_learning/tests/test_contract.py
```

至少包含：

## 配置测试

```text
vanilla run 不能解析成 task_conditioned prior
structure run 不能解析成 unit_normal prior
旧输出目录 hash 不一致必须拒绝 resume
```

## mechanism casebook 测试

```text
两个 Task 的 matched case 使用同一目标物理条件
不使用 task-specific threshold multiplier
mechanism schema 与 benchmark schema 不同
```

## leakage 测试

```text
context probe 不读取 task descriptor
scenario prior 不读取 query outcome
support/query case 不重叠
```

## causal audit 测试

构造 toy actor：

```text
z 变化 -> action 变化
```

确保：

```text
R_act
R_ctx
```

计算正确。

---

# 13. 每个 Gate 的停止规则

这是本轮最重要的实验管理规则。

| Gate | 最大初始成本 | 失败后禁止做什么 | 下一步 |
|---|---:|---|---|
| Gate 0 配置溯源 | 单元测试级 | 禁止解释旧 vanilla 对照 | 修 manifest/config |
| Gate 1 scripted policy conflict | 2 Tasks × 8~16 cases × 5~7 policies | 禁止训练 PEARL | 改 Task/case/reward |
| Gate 1B quick SAC | 10k steps × 2 Tasks | 禁止扩大到多 Task PEARL | 查 Task specificity |
| Gate 2 context probe | 约 12~20 trajectories/Task | 禁止堆 Context Encoder | 改 support/probing |
| Gate 3 vanilla PEARL | 20k steps | 禁止直接升 200k | 查 posterior evidence |
| Gate 4 structure prior | 20k~30k steps | 禁止上 full descriptor/MoE | 修 prior strength |
| Gate 5 full pilot | 50k steps/method | 禁止直接多 seed/大 query | 看 validation 后续决定 |

---

# 14. 本轮明确暂缓的内容

当前以下内容全部不做：

```text
× Posterior-routed MoE
× 多 expert 数量搜索
× GNN / Transformer Scenario Encoder
× learned relatedness
× active support selection
× 多 SUT controller
× IDM parameterized task family
× y-merge / on-ramp OOD 全量
× 3~5 training seeds
× 20~60 query cases / Task
× 1.5M training steps
× 大规模 hyperparameter sweep
× 同时调 latent_dim / KL / network depth / reward
```

只有 Gate 5 成功后，才进入正式论文实验设计。

---

# 15. 建议的最小结果目录

```text
results/pearl_learning/merge_method_flow_mechanism/
  run_manifest.json

  mechanism_assets/
    casebooks/
      ...
    mechanism_casebook_manifest.json

  gate_1_policy_conflict/
    scripted_policy_matrix.json
    scripted_policy_case_rankings.json
    policy_conflict_gate.json

  gate_1b_single_task/
    task_A/
    task_B/
    policy_transfer_matrix.json

  gate_2_context_identifiability/
    context_probe_metrics.json
    context_feature_distance.json
    context_identifiability_gate.json

  gate_3_vanilla_pearl/
    config_resolved.json
    training_updates.jsonl
    latent_context_audit.json
    mechanism_gate.json

  gate_4_structure_prior/
    config_resolved.json
    training_updates.jsonl
    latent_context_audit.json
    mechanism_gate.json

  gate_5_method_flow/
    pooled/
    vanilla/
    structure/
    fewshot_metrics.json
    pilot_summary.json
```

---

# 16. Gate 汇总文件

新增统一：

```text
mechanism_gate.json
```

建议 schema：

```json
{
  "gate_name": "vanilla_pearl_context_mechanism",
  "status": "pass | fail | inconclusive",
  "inputs": {
    "config_hash": "...",
    "taskbook_hash": "...",
    "casebook_hash": "..."
  },
  "metrics": {
    "R_ctx": 0.0,
    "R_act": 0.0,
    "correct_wrong_posterior_l2": 0.0,
    "correct_wrong_action_l2": 0.0
  },
  "stop_reason": "...",
  "next_allowed_stage": "..."
}
```

Codex 运行脚本必须根据 Gate 状态决定是否提示进入下一阶段。

不建议自动连续执行所有阶段。

---

# 17. 推荐开发与实验顺序

严格按照：

```text
Step 1
修复 vanilla / structure 配置溯源
        ↓
Step 2
构建 2-Task matched mechanism casebook
        ↓
Step 3
scripted longitudinal policy conflict audit
        ↓
[FAIL] -> 修改 Task / case / reward
[PASS]
        ↓
Step 4
2× quick single-task SAC + 2×2 transfer matrix
        ↓
[FAIL] -> 停止 PEARL
[PASS]
        ↓
Step 5
fixed probing policy context-identifiability audit
        ↓
[FAIL] -> 修改 support protocol
[PASS]
        ↓
Step 6
真正 Vanilla PEARL 20k
        ↓
Step 7
fixed-state z_prior/z_correct/z_wrong/z_zero causal audit
        ↓
[FAIL] -> 定位 support->posterior 或 z utilization
[PASS]
        ↓
Step 8
coarse Structure-Aware prior
        ↓
Step 9
确认 support adaptation 没有被 prior shortcut 压掉
        ↓
Step 10
4/2/2 Task 小规模 Method-Flow 50k
        ↓
Step 11
只有完整方法链成立，再设计正式论文实验
```

---

# 18. 本轮最终“成功”的定义

本轮成功不要求：

```text
Structure-Aware PEARL 已显著优于所有 baseline。
```

真正需要得到如下证据链：

## 证据 1：Task 有策略意义

```text
相同/匹配交互条件下，
不同 Merge Task 存在可重复的策略偏好差异。
```

## 证据 2：Support 有 Task 信息

```text
固定 probing policy 的少量 trajectory
可以区分 Task。
```

## 证据 3：Vanilla PEARL 真正利用 Context

```text
correct support
    ->
different posterior
    ->
different action trajectory
    ->
query outcome 有方向一致的变化
```

## 证据 4：Structure prior 是“辅助”，不是“替代 Context”

```text
结构 prior 改善少样本初始化，
但 support 仍然能够显著修正 posterior。
```

## 证据 5：完整小规模方法流可复现

```text
Pooled
Vanilla PEARL
Structure-Aware PEARL
```

在相同 Task/case/protocol 下完成训练与 few-shot query evaluation。

---

# 19. 成功以后才允许进入的下一阶段

只有上述证据链成立，下一阶段才能讨论：

```text
多 training seeds
更大 query 数
更长训练
unseen logical type
Y-merge / on-ramp
GNN Scenario Encoder
learned task relatedness
Posterior-routed MoE
多 SUT 泛化
正式统计显著性
```

正式论文实验的扩大必须服务于：

```text
验证已经成立的方法
```

而不能用于：

```text
寻找某一个偶然能使机制看起来成立的配置。
```

---

# 20. Codex 执行原则

Codex 在后续修改中应遵守以下约束：

1. **优先复用现有脚本与 audit 工具，不重复造轮子。**
2. **每次提交只解决一个 Gate 所需的问题。**
3. **任何 Gate 未通过，不自动启动下一阶段大预算实验。**
4. **所有机制判断必须输出机器可读 JSON，而不是只打印日志。**
5. **benchmark casebook 与 mechanism casebook 严格隔离。**
6. **Vanilla / Structure run 必须由 config assertion 保证真实匹配。**
7. **先解决 Task-policy relevance，再解决 PEARL 优化。**
8. **先证明 Vanilla context adaptation，再引入 structure prior。**
9. **Structure prior 第一版使用 coarse descriptor，避免 Task-ID shortcut。**
10. **当前阶段不进行大规模超参数搜索。**
11. **如果 2 个极端 Task 都没有策略冲突，应优先重新定义 Task，而不是增加 Task 数量。**
12. **所有实验量的增加必须由前一阶段 Gate 的结果明确触发。**

---

# 21. 本轮最核心的判断标准

本轮不再把：

```text
20k / 50k / 200k 是否跑完
```

作为成功标准。

真正的判断链是：

```text
Task 不同
    ↓
最优策略确实不同
    ↓
support trajectory 能辨认这种不同
    ↓
posterior 因 support 而改变
    ↓
policy 因 posterior 而改变
    ↓
query 搜索结果随正确 context 改善
```

只有这条链跑通以后，才进入：

```text
Structure-aware Meta-RL
多 Task
OOD
MoE / GNN
正式大规模实验
```

因此当前阶段的核心目标应表述为：

> **在固定 IDM SUT 与 Merge 功能场景族下，首先通过最小成本的 matched-case 策略冲突审计证明不同逻辑/几何 Task 对 adversarial policy 具有真实决策意义；随后证明少量 support trajectory 包含可辨识的 Task 信息；在这两个必要条件成立后，再验证 Vanilla PEARL 的 `support -> posterior -> policy -> query outcome` 因果链，并最终检验 coarse logical structure 是否能够作为有益但不替代交互证据的 task prior。整个过程采用逐级 Gate 机制，任何前置机制失败时停止增加训练规模，从问题定义或信息通路本身进行修正。**

这就是当前代码与实验状态下，下一轮最应该优先实现和跑通的方法论闭环。
