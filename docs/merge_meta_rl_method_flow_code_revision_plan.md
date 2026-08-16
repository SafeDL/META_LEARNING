# Merge 类场景 Meta-RL 方法流跑通：代码修改目标与最小实验方案
 
> 适用代码：本工程中的 `sac_scenario_mining/` 与 `pearl_learning/`  
> 当前目标：**先跑通完整方法论，不追求全量数据、完整统计显著性、跨 SUT 泛化或最终论文性能。**  
> 当前固定 SUT：**IDMPolicy，所有 Meta-RL Task 使用同一套 IDM 行为参数。**

---

## 1. 本轮修改的核心目标

本轮不继续扩大模型复杂度，而是先把方法论的三层参数契约、训练/适应流程和数据隔离真正落实到代码中。最终需要跑通下面这条最小闭环：

```text
Functional Scenario Family: Merge
            │
            ▼
   Meta-RL Task parameter λ_i
  （Task 级逻辑/几何配置）
            │
      ┌─────┴───────────────┐
      │                     │
      ▼                     ▼
Scenario Encoder        Case sampler
 e_i = Eω(λ_i)          x_ij ~ p_i(x)
      │                     │
      ▼                     ▼
Conditional prior        reset env
 pη(z|e_i)                  │
      │                     │
      └────────┐            │
               ▼            ▼
          PEARL Actor π(a|s,z)
               │
               ▼
       adversarial action a_t
               │
               ▼
        MetaDrive + fixed IDM
               │
               ▼
       transition (s,a,r,s')
               │
               ▼
          support context C
               │
               ▼
      qφ(z | C, e_i)
               │
               └────► updated task-conditioned policy
```

本轮要证明的不是“最终模型已经优于所有方法”，而是三个事实：

1. **一个 Meta-RL Task 内仍包含大量不同的 SAC episode/case。**
2. **PEARL 能用少量 support cases 的交互轨迹形成 task posterior，并改善新 Task 上的 query 搜索。**
3. **测试前已知的逻辑场景结构编码 `e_i` 能够作为 PEARL 的结构先验，而不是与 episode case 参数混为一谈。**

MoE、GNN/Transformer、learned task relatedness 暂时不进入本轮主训练闭环。

---

# 2. 统一三层参数定义

后续代码、配置、实验记录和论文描述应统一使用三个符号：

## 2.1 第一层：Meta-RL Task 参数 `λ_i`

```text
变化时间尺度：只有切换 Meta-RL Task 时变化；
在一个 Task 的整个 SAC/PEARL 训练和评估过程中保持固定。
```

记为：

```text
T_i = T(λ_i)
```

它描述的是 **Merge 类功能场景内部更高层次的逻辑/物理结构配置**。

本轮推荐只使用现有代码已经支持的变量，避免重新开发地图：

```text
λ_i = {
    logical_type,          # lane_drop_merge / bottleneck_merge
    bottle_lane_num,
    neck_lane_num,
    merge_length_m,
    route connectivity / route ids,
    conflict geometry
}
```

本轮最主要的两个变化轴为：

```text
A. merge subtype / lane-connectivity structure
B. merge_length_m
```

这些变量当前已经存在于 `pearl_learning/configs/dense_pearl_baseline.yaml` 的 geometry catalog 中，因此适合作为第一轮最小实现。

---

## 2.2 第二层：Task 内 Episode/Case 参数 `x_ij`

```text
变化时间尺度：每次 env.reset() 可以变化；
单个 episode 内固定；
同一个 Meta-RL Task 中包含大量不同 x_ij。
```

记为：

```text
x_ij ~ p_i(x)
```

它就是普通单 Task SAC 训练时不断变化的“训练素材”。

当前单 SAC 代码已经清楚体现这一层：

`/sac_scenario_mining/configs/merge_sac.yaml` 当前随机化：

```text
sut_speed_mps
adversary_speed_mps
longitudinal_gap_m
adversary_ramp_position_m
background_density
```

`Stage1AdversarialMergeEnv.reset()` 每个 episode 从 case table 中重新选择 case，因此 **这些变量当前本质上就是 case-level，而不是 Meta-RL task-level。**

本轮 PEARL 侧推荐统一为：

```text
x_ij = {
    sut_initial_speed_mps,
    adversary_initial_speed_mps,
    adversary_spawn_m,
    sut_spawn_m,
    case_seed
}
```

为了降低试验噪声，本轮暂时：

```text
background_density = 0
```

不引入第三辆背景车。

### 一个重要修正

当前单 SAC 实现中，`sut_speed_mps` 同时被用于：

1. SUT 初始速度；
2. `IDMPolicy.target_speed`。

这会使“每个 case 改变初始状态”同时变成“每个 case 改变 IDM 控制器行为”。

本轮必须拆分为：

```text
sut_initial_speed_mps     # case-level，允许每个 episode 变化
sut_idm_target_speed_mps  # fixed protocol，全实验固定
```

PEARL 现有代码已经更接近正确做法：`config["sut"]["target_speed_mps"]` 是固定值；需要在 casebook 中补充 `sut_initial_speed_mps`，并在 reset 时只改变初始速度，不改变 IDM target speed。

---

## 2.3 第三层：Policy 动作 `a_ijt`

```text
变化时间尺度：一个 episode 内每个 simulation step 都变化；
由 SAC / PEARL actor 实际搜索。
```

当前代码保持：

```text
a_t = [steering_residual, throttle_or_brake]
```

`task_env.py` 已通过 route tracking 组合基础转向和策略 residual，使策略主要学习交互时序而不必从零学习路线跟踪，这一设计本轮建议保留。

最终被挖掘出的对抗场景不是一个静态参数向量，而是：

```text
ScenarioTrajectory = {
    task λ_i,
    episode case x_ij,
    state sequence s_0:H,
    adversarial action sequence a_0:H,
    safety events / metrics
}
```

---

# 3. 固定协议 Ω：不是第四层研究变量

以下参数在本轮所有 Task 中固定，不能参与 task identity：

```text
SUT controller          = IDMPolicy
IDM target speed        = fixed
IDM lane-change setting = fixed
reward definition       = fixed
observation schema      = fixed
policy action semantics = fixed
horizon                 = fixed
TTC/critical thresholds = fixed
SAC/PEARL optimizer contract = fixed
```

记为：

```text
Ω = fixed experimental protocol
```

特别是：**本轮不通过改变 SUT 参数制造 Task 差异。**

---

# 4. 当前代码与新方法目标的差距

## 4.1 已经正确实现、建议保留的部分

### A. 单 SAC 的 case-level 环境随机化

现有：

- `sac_scenario_mining/src/casebook.py`
- `sac_scenario_mining/src/env.py`
- `sac_scenario_mining/configs/merge_sac.yaml`

已经实现：

```text
一个固定 SAC 环境 / 逻辑任务
    ↓
每次 reset 从 case table 取不同 x
    ↓
同一 SAC 策略在大量初始条件上训练
```

这正是第二层参数的模板，应复用这个思想，而不是把每个 case 提升成 Meta-RL Task。

### B. PEARL 的 Context / Replay 分离

现有：

- `pearl_learning/src/context_encoder.py`
- `pearl_learning/src/replay.py`
- `pearl_learning/src/pearl_trainer.py`

已经实现原 PEARL 风格的：

```text
recent context buffer -> q(z|C)
long-term replay      -> off-policy SAC actor/critic update
```

并且 context 使用 transition-level Product-of-Gaussians。此处不要推翻重写。

### C. Meta-test 只通过 posterior adaptation 适应

现有 PEARL 代码已经按照：

```text
support trajectories -> context -> posterior update
```

而不是在新 Task 上重新梯度训练网络。此契约继续保留。

---

## 4.2 当前不符合新目标、必须修改的部分

### 问题 1：当前 Meta-RL Task 被“隐藏 reward rule”人为扩增

`pearl_learning/src/taskbook.py` 当前会读取：

```text
meta_train_target_contact_rule_variants
meta_test/evaluation rule variants
```

并生成类似：

```text
geometry_id__rule_adversary_first
geometry_id__rule_sut_first
```

因此同一个物理几何可能因为隐藏的接触规则被拆成多个 Task。

这与本轮目标不一致。本轮 Task 差异应该来自：

```text
物理/逻辑 Merge 配置 λ_i
```

而不是：

```text
同一物理场景 + 不同隐藏奖励标签
```

#### 修改意见

本轮新增独立 pilot config，不直接破坏正式历史配置：

```text
pearl_learning/configs/merge_method_flow_pilot.yaml
```

在 pilot config 中：

```text
- 不定义 meta_train_target_contact_rule_variants
- target_contact_entry_order = any
- target_contact_speed_relation = any
- 所有 Task 使用同一 reward semantics
```

`taskbook.py` 应允许：

```text
rule-variant expansion disabled
```

建议配置字段：

```yaml
task_definition:
  allow_hidden_reward_rule_variants: false
```

当为 `false` 时，同一 geometry 只产生一个 Meta-RL Task。

---

### 问题 2：PEARL 的 casebook 与单 SAC 的 case-level 参数不一致

当前 `pearl_learning/src/casebook.py` 只采样：

```text
adversary_speed_mps
adversary_spawn_m
sut_spawn_m
```

SUT 初速度实际上由固定 `sut.target_speed_mps` 同时决定。

这使 PEARL Task 内的 case diversity 小于当前单 SAC 的训练环境 diversity。

#### 修改意见

新增 case-level 字段：

```text
sut_initial_speed_mps
adversary_initial_speed_mps
adversary_spawn_m
sut_spawn_m
case_seed
```

保留：

```text
sut.target_speed_mps = fixed IDM target speed
```

Adapter 初始化时：

```text
spawn IDM SUT with fixed policy target speed
then set vehicle initial velocity = case.sut_initial_speed_mps
```

而不是：

```text
initial speed == IDM target speed
```

---

### 问题 3：Task-level descriptor 尚未形成统一的学习式接口

当前代码中存在多套“场景描述”：

1. `observation.py` 中的 13 个 topology fields；
2. `moe.py` 中的 `PhysicalTaskDescriptor`；
3. `transferability.py` 中的 task descriptor；
4. `task_representation.py` 中的 geometry supervision target。

这些接口目的不同，目前不能等同于：

```text
e_i = Eω(λ_i)
```

#### 修改意见

新增唯一的静态 Task 编码入口：

```text
pearl_learning/src/scenario_encoder.py
```

同时新增统一 descriptor 构造函数：

```text
build_task_descriptor(task, runtime_geometry) -> np.ndarray
```

本轮 descriptor 只读取第一层 `λ_i`，绝不读取：

```text
case x_ij
query outcome
hidden reward label
policy return
```

建议 pilot descriptor：

```text
[
  logical_type_onehot,
  bottle_lane_num,
  neck_lane_num,
  merge_length_m,
  num_incoming_branches,
  num_outgoing_branches,
  adversary_lane_count,
  sut_lane_count,
  conflict_radius_m,
  adversary_route_curvature,
  sut_route_curvature
]
```

归一化后输入小型 MLP：

```text
input -> 32 -> 16 -> e_dim(8 or 16)
```

本轮 **不使用 GNN/Transformer**。原因不是它们没有潜力，而是当前 pilot 的 Task 数量很小，MLP 足以验证“结构先验是否有用”；过早引入图模型会同时改变表示能力和训练复杂度，使结果无法归因。

---

### 问题 4：PEARL 仍使用统一 `N(0,I)` prior

当前 `context_encoder.py`：

```text
prior() -> μ=0, log_var=0
```

因此不同逻辑 Task 在 K=0 时没有任何结构差异。

#### 修改目标

实现：

```text
e_i = Eω(λ_i)
pη(z|e_i) = Normal(μ0(e_i), σ0²(e_i))
```

新增模块可放在：

```text
pearl_learning/src/scenario_encoder.py
```

其中包含：

```text
ScenarioEncoder
ScenarioConditionedPrior
```

推荐接口：

```python
e = scenario_encoder(task_descriptor)
prior_mu, prior_log_var = scenario_prior(e)
```

---

### 问题 5：Context Encoder 需要把“结构 prior”与“交互 evidence”正确融合

不能简单把同一个 `e_i` 拼到每条 context transition 上，再做 transition Product-of-Gaussians，因为相同静态结构会随着 transition 数量被重复计数。

正确的 pilot 实现应为：

```text
prior factor from e_i
+
independent transition evidence factors from C
```

即按精度相加：

```text
Λ_post = Λ_prior + Σ Λ_n
μ_post = Λ_post^-1 (Λ_prior μ_prior + Σ Λ_n μ_n)
```

修改：

```text
pearl_learning/src/context_encoder.py
```

新增：

```python
product_of_gaussians_with_prior(
    evidence_mu,
    evidence_log_var,
    prior_mu,
    prior_log_var,
)
```

保证：

```text
C = empty  -> posterior == p(z|e)
C != empty -> posterior combines structure prior and interaction evidence
```

同时保留 vanilla PEARL 模式作为 baseline：

```yaml
scenario_prior:
  mode: unit_normal | task_conditioned
```

---

### 问题 6：KL loss 仍然对应 Unit Normal

当前 PEARL 的 KL 正则面向：

```text
KL(q(z|C) || N(0,I))
```

Structure-aware PEARL 应改为：

```text
KL(q(z|C,e) || p(z|e))
```

因此修改：

```text
context_encoder.py
pearl_agent.py
```

新增通用 diagonal-Gaussian KL：

```python
kl_diag_normal(q_mu, q_log_var, p_mu, p_log_var)
```

vanilla baseline 仍以 unit normal 调用同一函数，避免两套 KL 实现。

---

### 问题 7：Actor/Critic 暂时不要同时大改

当前 `observation.py` 已包含静态 topology descriptor，所以 pilot 阶段不建议同时：

```text
删除 topology observation
+ 引入 e
+ 改 actor input
+ 改 critic input
```

否则一次改动太大。

#### 本轮最小实现

```text
Actor/Critic 输入仍维持当前 37D observation + z
```

新编码 `e_i` **只用于 conditional prior**。

这样可以最干净地回答：

```text
“测试前结构信息用于 PEARL task prior，是否能改善 K=0/K=1 adaptation？”
```

后续方法通过后，再做：

```text
Dynamic observation + e + z
```

并清理 observation 中重复的静态 topology fields。

---

### 问题 8：MoE 暂时关闭

当前代码已经有 `PosteriorRoutedMoEActor`，但这不是本轮必须验证的环节。

本轮统一：

```yaml
actor_architecture: dense
posterior_routed_moe:
  enabled: false
```

原因：先验证：

```text
Task 定义 -> case 分布 -> PEARL few-shot adaptation -> structure prior
```

如果这个主链条都未稳定，MoE 的结果没有解释基础。

---

### 问题 9：Transferability 暂时只做离线诊断

当前 `transferability.py` 自己已经明确说明：它目前是 descriptor-space coverage，不是 learned policy-transfer predictor。

本轮不训练新的 relatedness 网络。

只在方法流跑通后，用很小预算生成：

```text
M_ij = performance of quick SAC policy π_i on Task T_j
```

用于判断：

```text
embedding distance 是否和真实 policy transfer 有趋势相关性
```

这一结果只作为下一阶段是否值得加入 learned relatedness / MoE 的依据。

---

# 5. 推荐代码修改清单

## P0：新增 pilot 配置，不破坏现有正式配置

新增：

```text
pearl_learning/configs/merge_method_flow_pilot.yaml
```

要求：

```text
- fixed IDM
- no hidden reward-rule variants
- dense actor
- no MoE
- no disentangled hidden-rule supervision
- no learned transferability
- reduced case counts
- reduced environment steps
```

不要直接覆盖：

```text
dense_pearl_baseline.yaml
posterior_adaptation_protocol.yaml
```

历史配置继续作为原有实验记录。

---

## P1：重构 Task / Case 契约

### 修改 `pearl_learning/src/task_spec.py`

目标：明确：

```text
LogicalScenarioTaskSpec = Meta-RL Task-level λ
```

Pilot 中：

```text
priority_spec 只保留真实可见/固定的交通规则字段；
不允许 hidden target-contact order/speed rule 作为 task identity。
```

建议增加：

```text
task_descriptor_schema
```

写入 checkpoint/taskbook manifest。

### 修改 `pearl_learning/src/taskbook.py`

增加开关：

```text
allow_hidden_reward_rule_variants: false
```

当关闭时：

```text
one physical geometry = one Meta-RL Task
```

### 修改 `pearl_learning/src/casebook.py`

将 case 字段改为：

```text
case_id
case_seed
sut_initial_speed_mps
adversary_initial_speed_mps
adversary_spawn_m
sut_spawn_m
```

case 参数范围统一放入 config：

```yaml
cases:
  parameter_space:
    sut_initial_speed_mps: [10.0, 14.0]
    adversary_initial_speed_mps: [10.0, 17.0]
```

不同 Task 尽量使用相同的归一化 case sampling 规则。

---

## P2：固定 IDM 行为，分离初始速度

修改：

```text
pearl_learning/src/adapters/base.py
sac_scenario_mining/src/metadrive_compat.py  # 如果需要让单 SAC 契约同步
```

目标接口：

```text
IDM target speed = config.sut.target_speed_mps      # 固定
SUT initial speed = case.sut_initial_speed_mps      # 每 episode 变化
```

本轮不让 `x_ij` 改变 IDMPolicy 内部参数。

---

## P3：统一静态 Task Descriptor

新增：

```text
pearl_learning/src/scenario_encoder.py
```

建议包含：

```python
class TaskDescriptorBuilder:
    ...

class ScenarioEncoder(nn.Module):
    ...

class ScenarioConditionedPrior(nn.Module):
    ...
```

配置示例：

```yaml
scenario_representation:
  enabled: true
  encoder_type: mlp
  embedding_dim: 8
  hidden_sizes: [32, 16]

scenario_prior:
  mode: task_conditioned
  hidden_sizes: [32]
```

---

## P4：修改 PEARL posterior

修改：

```text
pearl_learning/src/context_encoder.py
```

新增：

```text
conditional prior + transition evidence PoG
```

必须写单元测试验证：

1. 空 context == conditional prior；
2. context 重排不改变 posterior；
3. 相同 transition 集合改变 episode 分组不改变 transition-product posterior；
4. prior variance/mean 被正确纳入 precision sum；
5. unit prior 模式数值退化为 vanilla baseline。

---

## P5：Agent / Trainer / Evaluator 传递 Task Descriptor

修改：

```text
pearl_learning/src/pearl_agent.py
pearl_learning/src/pearl_trainer.py
pearl_learning/src/evaluator.py
pearl_learning/src/checkpoint.py
```

需要保证：

```text
Task λ -> descriptor -> e -> prior
```

在：

```text
meta-training
meta-validation
meta-test support
meta-test query
```

使用完全一致的 descriptor 构造逻辑。

Checkpoint 必须额外保存：

```text
scenario_encoder state_dict
scenario_prior state_dict
descriptor schema/hash
scenario representation config
```

---

## P6：新增最小泄漏审计

在 `pearl_learning/tests/test_contract.py` 或新测试中检查：

```text
Task descriptor 不读取 case_id
Task descriptor 不读取 case_seed
Task descriptor 不读取 support/query reward
Task descriptor 不读取 hidden target-contact labels
query cases 不进入 context
query cases 不进入 scenario prior
```

---

# 6. Merge 类最小 Meta-RL Task 设计

## 6.1 为什么本轮不做全量拓扑

当前代码已有：

```text
on_ramp_merge
lane_drop_merge
bottleneck_merge
y_merge
```

但第一轮同时做全部拓扑会引入：

```text
不同 map adapter
不同 route topology
不同 geometry descriptor
不同 OOD regime
```

这会显著增加 debugging 成本。

因此本轮只使用现有 `bottleneck` map adapter 下已经稳定实现的：

```text
lane_drop_merge
bottleneck_merge
```

它们共享较多底层实现，但 lane connectivity 不同，适合验证 task-level 变化。

---

## 6.2 Pilot Task 划分

### Meta-train：4 Tasks

```text
T1 = lane_drop_merge, merge_length=24 m, 3 -> 2 lanes
T2 = lane_drop_merge, merge_length=32 m, 3 -> 2 lanes
T3 = bottleneck_merge, merge_length=24 m, 3 -> 1 lane
T4 = bottleneck_merge, merge_length=32 m, 3 -> 1 lane
```

### Meta-validation：2 Tasks

```text
V1 = lane_drop_merge, merge_length=40 m
V2 = bottleneck_merge, merge_length=40 m
```

### Meta-test：2 Tasks

```text
Q1 = lane_drop_merge, merge_length=48 m
Q2 = bottleneck_merge, merge_length=48 m
```

这组划分直接复用当前 geometry catalog 已有的物理场景，几乎不增加地图开发工作。

### 暂缓

```text
on_ramp_srs
all y_merge tasks
```

`y_merge` 留作后续真正的 unseen logical type/OOD 检验；当前跑通方法时不需要。

---

# 7. Task 内 case 参数与数量

## 7.1 Pilot case 参数

所有 Task 使用同一 case 参数分布原则：

```text
sut_initial_speed_mps         ~ U(10, 14)
adversary_initial_speed_mps   ~ U(10, 17)
adversary_spawn_m             ~ task-valid spawn region
sut_spawn_m                   ~ task-valid spawn region
background_density            = 0
```

应额外计算并记录派生量：

```text
initial_arrival_time_gap_s
initial_relative_speed_mps
initial_distance_m
```

这些派生量先只用于 audit，不作为新的自由变量。

### 可选的可行性约束

拒绝明显没有交互可能的 reset：

```text
abs(initial_arrival_time_gap_s) <= 3~4 s
initial_distance > safety reset threshold
both vehicles before conflict zone
```

目的只是避免把大量训练预算浪费在“双方根本碰不到”的 case 上。

---

## 7.2 Pilot case 数量

建议新 pilot config：

```yaml
cases:
  per_task:
    train_pool: 12
    validation_support: 4
    validation_query: 6
    test_support: 4
    test_query: 8
```

这不是论文最终规模，只用于：

```text
确认 reset / support / query / posterior / checkpoint / evaluation 全链路可运行。
```

---

# 8. 最小训练预算

## 8.1 Stage A：Engineering Smoke Test

目的：只检查代码，不解释性能。

建议：

```text
meta_train tasks       = 4
train_pool/task        = 4 cases
meta_batch_size        = 2 or 4
bootstrap episodes     = 1~2/task
context shots          = [0, 1, 2]
total env steps        = 20k~30k
training seed          = 1
query/task             = 2~4
```

通过条件：

```text
- taskbook/casebook 构建成功
- 所有 Task reset 成功
- context posterior 可计算
- conditional prior 可计算
- checkpoint save/load 成功
- K=0/1/2 few-shot evaluation 可以完整输出
- support/query 没有 leakage
- 无 NaN
```

这一步不看“谁性能最好”。

---

## 8.2 Stage B：Method-Flow Pilot

目的：获得初步真实实验证据，但仍不是全量论文结果。

建议：

```text
meta_train tasks       = 4
meta_validation tasks  = 2
meta_test tasks        = 2
train_pool/task        = 12 cases
support/task           = 4 cases
query/test task        = 8 cases
shots                  = [0, 1, 2, 4]
training seed          = 1
total environment steps= 150k~250k
validation interval    = 25k
meta_batch_size        = 4
```

若 150k 后 validation 已完全不再改善，则提前停止，不需要强行跑到原配置的 1.5M steps。

---

# 9. 本轮只比较 3 个方法

## Baseline 1：Pooled SAC / No-context policy

目的：回答：

```text
“不做 task inference，一个共享策略能做到什么程度？”
```

无需为每个 Task 做完整 300k SAC。

---

## Baseline 2：Vanilla Dense PEARL

```text
p(z) = N(0,I)
q(z|C)
Dense actor
```

使用现有 transition-product context encoder。

---

## Method：Structure-Aware Dense PEARL

```text
e_i = Eω(λ_i)
p(z|e_i)
q(z|C,e_i)
Dense actor
```

除 scenario encoder/prior 外，其余训练预算、Task、case、query 完全与 vanilla PEARL 一致。

---

# 10. 本轮不要比较的内容

以下内容全部推迟，避免实验量和变量同时爆炸：

```text
× Posterior-routed MoE
× 4/8 experts
× GNN scene encoder
× Transformer scene encoder
× learned transferability predictor
× active support selection
× multiple SUT controllers
× IDM parameterized task family
× y-merge unseen-type OOD full evaluation
× 3~5 training seeds
× 20~60 query cases/task
× 1.5M+ environment steps/method
× constrained RL / RSS / diversity objective
```

本轮只能回答“方法链是否成立”，不能用 pilot 结果宣称最终 SOTA 或统计显著优越。

---

# 11. Pilot 评估指标

每个 meta-test Task、每个 K 都记录：

## 场景搜索效果

```text
valid critical scenario rate (VCSR)
target collision rate
critical rate
invalid rate
median / mean min TTC
average return
```

## Few-shot adaptation

定义：

```text
AdaptationGain(K) = Metric(K) - Metric(K=0)
```

重点观察：

```text
K=1
K=2
K=4
```

不要只报告 K=4 的最终值。

## Posterior 行为

记录：

```text
posterior mean μ_z
posterior log variance
average posterior variance
KL(q || prior)
```

检查：

```text
support 增加后 uncertainty 是否总体收缩；
不同 Task posterior 是否出现可区分趋势。
```

## Structure-aware prior

特别比较 K=0/K=1：

```text
Vanilla PEARL vs Structure-Aware PEARL
```

因为场景结构先验最应该在少量/零 context 时产生价值。

---

# 12. Pilot 的“通过标准”

本轮不设置过高性能门槛，但至少应满足以下 Gate。

## Gate 0：Task / Case 层次正确

必须审计：

```text
同一 T_i 内 λ_i 恒定；
不同 episode 的 x_ij 确实变化；
IDM policy 参数在所有 Task/case 中一致；
只有 adversary action a_t 由 policy 连续输出。
```

## Gate 1：Case diversity 有效

同一 Task 的 12 个 train cases 至少在：

```text
初始相对速度
arrival time gap
spawn location
```

上有明显覆盖，而不是重复近似相同 reset。

## Gate 2：Vanilla PEARL 出现适应趋势

至少在大多数 test Task 上观察到：

```text
K=2 或 K=4 比 K=0 更好
```

指标优先使用 VCSR / critical rate / return。

若没有该趋势，先检查 Task 是否可区分、reward 是否有效、context 是否含足够信息，不要进入 MoE。

## Gate 3：Structure-aware prior 没有失效

至少满足：

```text
K=0/K=1 与 vanilla 相比不出现系统性退化；
并在至少一个主要指标上显示合理正向趋势。
```

Pilot 不要求统计显著性。

## Gate 4：数据隔离正确

必须证明：

```text
support cases 与 query cases 不重叠；
query transition 不进入 context/replay 的 adaptation 条件；
test Task 不参与模型选择；
scenario descriptor 只由 λ_i 构造。
```

---

# 13. 低成本的 Task Transfer 诊断（Stage B 通过后再做）

只有 Stage B 通过后，才建议花少量成本验证“场景相关性”研究方向。

对 4 个 meta-train Task：

```text
T1, T2, T3, T4
```

各自训练一个 **quick single-task SAC**，不需要 300k：

```text
30k~50k steps/task
1 seed
```

然后构造 4×4 transfer matrix：

```text
M_ij = π_i 在 T_j 的少量固定 query cases 上的表现
```

每个 T_j 只用 4~6 query cases 即可。

同时计算：

```text
d_ij = ||e_i - e_j||
```

观察：

```text
-d_ij 与 M_ij / transfer gain 是否存在排序趋势。
```

这一阶段只做：

```text
Spearman correlation
nearest-neighbor ranking
简单 scatter plot
```

不训练新的 relatedness neural network。

只有当这一现象存在时，下一阶段才值得进入：

```text
learned directed task relatedness
MoE expert routing
```

---

# 14. 场景编码器后续升级路线

## 当前：MLP

当前 Task descriptor 是少量明确结构字段，Task 只有个位数，因此：

```text
MLP 是正确的最小实现。
```

## 何时升级为 GNN

当下一阶段 Task 变成真正 variable-topology：

```text
on-ramp
Y-merge
lane-drop
bottleneck
multi-lane merge
```

并希望显式编码：

```text
lane nodes
successor/predecessor
merge/conflict edges
route-role relations
```

再将 `ScenarioEncoder` 替换为 graph encoder。

## 何时考虑 Transformer

只有在：

```text
图规模明显变化
多 agent / 多 lane / 多冲突区
需要联合 static topology + trajectory tokens
拥有足够预训练场景数据
```

时再考虑 graph transformer / scene transformer。

因此代码接口应该从现在就抽象：

```python
ScenarioEncoder(descriptor) -> e
```

但 pilot 具体实现只使用 `MLPScenarioEncoder`。

---

# 15. 推荐新增/修改文件总表

| 文件 | 动作 | 本轮目标 |
|---|---|---|
| `pearl_learning/configs/merge_method_flow_pilot.yaml` | 新增 | 唯一 pilot 配置 |
| `pearl_learning/src/task_spec.py` | 修改 | 明确 Task-level λ；禁用 hidden reward rule identity |
| `pearl_learning/src/taskbook.py` | 修改 | 支持关闭 rule variant expansion |
| `pearl_learning/src/casebook.py` | 修改 | 明确 case-level x；加入 SUT initial speed |
| `pearl_learning/src/adapters/base.py` | 修改 | SUT initial speed 与 IDM target speed 分离 |
| `pearl_learning/src/scenario_encoder.py` | **新增** | Task descriptor、MLP encoder、conditional prior |
| `pearl_learning/src/context_encoder.py` | 修改 | conditional-prior PoG + general Gaussian KL |
| `pearl_learning/src/pearl_agent.py` | 修改 | prior/posterior 接收 task embedding |
| `pearl_learning/src/pearl_trainer.py` | 修改 | 每个 meta-batch 传递 task descriptor |
| `pearl_learning/src/evaluator.py` | 修改 | support/query 共享固定 descriptor；输出 prior/posterior diagnostics |
| `pearl_learning/src/checkpoint.py` | 修改 | 保存 encoder/prior/schema/hash |
| `pearl_learning/tests/test_contract.py` | 修改 | 新三层参数契约与 leakage tests |
| `pearl_learning/tests/test_scenario_prior.py` | **新增** | conditional prior/PoG 单元测试 |
| `pearl_learning/scripts/run_method_flow_pilot.py` | 可新增 | 一次运行 smoke / pilot 三方法的小预算入口 |

### 暂时不修改核心逻辑

```text
moe.py
transferability_calibration.py
transferability_decision.py
constrained scenario mining modules
```

---

# 16. 推荐 pilot 配置草案

下面参数只作为跑通方法的起始点：

```yaml
project:
  output_root: results/pearl_learning/merge_method_flow_pilot

environment:
  horizon: 180
  observation_schema: logical_merge_obs
  observation_dim: 37
  action_dim: 2

sut:
  controller: IDMPolicy
  target_speed_mps: 12.0
  enable_lane_change: false

case_sampling:
  background_density: 0.0
  sut_initial_speed_mps: [10.0, 14.0]
  adversary_initial_speed_mps: [10.0, 17.0]
  max_initial_arrival_gap_s: 4.0

cases:
  per_task:
    train_pool: 12
    validation_support: 4
    validation_query: 6
    test_support: 4
    test_query: 8

task_definition:
  allow_hidden_reward_rule_variants: false

scenario_representation:
  enabled: true
  encoder_type: mlp
  embedding_dim: 8
  hidden_sizes: [32, 16]

scenario_prior:
  mode: task_conditioned
  hidden_sizes: [32]

pearl:
  latent_dim: 5
  context_aggregation: transition_product
  context_transitions_per_episode: 32
  context_min_episodes: 1
  kl_beta: 0.1

meta_training:
  meta_batch_size: 4
  replay_capacity_transitions: 50000
  recent_context_episodes_per_task: 8
  bootstrap_episodes_per_task: 2
  gradient_updates_per_iteration: 20
  total_environment_steps: 200000
  validation_interval_steps: 25000

evaluation:
  shots: [0, 1, 2, 4]
  query_cases_per_task: 8

experiment:
  training_seeds: [0]
  device: auto

posterior_routed_moe:
  enabled: false
```

实际运行时可先把 `total_environment_steps` 改成 `20000` 做 smoke；只有 smoke 完全通过后再恢复到约 `150k~250k`。

---

# 17. 推荐开发顺序

严格按下面顺序改，不要并行增加多个方法模块。

```text
Step 1
冻结三层参数契约 λ / x / a
        ↓
Step 2
去掉 pilot 中 hidden reward-rule Task
        ↓
Step 3
PEARL casebook 与普通 SAC case-level 语义对齐
        ↓
Step 4
固定 IDM 参数，分离 SUT initial speed
        ↓
Step 5
运行 vanilla PEARL smoke
        ↓
Step 6
新增 Task descriptor + MLP Scenario Encoder
        ↓
Step 7
实现 conditional prior p(z|e)
        ↓
Step 8
实现 q(z|C,e) 的 prior+evidence Product-of-Gaussians
        ↓
Step 9
三方法小预算 pilot
        ↓
Step 10
只有方法流通过后，再做 quick SAC transfer matrix
        ↓
Step 11
根据 transfer evidence 决定是否继续 MoE / GNN / learned relatedness
```

---

# 18. 本轮最终应产出的结果文件

建议 pilot 输出至少包括：

```text
results/pearl_learning/merge_method_flow_pilot/
  taskbook.json / taskbooks/
  casebooks/
  descriptor_schema.json
  run_manifest.json
  vanilla_pearl/
  structure_aware_pearl/
  pooled_sac/
  fewshot_metrics.jsonl
  posterior_diagnostics.jsonl
  leakage_audit.json
  pilot_summary.json
```

`pilot_summary.json` 至少记录：

```text
method
training_seed
train_steps
meta_train_tasks
meta_validation_tasks
meta_test_tasks
K
VCSR
critical_rate
target_collision_rate
invalid_rate
median_min_ttc
return
posterior_variance
checkpoint
```

---

# 19. 本轮可以得到什么结论，不能得到什么结论

## 如果 pilot 成功，可以说

```text
1. 已建立 Meta-RL Task / SAC episode case / adversarial action 三层参数契约；
2. 一个固定 Merge Task 内可像普通 SAC 一样通过多 case 训练；
3. PEARL 能够通过少量 support cases 的交互 context 对未见 Task 进行 posterior adaptation；
4. 静态逻辑场景结构可以作为 task-conditioned prior 接入 PEARL；
5. 完整训练、support、query、posterior、evaluation 方法流已经跑通。
```

## 此时还不能说

```text
× 方法已经达到 SOTA
× GNN 比 MLP 更好
× Transformer 必要
× MoE 必然有益
× learned relatedness 有效
× 对真实 ADS 泛化
× 对不同 SUT controller 泛化
× 已完成完整 OOD logical-scenario 泛化
```

---

# 20. 最核心的实施原则

本轮所有代码修改都应围绕一个判断标准：

```text
Task 参数 λ：一个单 Task SAC 训练全过程保持不变；
Case 参数 x：同一个 SAC Task 内每次 episode/reset 都可以变化；
Action a_t：同一个 episode 内由 SAC/PEARL 每个 step 搜索。
```

因此本轮真正要跑通的问题是：

> **在固定 Merge 功能场景族和固定 IDM SUT 下，Meta-RL 从多个由高层逻辑/几何配置 `λ_i` 定义的训练 Task 中学习；每个 Task 内仍通过大量 case `x_ij` 提供普通 SAC 所需的环境多样性；面对新的 `λ_*` 时，仅使用少量 support cases 的交互轨迹建立 posterior，就能够快速调整对抗车辆时间序列动作策略，并在未参与适应的 query cases 上继续挖掘有效安全关键场景。**

这就是当前阶段最应该优先跑通的完整方法论闭环。
