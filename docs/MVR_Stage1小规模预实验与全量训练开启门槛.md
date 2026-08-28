# MVR Stage1 小规模预实验与全量训练开启门槛
## 基于当前 2D vehicle-residual SAC 的 Pre-Stage1 Verification Plan

> 仓库：`SafeDL/META_LEARNING`  
> 审计基线：`f0238b435e2dbe7fcf7052de26519f9c4f9d0bb6`  
> 当前 Inner 状态：11D interaction-centric physical state  
> 当前 Inner 动作：`[Δsteering, Δacceleration]`  
> 当前控制契约：`vehicle_residual_2d`  
> 当前目标：**先用少量、可诊断的预实验确认 Stage1 代码与训练问题定义正确，再启动 36-task Formal Stage1。**

---

# 1. 结论

当前**应该先做小规模预实验，不应该立即启动 Formal Stage1 全量训练**。

原因不是 SAC 架构本身已经被证明错误，而是最新 S0 无训练筛查已经暴露出三类 Functional Scenario 的训练分布不均衡：

```text
Cut-in:
    当前 2D residual 可产生合法 critical behavior

Merge:
    当前固定 x0 下 0/20 critical
    braking 能降低 TTC，但还没有进入 critical interaction

Roundabout:
    当前固定 x0 下 0/20 critical
    两车仍远离真实 interaction region
```

因此当前最大风险是：

> **在 Merge / Roundabout 还没有形成稳定 challenge-state visitation 的情况下，直接把 36 个 task 混在一起训练 SAC。**

此时即使 SAC loss 正常下降，学习到的策略也可能只是：

- 被 Cut-in 的少量 event signal 主导；
- 对 Merge / Roundabout 学不到任何东西；
- 或仅学会 nominal action 附近的无效平均策略。

因此推荐采用：

```text
P0 Engineering Contract
        ↓
P1 Scenario Semantic / Base Contract
        ↓
P2 x0 + Action Reachability Calibration
        ↓
P3 Reward / Credit Assignment Audit
        ↓
P4 SAC Plumbing Smoke
        ↓
P5 Mini Shared-Learning Pilot
        ↓
P6 36-task Coverage Smoke
        ↓
PRE_STAGE1_PASS
        ↓
Formal Stage1
```

---

# 2. 当前代码设计的基线

当前方法链：

```text
Outer:
candidate + x0 + maneuver_onset_progress + scenario profile
        ↓
Functional / Logical Scenario Contract
        ↓
Native IDM nominal behavior
        ↓
Inner SAC
[Δsteering, Δacceleration]
        ↓
TrafficActionShield
        ↓
ScenarioSemanticMonitor
        ↓
valid event / failure
```

Inner SAC 不负责决定：

```text
“是不是 Cut-in”
“从哪条路 Merge”
“Roundabout 从哪个入口到哪个出口”
```

这些均由 Functional / Logical Scenario 与 Outer 决定。

Inner SAC 只回答：

> **在合法机动已经被定义的前提下，如何通过闭环车辆修正增加安全挑战。**

因此预实验的目的也不是重新证明单任务 SAC 能否制造危险。

`archives/sac_scenario_mining/` 已经作为冻结 positive control 证明：

```text
固定 Merge task 上单任务 SAC 可以学到强 adversarial policy
```

现在需要验证的是：

> **当前共享 2D residual MDP 是否被定义成了一个可学习的 multitask RL 问题。**

---

# 3. 预实验总原则

所有预实验遵守以下原则。

## 3.1 不使用 final test set 调参

开发期只使用：

```text
train:
    g01 / g02 / g03
    cautious / defensive / normal / assertive

validation:
    g04
    fast_small_gap
```

禁止使用：

```text
g05
late_response
```

决定：

```text
x0 band
reward
action scale
训练轮数
checkpoint selection
```

---

## 3.2 所有对比必须 paired

同一比较中必须固定：

```text
geometry
candidate
x0
maneuver onset
scenario profile
SUT
episode seed
```

只改变被测试因素，例如：

```text
residual action
reward version
trained vs random policy
```

---

## 3.3 “危险”与“有效”分开

训练/诊断必须同时记录：

```text
SemanticValid
TrafficValid
Criticality
```

不能用：

```text
collision rate
```

单独判断成功。

---

# 4. P0 — Engineering Contract Test
## 目的：验证代码接口和 schema 没有低级错误

这一阶段不做训练。

执行：

```powershell
conda activate metadrive

python -m pytest mvr/tests -q

python -m compileall -q mvr
```

---

## P0 必查项

### 状态

```text
PhysicalStateExtractor.dimension == 11
```

11 个字段必须全部 finite，归一化后在：

```text
[-1, 1]
```

### Inner action

```text
action_dim == 2
```

必须保证：

```text
raw_action.shape == (2,)
replay.action == raw SAC action
```

禁止再次出现：

```text
actor 输出一种 action
replay 保存另一种 filtered action
```

### 控制链

Zero residual：

```text
[0, 0]
```

必须满足：

```text
candidate_action == native base action
```

Shield 不得替代 nominal driving。

### checkpoint

必须记录：

```text
git commit
config hash
taskbook hash
control schema
source tree provenance
```

## P0 PASS

```text
pytest: 100% PASS
compileall: 0 error
NaN/Inf: 0
2D action contract: PASS
checkpoint schema: PASS
```

失败：

```text
STOP
```

---

# 5. P1 — Scenario Semantic / Base Contract
## 目的：先证明“没有 SAC 时，三个 Logical Scenario 本身就是正确的”

不训练。

每个 family 选择：

```text
2 个代表性 train case
```

建议：

```text
g01-normal
g03-assertive
```

总计：

```text
3 families × 2 cases = 6 episodes
```

全部使用：

```text
residual = [0, 0]
```

## P1.1 Cut-in

必须人工/日志确认：

```text
source lane 正确
target lane 正确
onset 位于 legal merge window
车辆真实发生 footprint intrusion
最终进入 target lane
没有 out-of-road
没有 wrong-route
```

## P1.2 Merge

必须确认：

```text
red adversary = branch/ramp
blue SUT = mainline
两车保持规定 route
shared conflict region 正确
```

## P1.3 Roundabout

必须确认：

```text
entry / exit route 正确
车辆不逆行
不穿越环岛内部
shared conflict reference 正确
```

## P1 PASS

```text
6/6 episode 执行语义正确
traffic violation = 0
wrong route = 0
```

注意：

P1 不要求：

```text
产生 collision / near-miss
```

只要求：

> Functional / Logical Scenario 本身正确。

---

# 6. P2 — x0 + Action Reachability Calibration
## 当前最重要的预实验

当前 S0 已经证明：

```text
Cut-in: reachable
Merge: 当前 x0 不够
Roundabout: 当前 x0 不够
```

所以 P2 不是重新跑相同 S0，而是：

> **为每个 family 找到一个“Base 不饱和、residual 又有能力明显推进风险”的 interaction band。**

---

# 7. P2.1 目标 interaction band

每个 family 的 calibration case 应尽量满足：

```text
Base:
    大多数情况下合法
    不应天然 100% critical

Residual:
    至少某些合法 residual 能明显降低 TTC / distance
    最好能产生少量 valid near-miss / collision
```

推荐经验目标：

```text
Base valid critical rate:
    0% ~ 25%

scripted residual valid critical rate:
    > Base

challenge-phase reachability:
    >= 30%
```

这不是论文正式阈值，只是训练可学习性阈值。

---

# 8. P2.2 Merge 定向校准

当前 S0：

```text
0/20 critical
```

因此只调：

```text
adversary_distance_to_conflict
sut_distance_to_conflict
initial speeds
```

不要改 SAC。

建议构造 4 组 paired ETA：

```text
ΔETA ≈ -1.0 s
ΔETA ≈ -0.5 s
ΔETA ≈ +0.5 s
ΔETA ≈ +1.0 s
```

每组测试：

```text
base              [ 0.00,  0.00]
acceleration_brake[ 0.00, -0.75]
acceleration_push [ 0.00, +0.75]
```

预算：

```text
4 x0 × 3 actions = 12 episodes
```

若仍完全无 interaction：

再扩大到：

```text
6 x0 × 3 actions = 18 episodes
```

---

# 9. P2.3 Roundabout 定向校准

当前 S0：

```text
median distance ≈ 50 m
0/20 critical
```

因此优先调：

```text
双方 distance-to-conflict
双方初速度
candidate entry/exit pairing
```

目标不是直接制造 collision，而是先让：

```text
challenge_phase_active
```

真实出现。

建议：

```text
4 candidate/x0 combinations
×
3 residuals
=
12 episodes
```

若：

```text
challenge steps / total steps < 10%
```

说明 x0 仍未校准好。

---

# 10. P2.4 Cut-in 只做稳定性复核

Cut-in 当前已经可达。

只需：

```text
2 x0
×
base / brake / acceleration
=
6 episodes
```

确认：

```text
footprint intrusion 正确
valid critical event 仍可复现
residual 不破坏合法 Cut-in
```

---

# 11. P2.5 residual sensitivity

当前 2D action：

```text
[Δsteering, Δacceleration]
```

必须确认两个维度不是完全 dead。

至少记录：

```text
ΔTTC
Δdistance
Δchallenge duration
Δevent probability
Shield intervention
```

推荐 action set：

```text
base:
[0.00, 0.00]

steer+:
[+0.75, 0.00]

steer-:
[-0.75, 0.00]

brake:
[0.00, -0.75]

push:
[0.00, +0.75]
```

如果发现：

```text
Δsteering
```

在三个 family 中几乎没有任何可测效果，则正式 Stage1 前应决定：

```text
提高 steering scale
或
暂时移除该维度
```

不要让 SAC 学一个环境不响应的动作维。

---

# 12. P2 PASS

每个 family 必须至少满足：

```text
legal Base rollout exists
challenge phase reachable
scripted residual changes risk measurably
traffic validity not destroyed
```

推荐最低：

```text
Cut-in:
    PASS

Merge:
    >= 1 个 calibrated x0 能被 residual 推向明显更高风险

Roundabout:
    >= 1 个 calibrated x0 进入真实 challenge phase
    residual 能改变冲突程度
```

如果 Roundabout 仍完全不可达：

```text
STOP
```

继续调 Logical Scenario / x0。

不要启动 SAC 全量训练。

---

# 13. P3 — Reward / Credit Assignment Audit
## 目的：验证 SAC 收到的 reward 与我们真正想优化的东西一致

这个实验主要是 offline / unit test。

几乎不消耗 simulator budget。

---

# 14. P3.1 reward curve

固定合法 event/无 violation，人工扫描：

```text
TTC:
0.5, 1, 2, 3, 5, 8, 12 s

distance:
1, 2, 5, 10, 20, 40 m
```

绘制：

```text
reward vs TTC
reward vs distance
```

必须明确回答：

> 当前 reward 是在奖励“越危险越高”，还是“靠近 threshold 最高”？

当前代码使用 threshold-centered Gaussian shaping。

如果论文目标是：

```text
maximize safety-critical pressure
```

则建议正式 Stage1 前改为单调 risk，例如：

```text
exp(-TTC/tau)
exp(-distance/d0)
```

如果保留当前 Gaussian，则必须在实验计划中明确：

> 它优化的是 threshold proximity，而不是 severity monotonicity。

不能无意识地保留。

---

# 15. P3.2 semantic gating

至少构造以下 4 个条件：

```text
A:
很低 TTC
但 challenge_phase = False

B:
中等 TTC
challenge_phase = True

C:
valid near-miss event just captured

D:
adversary traffic violation
```

预期：

```text
A:
不能因为低 TTC 获得“成功事件”奖励

B:
可获得风险 shaping

C:
获得一次 event bonus

D:
总 reward 必须明显更差
```

---

# 16. P3.3 one-shot event bonus

同一个 valid near-miss：

```text
event capture step
后续 5 steps
```

必须满足：

```text
event bonus 只出现 1 次
```

---

# 17. P3.4 event_action_weight 消融

当前正式配置中仍有：

```yaml
event_action_weight: 2.0
```

建议 Pre-Stage1 先改为：

```yaml
event_action_weight: 0.0
```

理由：

critical event 的形成通常来自此前多步 action，不一定来自最后 event timestep。

预实验优先验证：

```text
标准 SAC Bellman backup
```

本身能否工作。

等 Formal Stage1 学通后，再把 event imitation 作为 ablation。

---

# 18. P3 PASS

```text
reward finite
invalid penalty direction correct
event bonus exactly once
semantic-invalid proximity 不会被标记为成功
reward curve 与预期优化目标一致
```

失败：

```text
STOP
```

---

# 19. P4 — SAC Plumbing Smoke
## 目的：确认“优化器真的在学习正确的 2D action MDP”

这一阶段只需要非常少的 simulator episode。

选择：

```text
Cut-in: g01-normal
Merge: calibrated g01-normal
Roundabout: calibrated g01-normal
```

每个 task：

```text
2 episodes
```

总训练：

```text
6 episodes
```

---

# 20. P4 配置

建议新建：

```text
mvr/configs/mvr_stage1_preflight.yaml
```

核心：

```yaml
training:
  family_filter: all
  step_budget: 240

inner:
  action_dim: 2
  episodes_per_task: 2
  updates_per_episode: 2
  batch_size: 32
  event_sample_fraction: 0.25
  event_action_weight: 0.0
```

注意：

这不是性能训练。

---

# 21. P4 必查项

### Replay

每个 transition：

```text
state: 11D
action: 2D raw SAC residual
next_state: 11D
reward: finite
done: bool
```

### 参数更新

记录训练前后：

```text
actor parameter L2 change
critic parameter L2 change
```

要求：

```text
> 0
```

### Gradient

```text
NaN = 0
Inf = 0
gradient norm finite
```

### Action

记录：

```text
mean
std
saturation rate
```

不能一开始就：

```text
>95% action 在 ±0.75 附近
```

### Training signal

新增加的：

```text
metrics.training_signal
```

必须正常生成：

```text
overall
family:merge
family:cutin
family:roundabout
option:...
```

---

# 22. P4 PASS

```text
all updates finite
actor changed
critic changed
replay action contract correct
training_signal generated
no immediate policy saturation
```

---

# 23. P5 — Mini Shared-Learning Pilot
## 这是全量训练前最关键的小训练

目的不是论文性能，而是回答：

> **当前 2D shared SAC 在三个已经校准到可达区域的 Functional Scenario 上，是否出现最基本的学习方向？**

---

# 24. P5 训练 task

不要立刻使用全部 36 task。

推荐：

```text
Merge:
    g01-normal
    g01-assertive

Cut-in:
    g01-normal
    g01-assertive

Roundabout:
    g01-normal
    g01-assertive
```

共：

```text
6 tasks
```

---

# 25. P5 训练预算

每个 task：

```text
3 episodes
```

总训练：

```text
18 simulator episodes
```

更新：

```text
updates_per_episode = 4
batch_size = 32
event_action_weight = 0
```

这是一个很小的 multi-task shared-SAC pilot。

---

# 26. P5 Validation

每个 family：

```text
2 个固定 validation x0
```

比较：

```text
Base
Random Residual
Trained Inner
```

总 validation：

```text
3 families
× 2 cases
× 3 policies
=
18 episodes
```

训练 + 验证：

```text
36 episodes
```

---

# 27. P5 不要求什么

不要求：

```text
显著性检验
multi-seed
R4
trained failure rate 大幅超过 baseline
```

---

# 28. P5 必须看到什么

至少要看到以下之一：

### 连续风险方向

```text
Trained Inner
相对 Base / Random
在 >= 2/3 families 中
降低 median TTC 或 distance
```

同时：

```text
invalid rate 不增加
```

### 或有效事件方向

```text
Trained valid-event count
>
Base / Random
```

### Training signal

三个 family：

```text
valid_event_episodes > 0
或
positive_reward_transition_fraction > 0
```

若某个 family：

```text
positive signal == 0
```

说明仍不适合进入全量训练。

---

# 29. P5 FAIL 的解释

### Cut-in 学到，Merge/Roundabout 不学

优先检查：

```text
x0 / challenge reachability
```

不是先调 SAC 网络。

### 三类都有 positive event，但 SAC 完全不改变风险

检查：

```text
reward
critic target
action scaling
learning rate
replay sampling
```

### SAC 只会输出极端 action

检查：

```text
entropy alpha
action scale
reward magnitude
shield projection
```

---

# 30. P6 — 36-task Coverage Smoke
## 目的：在 full training 前确认全任务工程覆盖

只有 P5 PASS 后执行。

使用：

```text
36 train tasks
1 episode / task
1~2 update / episode
```

总：

```text
36 episodes
```

这次不看性能。

只看：

```text
36/36 task visited
3/3 family
9/9 train geometry
4/4 train SUT
all candidates
all scenario profiles
all checkpoints reloadable
training_signal 完整
```

---

# 31. P6 额外必须审计：maneuver_onset_progress nuisance

当前统一 continuous config 是 5D：

```text
[d_adv,
 d_sut,
 v_adv,
 v_sut,
 maneuver_onset_progress]
```

但：

```text
maneuver_onset_progress
```

只对 Cut-in 生效。

因此 Merge / Roundabout 必须做一个 invariance test：

固定：

```text
candidate
四个有效 x0
residual
seed
```

只改变：

```text
maneuver_onset_progress
0.2 → 0.5 → 0.8
```

预期：

```text
Merge trajectory identical
Roundabout trajectory identical
```

如果环境确实不响应这一维，则 Formal Stage1 前建议：

```text
Merge/Roundabout 固定该维为 0
或
加入 continuous mask
```

不要给共享 SAC 一个随机但无因果作用的 nuisance feature。

---

# 32. P6 额外审计：scenario profile

当前仍有：

```text
approach_conflict
yield_then_press
gap_close
```

这些 profile 会影响 nominal IDM intent。

因此做一个极小消融：

同一 x0、zero residual：

```text
3 options
```

比较：

```text
min TTC
min distance
challenge duration
valid event
```

如果 profile 本身就主导绝大部分危险性：

> Stage1 可能学的是“选到人工 profile”，而不是 residual policy。

此时建议 Formal Stage1：

```text
先固定一个 neutral profile
```

或者将 profile 明确作为 Outer baseline/ablation。

---

# 33. PRE_STAGE1_PASS 的硬门槛

只有以下全部通过，才允许启动 Formal Stage1。

| Gate | 条件 |
|---|---|
| P0 Engineering | pytest/compile/schema 全通过 |
| P1 Scenario | 3 families Base 语义正确 |
| P2 Reachability | 3 families 均进入可学习 interaction band |
| P2 Residual | scripted residual 能改变 risk |
| P3 Reward | reward 与预期目标方向一致 |
| P3 Event | event bonus one-shot |
| P4 SAC | actor/critic 更新 finite 且参数变化 |
| P4 Replay | raw 2D action contract 正确 |
| P5 Learning | 至少出现基础 shared-learning signal |
| P5 Validity | learning 不靠 traffic violation |
| P6 Coverage | 36/36 tasks 可执行 |
| Nuisance | 非 Cut-in onset 不污染学习 |
| Profile | option effect 被理解并控制 |

全部满足：

```text
PRE_STAGE1_PASS
```

---

# 34. 预实验总预算

不计算已有 S0 的情况下，新增预算可控制在约：

### P1

```text
6 episodes
```

### P2 targeted calibration

约：

```text
24 ~ 42 episodes
```

### P3

```text
0 simulator episode
```

### P4

```text
6 episodes
```

### P5

```text
18 train + 18 validation = 36 episodes
```

### P6

```text
36 episodes
```

总新增：

```text
约 108 ~ 126 simulator episodes
```

而且每一步都可以提前停止。

这比直接启动 Formal Stage1 后再诊断失败更高效。

---

# 35. Formal Stage1 开启后的配置

只有：

```text
PRE_STAGE1_PASS
```

之后才恢复正式：

```yaml
inner:
  episodes_per_task: 5
  updates_per_episode: 8
  batch_size: 64
```

总：

```text
36 tasks × 5 episodes
=
180 simulator episodes
```

之后再做正式 validation。

---

# 36. Formal Stage1 的比较对象

保持：

### Base

```text
Native IDM
+
zero residual
```

### Random

```text
Native IDM
+
random 2D residual
```

### Method

```text
Native IDM
+
trained shared SAC residual
```

---

# 37. Formal Stage1 验收重点

Formal Stage1 才正式要求：

```text
Trained > Base
Trained > Random
```

同时：

```text
SemanticValid
TrafficValid
```

不能下降。

重点验证：

```text
seen SUT + seen geometry
unseen validation SUT + seen geometry
seen SUT + validation geometry
validation SUT + validation geometry
```

---

# 38. 建议的执行决策树

```text
P0 code contract
      │
      ▼
PASS?
├─ NO → 修代码
└─ YES
      │
      ▼
P1 scenario semantics
      │
      ▼
PASS?
├─ NO → 修 Logical Scenario
└─ YES
      │
      ▼
P2 calibrated S0
      │
      ▼
3 families reachable?
├─ NO → 修 x0 / conflict band
└─ YES
      │
      ▼
P3 reward audit
      │
      ▼
PASS?
├─ NO → 修 reward
└─ YES
      │
      ▼
P4 SAC plumbing
      │
      ▼
PASS?
├─ NO → 修 replay / optimizer
└─ YES
      │
      ▼
P5 6-task shared mini-training
      │
      ▼
有基础 learning signal?
├─ NO → 定位 multitask / reward / scale
└─ YES
      │
      ▼
P6 36-task coverage smoke
      │
      ▼
PASS?
├─ NO → 修 task/coverage contract
└─ YES
      │
      ▼
PRE_STAGE1_PASS
      │
      ▼
Formal Stage1 180 episodes
```

---

# 39. 对当前代码的具体判断

当前已经可以冻结：

```text
11D physical state
2D [Δsteering, Δacceleration]
Native IDM nominal controller
footprint Cut-in semantic monitor
decisive collision > near miss
one-shot event capture
raw action replay contract
training_signal metrics
```

当前暂时不要再重构：

```text
SAC actor/critic architecture
PEARL
MoE
HPTR / Interaction Encoder
```

当前最应该先解决：

```text
1. Merge x0 reachability
2. Roundabout x0 reachability
3. reward 形状是否符合预期
4. event_action_weight 是否关闭
5. non-Cutin maneuver_onset_progress nuisance
6. scenario profile 对 nominal risk 的影响
```

---

# 40. 最终建议

当前 Stage1 不需要再进行“大规模试错训练”。

应该先完成一套：

\[
\boxed{
\text{Reachability}
\rightarrow
\text{Reward}
\rightarrow
\text{SAC Plumbing}
\rightarrow
\text{Mini Learning}
\rightarrow
\text{Coverage}
}
\]

的小规模验证。

只有当：

\[
\boxed{
\text{三类场景都有可学习状态}
+
\text{reward 有正确梯度}
+
\text{SAC 确实能产生学习方向}
}
\]

之后，再启动 36-task Formal Stage1。

这样如果 Formal Stage1 后续仍失败，才有资格把问题进一步归因到：

```text
跨任务表示
共享策略容量
gradient conflict
迁移假设
```

而不是最基础的场景、reward 或 RL plumbing 问题。

---

## 当前代码入口与首轮证据

门槛已落成可执行入口：

```powershell
conda run -n metadrive python -m mvr.scripts.preflight_stage1 --config mvr/configs/mvr_stage1_preflight.yaml --output results/mvr/diagnostics/stage1_preflight.json
```

`mvr/validation/stage1_preflight.py` 负责无模拟器的 reward、one-shot event、raw replay、参数更新、reachability 和 mini-learning 判据；`mvr/scripts/preflight_stage1.py` 负责按配置运行 P0/P1/P2/P4，并将未启用阶段标记为 `skipped`。默认配置的 event action weight 为 0，Inner action 为二维 residual。

首轮运行结果已写入 `results/mvr/diagnostics/stage1_preflight.json`：P0、P1、P3、P4 通过；P2 的 Roundabout 尚未进入 challenge phase，因此 `pre_stage1_pass` 保持为 false。P5/P6、onset nuisance 和 profile 消融尚未启动，不能据此宣称 Formal Stage1 通过。

随后按同一入口单独执行了 P5（18 个训练 episode + 18 个 validation episode），结果保存在 `results/mvr/diagnostics/stage1_p5.json`。P5 仅 Cut-in 产生正向训练信号，Merge/Roundabout 均无 positive reward 或 valid event，trained residual 也未形成相对 base/random 的 family-level 风险改善，因此 P5 失败，P6 不应启动。

当前 base demo 已使用最新 2D residual 与 scenario contract 重新生成到 `results/mvr/diagnostics/base_visualization/`；旧的 `results/mvr/pilot/base_visualization/` 已安全删除。
