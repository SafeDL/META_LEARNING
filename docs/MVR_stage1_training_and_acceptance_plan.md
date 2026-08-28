# MVR Stage1 目标与完整工作流更新

系统的gpu环境在：
conda activate metadrive

> 仓库：`SafeDL/META_LEARNING`  
> 当前代码基线：`0d02a6d19bd7e860413051fea763b9138e7ec14c`  
> 当前方法：`mvr/`  
> 当前状态：**Framework Pilot 已跑通；Formal Stage1 尚未开始**  
> 本文最低目标：**unseen SUT + unseen road geometry + few-shot transferable scenario mining**

---

# 1. 本文档为什么需要更新

旧版 Stage1 文档将训练流程设计成：

```text
Preflight
→ Formal Stage1
→ Stage1 必须达到正式 transfer gate
→ Posterior
→ Latent calibration
→ Outer
```

这套逻辑适合最终论文实验，但已经不符合当前开发状态。当前代码已经实际跑通：

```text
Mini Inner Pretrain
→ Mini PEARL Posterior
→ Mini Inner Latent Calibration
→ Mini MoE Outer
→ Validation-only End-to-End Pilot
```

并产生 `results/mvr/pilot/`，当前状态为：

```text
FRAMEWORK_PILOT_PASS
```

因此 Stage1 文档必须拆成两层：

```text
Level A — Framework-Pilot Stage1
目的：验证完整方法链路是否可执行、可更新、信息流是否存在
状态：已完成

Level B — Formal Stage1
目的：正式验证 transferable Inner adversarial prior 的性能
状态：下一步
```

---

# 2. 当前代码设计是否已经符合论文预期

总体判断：

\[
\boxed{\text{YES，框架级设计已经符合预期}}
\]

当前实现形成了完整的层次化测试机制：

```text
Outer x0 / interaction proposal
        ↓
Native navigation
        ↓
Functional-Scenario semantic schedule
        ↓
Native IDM nominal traffic behavior
        ↓
3-D adversarial interaction residual
        ↓
TrafficActionShield
        ↓
ScenarioSemanticMonitor
        ↓
event-time semantic / traffic validity
        ↓
trajectory + outcome
        ↓
PEARL posterior z
        ↓
MoE Outer / Inner conditioning
```

当前对抗策略不是“任意 SAC steering/throttle policy”，而是：

\[
\pi_{adv}^{F}
=
Shield_F
\circ
[
\pi_{nominal}^{F}
+
A_F(\Delta\pi_{SAC})
]
\]

其中：

- \(\pi_{nominal}^{F}\)：执行给定 Functional Scenario 的正常交通行为；
- \(A_F\)：将统一 interaction residual 映射到当前场景；
- \(\Delta\pi_{SAC}\)：学习对抗压力；
- Shield：最后的交通/动力学硬约束。

因此当前代码实现的是：

> **在给定场景语义下执行正确交通机动，再学习如何把这个机动变得更具挑战性。**

---

# 3. “合理对抗行为”必须继续作为 Stage1 的硬契约

最终可报告 failure 必须满足：

\[
\boxed{
ReportableFailure
=
EventSemanticValid
\land
EventTrafficValid
\land
TargetConsequence
}
\]

## 3.1 Cut-in

正常交通意图：

```text
adjacent source lane
→ legal merge window
→ target-lane intrusion
→ target-lane occupation
```

SAC 可以改变：

```text
切入 timing
纵向速度 / acceleration pressure
gap closing
有限 lateral aggressiveness
```

但不能：

```text
不执行 Cut-in
仅在相邻车道靠近 SUT
然后利用 TTC/distance 获得 failure
```

## 3.2 Merge

角色固定：

```text
Red adversary = one-lane incoming branch / ramp
Blue SUT      = multi-lane mainline
```

两车随后进入共同 downstream / conflict region。

SAC 可以改变：

```text
merge arrival timing
gap
acceleration/deceleration
limited lateral residual
```

但不能脱离 ramp/merge route 直接撞击 SUT。

## 3.3 Roundabout

两车必须：

```text
follow declared entry/exit route
→ enter shared conflict phase
```

SAC 只改变：

```text
arrival timing
approach speed
yield / press intensity
limited lateral residual
```

---

# 4. 当前控制接口已经适合跨场景共享

当前统一 Inner action：

\[
\boxed{
a_t^{res}
=
[u_{long},u_{maneuver},u_{lat}]
}
\]

- `u_long`：逐 timestep 的纵向 interaction pressure；
- `u_maneuver`：低频、平滑、状态化的 conflict/maneuver timing reference；
- `u_lat`：合法名义路径附近的有限横向 aggressiveness。

不同场景的差异被限制在：

```text
ScenarioActionAdapter
+
TrafficBehaviorContract
+
Native nominal controller
```

而不是不同 SAC network head。

---

# 5. 当前 Framework Pilot 已经验证了什么

## 5.1 Preflight 已通过

SUT-only diagnostic 在 Merge、Cut-in、Roundabout 中均验证：

```text
route completion = true
out-of-road = false
routing-target lane mismatch = 0
no sustained steering-sign oscillation
```

因此旧版本中蓝色 SUT 的剧烈左右摆动已经不再是当前方法链路的 blocker。

## 5.2 Mini Stage1 已覆盖全部 36 个训练 task

Pilot Inner：

```text
36 train tasks
1 episode / task
1 optimizer update / episode
```

已经覆盖：

```text
3 Functional Scenario
9 train geometries
4 train SUT profiles
2 Cut-in candidates
2 Merge candidates
3 Roundabout candidates
3 adversarial options
```

并完成有限、非 NaN 的 SAC 更新。

## 5.3 Mini Posterior 已跑通

```text
support episode
→ evidence token
→ PEARL-style Product-of-Gaussians posterior
→ z update
```

## 5.4 Mini Latent Calibration 已跑通

```text
z != prior
→ z-conditioned Inner
→ optimizer update
```

## 5.5 Mini MoE Outer 已跑通

```text
h_scene + z
→ MoE Router
→ expert
→ candidate / x0
→ executable scenario
→ reward
→ PPO update
```

## 5.6 Validation-only End-to-End 已跑通

当前 Pilot 使用：

```text
validation SUT = idm_fast_small_gap
validation geometry = g04
K = 0 / 1
total episode budget = 4
```

已验证：

```text
all_finite
K=0 stays at prior
K=1 updates posterior
Inner responds to posterior
MoE is active
posterior changes Outer proposal
```

所以当前合理的工程结论是：

\[
\boxed{FRAMEWORK\_PILOT\_PASS}
\]

---

# 6. 当前 Pilot 不能解释成什么

当前 Pilot 只证明：

> 方法链路能够工作。

不能解释为：

```text
Inner 已经学到强 transferable adversarial policy
PEARL 已经具有可靠 few-shot gain
MoE 已经优于非-MoE
K=1 已经优于 K=0
R4 已经成立
```

因为当前总训练预算只有约 80 simulator episodes。

---

# 7. Stage1 文档最关键的修改

旧流程：

```text
Formal Stage1 PASS
        ↓
Posterior
```

现在应改为：

```text
Framework Pilot
    ├─ Mini Stage1
    ├─ Mini Posterior
    ├─ Mini Calibration
    └─ Mini Outer
        ↓
FRAMEWORK_PILOT_PASS
        ↓
Formal Stage1
        ↓
Formal Posterior
        ↓
Formal Calibration
        ↓
Formal Outer
        ↓
R1–R4
```

Pilot 的目标是纵向验证整条方法链路；Formal experiment 的目标是横向扩大预算并建立科研证据。

---

# 8. 当前最需要区分的两个 Stage1

## 8.1 Pilot Stage1

目标：

> Inner training mechanism 是否能够工作？

预算：

```text
1 episode / task
36 episodes total
1 update / episode
```

PASS：

```text
36/36 task coverage
loss finite
3D residual finite
semantic contract valid
no policy collapse
checkpoint can reload
```

当前这一层已经通过。

## 8.2 Formal Stage1

目标：

> shared Inner adversarial residual prior 是否真的具有 transfer ability？

建议第一轮：

```text
5 episodes / task
36 tasks
=
180 episodes
```

正式比较：

```text
Base
Base + Random Residual
Base + Trained Residual
```

这里才要求：

```text
trained > Base
trained > Random
positive transfer on held-out validation SUT + geometry
```

---

# 9. Formal Stage1 前仍需修正的 3 个小问题

这些问题不要求重新设计整体架构，但在把 Stage1 结果作为正式论文证据前建议修复。

## 9.1 Cut-in 的 `target_lane_intrusion` 判据应改为车辆 footprint 级别

当前判据偏宽，应改成车辆 footprint 与 target lane corridor 的真实重叠，例如：

\[
|e_y^{target}|
<
\frac{w_{lane}}{2}
+
\frac{w_{vehicle}}{2}
\]

或直接使用 bounding box 与 target-lane polygon overlap。

必须新增反例：

```text
maneuver latched
+
车辆尚未真实跨入 target lane
+
low TTC
→ semantic failure = false
```

## 9.2 FailureAnalyzer 应使用“最终 decisive event”

SemanticMonitor 允许：

```text
near-miss
→ later target collision
```

其中 collision 应具有更高优先级。

Formal Stage1 前应保证：

```text
collision > near-miss
```

建议 Analyzer 使用 monitor 最终 latched event，而不是 episode 中最早的 event 记录。

## 9.3 Near-miss positive bonus 只奖励新事件一次

当前 near-miss event latch 后，后续 timestep 可能继续满足奖励条件。

建议新增：

```text
event_just_captured
```

或：

```text
new_valid_near_miss
```

只有第一次有效 near-miss 获得 success bonus。后续保留 continuous criticality shaping，但不能重复领取 event reward。

---

# 10. 上述三个修正不会改变论文方法

修正后仍保持：

```text
Native traffic-compliant nominal controller
+
3D interaction residual
+
Traffic Shield
+
Semantic Monitor
```

只是在：

```text
semantic validity
event priority
reward bookkeeping
```

上更严格。

---

# 11. 更新后的当前工作流

```text
────────────────────────────────────────
Phase A — Framework Integration
────────────────────────────────────────

A0  SUT / Scenario / Base Preflight
        ↓
A1  Mini Stage1
        ↓
A2  Mini Posterior
        ↓
A3  Mini Latent Calibration
        ↓
A4  Mini MoE Outer
        ↓
A5  validation-only K=0/K=1 E2E
        ↓
FRAMEWORK_PILOT_PASS

当前状态：
✓ 已完成

────────────────────────────────────────
Phase B — Formal Mechanism Verification
────────────────────────────────────────

B0  修复 3 个 semantic/event/reward 小问题
        ↓
B1  Formal Stage1
        ↓
B2  Formal Stage1 Validation
        ↓
STAGE1_PASS
        ↓
B3  Formal Posterior
        ↓
B4  Formal Latent Calibration
        ↓
B5  Formal MoE Outer
        ↓
FULL_METHOD_READY

────────────────────────────────────────
Phase C — Paper Evaluation
────────────────────────────────────────

R1 seen SUT / seen geometry
R2 unseen SUT / seen geometry
R3 seen SUT / unseen geometry
R4 unseen SUT / unseen geometry
        ↓
K = 0 / 1 / 2 / 4
        ↓
multiple seeds
```

---

# 12. Formal Stage1 的新目标

正式 Stage1 应表述为：

> **在多个已见 Functional Scenario、多个道路 geometry 和不同 SUT 风格上联合训练时，共享的 interaction-residual adversarial policy 能否学习“在正确交通语义内施加挑战”的可迁移控制先验，并在 held-out SUT + held-out geometry 上保持有效的 zero-shot adversarial controllability？**

形式化：

\[
\pi_{inner}
:
(s_t,h_{scene},x_0,o)
\rightarrow
[u_{long},u_{maneuver},u_{lat}]
\]

同时必须满足：

\[
\pi_{adv}^{F}\in\Pi_{adv,F}
\]

---

# 13. Formal Stage1 的训练分布

保持：

```text
Functional Scenario:
    Merge
    Cut-in
    Roundabout

Train geometry:
    g01 / g02 / g03

Validation geometry:
    g04

Final test geometry:
    g05

Train SUT:
    cautious
    defensive
    normal
    assertive

Validation SUT:
    fast_small_gap

Final test SUT:
    late_response
```

Train tasks：

\[
3\times3\times4=36
\]

---

# 14. Formal Stage1 第一轮预算

建议：

```yaml
inner:
  episodes_per_task: 5
  updates_per_episode: 8
  batch_size: 64
```

即：

\[
36\times5=180\text{ episodes}
\]

若 validation curve 仍持续改善，再扩展到 10 episodes/task。

---

# 15. Formal Stage1 baseline 更新

旧版 `Zero / No-op adversary` 不再合适。

现在正式 baseline 应为：

### B0 — Base

```text
Native nominal controller
+
zero interaction residual
```

### B1 — Base + Random Residual

```text
Native nominal controller
+
random 3D residual
```

### Method — Base + Trained Residual

```text
Native nominal controller
+
trained shared Inner SAC
```

比较的是学习到的 adversarial residual 是否比正常交通和随机扰动更有效。

---

# 16. Formal Stage1 Validation

建议每个 Functional Scenario 使用 16 个固定 initial-condition cases。

所有 policy 共享：

```text
geometry
candidate
x0
SUT
option
episode seed
```

只改变 residual policy。

验证：

```text
V1 seen SUT + seen geometry
V2 validation SUT + seen geometry
V3 seen SUT + validation geometry
V4 validation SUT + validation geometry
```

V4 是最重要的 Stage1 transfer diagnosis。

---

# 17. Stage1 成功标准必须同时包含挑战性和合理性

推荐报告：

```text
valid critical rate
valid target collision rate
valid near-miss rate
median min TTC
median min distance
invalid rate
traffic violation rate
shield intervention rate
mean candidate→executed action distance
```

不能只看 failure rate。

---

# 18. Formal Stage1 Hard Gate

## G1 — Engineering

```text
pytest PASS
compileall PASS
checkpoint reload PASS
NaN / Inf = 0
```

## G2 — Coverage

```text
36 / 36 tasks
3 / 3 families
9 / 9 train geometries
4 / 4 train SUTs
candidate coverage
option coverage
```

## G3 — Semantic / Traffic

```text
valid_rate >= 0.90 preferred
hard minimum >= 0.80
traffic violation not higher than Random
no Functional Scenario semantic violation
```

## G4 — Learned adversarial effect

整体：

```text
Trained > Base
Trained > Random
```

且至少 2/3 Functional Scenario 出现正提升。

## G5 — Joint transfer

在：

```text
validation SUT + validation geometry
```

上：

\[
G_{joint}>0
\]

---

# 19. Stage1 不需要证明什么

Stage1 不证明：

```text
few-shot K=1/2/4 gain
unseen Functional Scenario
unseen topology template
完整 MVR superiority
```

Stage1 只证明：

\[
\boxed{
zero\text{-}shot transferable Inner prior
}
\]

---

# 20. 更新后的执行顺序

当前 Framework Pilot 已完成，因此下一步建议：

1. 修复三个局部语义/事件/奖励问题；
2. `pytest` + `compileall`；
3. 重新跑 SUT / Base preflight；
4. 运行 Formal Stage1；
5. 运行 Formal Stage1 validation；
6. Stage1 PASS 后 resume posterior → calibration → outer；
7. 最后正式 R1–R4 + K=0/1/2/4 + multi-seed。

---

# 21. 是否需要重新跑 Framework Pilot

如果只修改：

```text
Cut-in intrusion criterion
event priority
one-shot near-miss bonus
```

建议做一次极小 regression pilot：

```text
80 training episodes
+
24 E2E validation episodes
```

确认 `FRAMEWORK_PILOT_PASS` 仍然成立。

之后冻结方法实现，进入 Formal Stage1。

---

# 22. 文档维护建议

建议以后明确维护两个文档：

```text
docs/MVR_快速框架预实验与Stage1修订计划.md
```

负责：

> integration / pilot / debugging

以及：

```text
docs/MVR_stage1_training_and_acceptance_plan.md
```

负责：

> Formal Stage1 scientific acceptance

避免一个文档同时承担工程 smoke、完整方法 integration 和论文 performance gate 三个目的。

---

# 23. 当前整体判断

当前代码已经达到：

\[
\boxed{
Architecture\ Correct
+
Control\ Semantics\ Integrated
+
Framework\ Information\ Flow\ Verified
}
\]

并已有：

```text
FRAMEWORK_PILOT_PASS
```

所以当前主要问题已经从：

> “这个方法能不能跑？”

转变为：

> “在严格合理的 adversarial scenario contract 下，这个方法是否真正具有可测量的 transferable advantage？”

这正是 Formal Stage1 应回答的问题。

---

# 24. 最终 Stage1 一句话目标

> **Stage1 aims to learn a shared interaction-residual adversarial prior that increases safety-critical pressure beyond a lawful nominal traffic policy while preserving the semantic and traffic validity of each Functional Scenario, and to verify that this control prior transfers zero-shot to held-out SUT/geometry combinations.**

中文：

> **Stage1 的目标是在不同功能场景、道路几何与 SUT 风格上学习一个共享的交互残差对抗先验：该策略必须在保持各 Functional Scenario 交通语义与行为有效性的前提下，相比正常交通基座产生更强的安全挑战，并能够零样本迁移到未参与训练的 SUT 与道路几何组合。**
