# MVR 快速框架预实验与 Stage1 修订计划
## 面向当前控制契约的最小可行验证流程

> 仓库：`SafeDL/META_LEARNING`  
> 当前审计基线：`8b666ef769f7c631b447bbb7b02cc05a8b062042`  
> 当前控制契约：场景、动作与 nominal controller 的一致组合  
> Inner 动作：三维交互残差  
> SUT / nominal controller：lane-stable native IDM  
> 当前目标：**先用少量预实验快速跑通整个 MVR 方法闭环，而不是立即获得论文级 Stage1 结论。**

---

# 1. 结论：Stage1 需要“改定位和预算”，不需要再改核心架构

当前 Stage1 的算法结构不建议继续重构。

已经具备：

- lane-stable native IDM SUT；
- Functional Scenario 交通角色与 completion contract；
- Semantic schedule；
- `u_long / u_maneuver / u_lat` 三维统一 residual；
- Native adversary nominal controller；
- TrafficActionShield；
- ScenarioSemanticMonitor；
- event-time semantic / traffic validity；
- multi-geometry taskbook；
- shared map / interaction encoder；
- shared Inner SAC；
- 后续 PEARL posterior；
- Universal MoE Outer；
- canonical pipeline：

```text
inner_pretrain
→ posterior
→ inner_latent_calibration
→ outer
```

因此当前真正应该修改的是：

\[
\boxed{
\text{“正式 Stage1 科研验收”}
\rightarrow
\text{“Mini Stage1 + 整体 pipeline pilot”}
}
\]

也就是说：

> **现在先验证整个方法框架能不能闭环工作；不要要求 Stage1 先达到论文级 transfer gain，再允许继续运行 Stage2–4。**

---

# 2. 为什么旧 Stage1 计划现在过重

旧计划的 Stage1 是按照“正式科研验收”设计的：

```text
36-task smoke
→ 180-episode Inner pilot
→ 16 cases/task validation
→ trained > random/base
→ ≥2/3 families positive transfer
→ joint validation gain > 0
→ 才允许进入 posterior
```

这套流程适合最终论文实验，但不适合当前阶段。

当前最重要的问题已经变成：

> **Native controller + 3D residual + semantic validity + PEARL posterior + latent calibration + MoE Outer 能否作为一个完整系统从头跑到尾？**

如果此时先花大量预算优化 Stage1，有两个风险：

1. Stage1 训练得很好，但后续 posterior / z-conditioning / MoE Outer 仍存在接口或训练问题；
2. 过早围绕 Stage1 指标调参，使整个方法开发周期过长。

因此建议把实验分成两层：

```text
A. Framework Pilot
   目标：证明整个框架可执行、可学习、信息流有效

B. Formal Experiments
   目标：证明统计意义上的 transferable scenario mining 性能
```

---

# 3. 当前 Stage1 的正确科研含义

即使在快速 Pilot 中，Stage1 的概念定位仍然保持：

> **学习一个共享的 geometry/interaction-conditioned adversarial residual prior。**

其训练任务来自：

```text
Functional Scenario:
    Merge
    Cut-in
    Roundabout

Train geometry:
    每个 family g01 / g02 / g03

Train SUT:
    cautious
    defensive
    normal
    assertive
```

所以正式训练分布仍为：

\[
3\text{ functional families}
\times
3\text{ geometries}
\times
4\text{ SUT styles}
=
36\text{ tasks}
\]

Stage1 训练：

```text
map_encoder
interaction_encoder
shared_feature_encoder
option_embedding
inner_sac
```

冻结：

```text
PEARL context
outcome decoder
MoE Outer
```

因此 Stage1 仍然回答：

> **同一个共享 Inner residual policy 能否在不同 Functional Scenario、不同 geometry 和不同 SUT style 上学习可复用的 adversarial interaction prior？**

但在本轮 Pilot 中，我们只验证这个机制“开始工作”，不要求它已经达到正式泛化性能。

---

# 4. 一个重要术语边界：Stage1 当前不是 unseen topology 实验

当前 geometry catalog 中：

```text
Merge      : SrS
Cut-in     : SSS
Roundabout : O
```

每个 family 使用：

```text
g01 / g02 / g03 → train
g04             → validation
g05             → test
```

因此：

\[
M_{validation}\notin M_{train}
\]

可以称为：

> **held-out / unseen geometry instance**

但同一个 Functional Scenario 内仍使用同一个 road-template / map-code。

因此当前不能称：

> unseen road topology template

也不能称：

> unseen Functional Scenario

当前 Stage1 的正确描述是：

\[
\boxed{
\text{multi-functional joint training}
+
\text{held-out geometry/SUT transfer diagnosis}
}
\]

真正的 unseen topology / unseen Functional Scenario 应留到 R5/R6。

---

# 5. 当前代码已经完成的 Preflight，不应重复重构

当前代码已经新增 SUT-only lane stability diagnostic，并要求三类 Functional Scenario 均满足：

```text
route completion = true
out_of_road = false
routing_target_lane_mismatch = 0
lateral RMS <= 0.30 m
no sustained steering sign oscillation
```

当前还增加了：

- lane-stable native IDM；
- scenario-specific nominal speed；
- curve-safe speed；
- explicit Merge branch/mainline role；
- SUT route completion 作为 test endpoint；
- target collision / hard adversary violation 的 early termination；
- `scenario_contract_v4`。

因此 Framework Pilot 之前不应再继续改 SUT 或场景结构，除非上述 preflight 自身失败。

---

# 6. 推荐的新整体流程

建议采用：

```text
Phase 0
Current-contract Preflight
        ↓
Phase 1
Mini Stage1
        ↓
Phase 2
Mini PEARL Posterior
        ↓
Phase 3
Mini Latent Calibration
        ↓
Phase 4
Mini MoE Outer
        ↓
Phase 5
Validation-split End-to-End Pilot
        ↓
FRAMEWORK_PILOT_PASS
        ↓
再开始 Formal Stage1 / R1–R4
```

核心思想是：

> **先纵向跑完整条技术路线，再横向扩大每个阶段的实验预算。**

---

# 7. Phase 0：只保留必要的 Preflight

## 7.1 必须通过

```powershell
conda run -n metadrive python -m pytest mvr/tests -q
```

以及：

```powershell
conda run -n metadrive python -m compileall -q mvr
```

## 7.2 SUT-only diagnostic

执行：

```powershell
conda run -n metadrive python -m mvr.scripts.diagnose_sut ^
  --output results/mvr/pilot/sut_only_diagnostic.json
```

只要求：

```text
3/3 families route complete
0 out-of-road
0 routing-target mismatch
no sustained oscillation
```

这是 Hard Gate。

## 7.3 Base behavior visualization

执行当前 base-only 可视化：

```powershell
conda run -n metadrive python -m mvr.scripts.visualize_stage1_base ^
  --config mvr/configs/mvr_stage1_smoke.yaml ^
  --output results/mvr/pilot/base_visualization
```

人工确认：

```text
Merge:
红车确实从 branch merge，蓝车在 mainline

Cut-in:
红车确实从 adjacent lane 合法向 target lane 切入

Roundabout:
双方沿合法 entry/exit route 行驶

Base:
没有明显乱打方向、越界或异常 Shield 接管
```

Pilot 阶段只要 Base 行为正确即可，不需要在这里分析危险发现率。

---

# 8. 不建议在 Pilot 前运行完整 G3

当前正式 G3：

```text
16 configs
× 3 families
× 4 train SUTs
= 192 simulator episodes
```

对于“先跑通框架”的目的太贵。

因此：

> **Framework Pilot 不把 full G3 作为 blocker。**

可选做法：

- 暂时跳过；
- 或后续为 `validate_mvr.py` 增加 `--configurations 4`，形成 G3-lite。

正式论文实验前再执行完整 G3。

---

# 9. Phase 1：Mini Stage1

当前 `mvr_stage1_smoke.yaml` 的 Stage1 配置已经合适：

```yaml
inner:
  action_dim: 3
  episodes_per_task: 1
  updates_per_episode: 1
  batch_size: 32
```

保留全部：

```text
36 train tasks
```

所以 Mini Stage1 只消耗：

\[
\boxed{36\text{ simulator episodes}}
\]

这样做比只挑一个 Merge task 更好，因为一次就能检查：

- 3 个 Functional Scenario；
- 9 个 train geometry；
- 4 个 train SUT；
- variable candidate sets；
- shared 3D residual action；
- shared map/interaction encoder。

---

# 10. Mini Stage1 不再要求正式 transfer superiority

旧计划要求：

```text
trained > random
trained > base
positive gain in 2/3 families
```

对于只有：

```text
1 episode/task
1 update/episode
```

的 Mini Stage1 不合理。

Mini Stage1 的 PASS 条件改成：

```text
36/36 tasks visited
3/3 families visited
9/9 train geometries visited
4/4 train SUTs visited

optimizer_updates > 0
actor/critic/alpha loss finite
no NaN/Inf

raw residual finite
candidate action finite
executed action finite

semantic monitor active on executable maneuvers
traffic validity logic works
checkpoint saves/reloads
```

另外增加两个软诊断：

```text
trained residual not identically zero
reward distribution not completely constant
```

不要求 trained policy 此时已经显著优于 Base / Random。

Mini Stage1 的任务只是验证：

> **shared Inner learning path 是真的，而不是空管道。**

---

# 11. Mini Stage1 后做一个极小 validation

当前 `validate_mvr --mode inner` 已经支持：

```text
Base
Random Residual
Trained Inner
```

并使用：

```text
validation SUT
+
validation geometry
```

建议：

```powershell
conda run -n metadrive python -m mvr.scripts.validate_mvr ^
  --config mvr/configs/mvr_stage1_smoke.yaml ^
  --checkpoint results/mvr/pilot/inner_pretrain.pt ^
  --mode inner ^
  --cases-per-task 1 ^
  --output results/mvr/pilot/stage1_validation.json
```

总计：

```text
3 families
× 1 validation task
× 3 policies
= 9 episodes
```

这里只看：

```text
all policies executable
semantic / traffic metrics finite
trained residual 对 trajectory/outcome 有可观测影响
无明显 invalid-policy collapse
```

如果 trained policy 没有优于 Base：

> **记录，但不因此停止 Framework Pilot。**

只有以下情况才停止：

```text
trained policy 大面积越界
Shield 几乎每步完全接管
NaN
SAC action 完全无效
场景语义被破坏
```

---

# 12. Phase 2：Mini Posterior

当前 smoke 配置：

```yaml
posterior:
  episode_budget: 4
  support_episodes: [1]
```

足够用于框架检查。

目标不是证明 few-shot accuracy，而是验证：

```text
support episode
→ episode token
→ PEARL Product-of-Gaussians posterior
→ z
```

形成真实信息流。

## PASS 条件

```text
posterior loss finite
mu/logvar finite
empty context = prior
support context 后 z != prior（至少部分 task）
support / target 无 leakage
checkpoint saves/reloads
```

不要求 K=1 此时显著提高危险发现率，这属于正式 Stage2。

---

# 13. Phase 3：Mini Inner Latent Calibration

当前：

```yaml
inner_latent_calibration:
  simulator_episode_budget: 4
  support_episodes: 1
  updates_per_episode: 1
```

保持即可。

目标只验证：

> Inner SAC 在 `z != 0` 的输入下仍然可以更新，并开始建立 `z → adversarial behavior` 的依赖。

## PASS 条件

```text
calibration groups > 0
replay contains z-conditioned target transitions
loss finite
z-conditioned Inner forward valid
checkpoint saves/reloads
```

建议增加一个极小诊断：

在完全相同：

```text
state
scene
x0
option
```

条件下比较：

```text
z = 0
z = posterior(z_support)
```

只要求 Inner action 存在非零差异，不要求差异已经方向正确。

---

# 14. Phase 4：Mini MoE Outer

当前 smoke：

```yaml
outer:
  ppo_epochs: 1
  batch_size: 32
  episodes_per_task: 1
```

保留全部 36 个 train tasks。

因此：

\[
\boxed{36\text{ Outer simulator episodes}}
\]

目标是检查：

```text
h_scene + z
→ MoE Router
→ expert
→ candidate / x0
→ executable scenario
→ reward
→ PPO update
```

是否形成闭环。

## PASS 条件

```text
router logits finite
expert selection valid
candidate selection valid
continuous x0 valid
36 train tasks executable
PPO loss finite
Outer parameters changed
no family-specific neural head required
```

不要求 MoE 已形成可解释专家分工，这是正式实验问题。

---

# 15. 整个 Mini Pipeline 的训练预算

使用当前 smoke 预算：

| 阶段 | simulator episodes |
|---|---:|
| Mini Stage1 | 36 |
| Posterior | 4 |
| Inner latent calibration | 4 |
| Outer | 36 |
| **合计** | **80** |

因此完整训练只需要约：

\[
\boxed{80\text{ simulator episodes}}
\]

这比先运行 180 Stage1 + formal validation + full G3 轻很多，同时已经完整穿过论文的四个核心训练阶段。

---

# 16. 建议新增一个专用 `mvr_pilot.yaml`

不要长期把：

```text
mvr_stage1_smoke.yaml
```

同时当作 Stage1 smoke 和 full-pipeline pilot。

建议复制为：

```text
mvr/configs/mvr_pilot.yaml
```

核心配置：

```yaml
schema: transferable_scenario_miner_v3
seed: 11

training:
  device: cuda
  family_filter: all
  step_budget: 240

inner:
  action_dim: 3
  episodes_per_task: 1
  updates_per_episode: 1
  batch_size: 32

posterior:
  episode_budget: 4
  support_episodes: [1]

inner_latent_calibration:
  simulator_episode_budget: 4
  support_episodes: 1
  updates_per_episode: 1
  batch_size: 32

outer:
  ppo_epochs: 1
  batch_size: 32
  episodes_per_task: 1

evaluation:
  total_episode_budget: 4
  support_shots: [0, 1]
  seeds: [11]
  deterministic: true
```

## 重要

不要把 `step_budget` 降回 60。

当前 scenario contract 要求完整 SUT route completion，Cut-in / Roundabout 可能需要 240–300 steps。

---

# 17. 建议只增加一个很小的代码功能：Pilot Evaluation Regime

当前正式 evaluator 的：

```text
R1–R4
```

会使用 train/test split。

Framework Pilot 不建议反复访问：

```text
final test SUT
final test geometry
```

否则开发期会逐渐污染正式 test set。

因此建议新增一个**非论文 regime**：

```text
PV_validation_sut_validation_geometry
```

选择：

```text
sut_split == validation
geometry_split == validation
functional_split == train
```

即当前：

```text
fast_small_gap
+
g04
+
Merge/Cut-in/Roundabout
```

这个 regime 仅用于工程开发，不进入论文 R1–R4 表格。

---

# 18. Pilot Evaluation 预算

建议：

```text
3 validation tasks
× K={0,1}
× 4 total episodes
× 1 seed
=
24 simulator episodes
```

于是完整 Framework Pilot 总预算大约：

```text
80 training
+
9 Stage1 quick validation
+
24 end-to-end validation
≈ 113 simulator episodes
```

相对于正式实验非常小。

---

# 19. Phase 5：End-to-End Pilot 真正要验证什么

完成 Outer checkpoint 后，在：

```text
validation SUT
+
validation geometry
```

运行：

```text
K=0
K=1
```

目标不是比较统计显著性，而是验证整个论文机制：

```text
Episode 1
    ↓
trajectory / outcome
    ↓
context token
    ↓
posterior z update
    ↓
MoE Router / Outer receives new z
    ↓
next x0 proposal
    ↓
Inner receives z
    ↓
next adversarial rollout
```

---

# 20. End-to-End Pilot PASS 条件

## E1 — support 真正更新 posterior

```text
K=0:
z stays prior

K=1:
z changes after support
```

## E2 — posterior 能影响搜索器

至少一个 validation task 中：

```text
Outer proposal under posterior z
!=
Outer proposal under prior z
```

可以表现为：

```text
candidate probability changed
expert mixture changed
x0 changed
```

不要求一定更危险。

## E3 — posterior 能进入 Inner

验证：

```text
Inner features / action depend on z
```

而不是 z 只影响 Outer。

## E4 — MoE 真正在 active path

验证：

```text
router weights finite
expert index / mixture 被记录
```

不能退化成 family hard-routing。

## E5 — 场景结果可报告

每个 episode 都能输出：

```text
geometry hash
candidate
x0
3D residual policy checkpoint/hash
semantic event
traffic validity
failure signature
episode seed
```

## E6 — K-shot 支持计入同一预算

仍保持：

```text
total budget = 4
```

而不是：

```text
4 query + 1 free support
```

---

# 21. Pilot 阶段哪些“效果差”可以接受

以下情况**不阻止 Framework Pilot PASS**：

```text
K=1 暂时没有优于 K=0
trained Inner 暂时没有显著优于 Base
MoE experts 暂时没有明显 specialization
failure discovery rate 很低
只有一个 seed
```

因为当前预算太小，不应该对此做科学解释。

---

# 22. Pilot 阶段哪些问题必须停止

以下问题必须修复后再继续：

```text
SUT route 不稳定
Functional Scenario 语义错误
Base policy 自身违法/失控
3D residual 不能影响 executed action
Shield 每步完全接管
event semantic validity 错误
posterior 永远等于 prior
z 没有进入 Outer / Inner
MoE 没有进入 active path
Outer 生成不可执行 x0
checkpoint 无法 resume
concrete scenario 无法 replay
NaN / Inf
```

这些属于方法实现错误，而不是小样本训练不足。

---

# 23. 快速 Pilot 推荐执行流程

## Step 1 — Tests

```powershell
conda run -n metadrive python -m pytest mvr/tests -q
```

## Step 2 — SUT diagnostic

```powershell
conda run -n metadrive python -m mvr.scripts.diagnose_sut ^
  --output results/mvr/pilot/sut_only_diagnostic.json
```

## Step 3 — Base GIF

```powershell
conda run -n metadrive python -m mvr.scripts.visualize_stage1_base ^
  --config mvr/configs/mvr_pilot.yaml ^
  --output results/mvr/pilot/base_visualization
```

## Step 4 — 一次跑完整 canonical pipeline

```powershell
conda run -n metadrive python -m mvr.scripts.train_mvr ^
  --config mvr/configs/mvr_pilot.yaml ^
  --output results/mvr/pilot ^
  --stop-after outer
```

这一步应该依次产生：

```text
inner_pretrain.pt
posterior.pt
inner_latent_calibration.pt
outer.pt
manifest.json
```

## Step 5 — 极小 Stage1 validation

```powershell
conda run -n metadrive python -m mvr.scripts.validate_mvr ^
  --config mvr/configs/mvr_pilot.yaml ^
  --checkpoint results/mvr/pilot/inner_pretrain.pt ^
  --mode inner ^
  --cases-per-task 1 ^
  --output results/mvr/pilot/stage1_validation.json
```

## Step 6 — End-to-End validation-split pilot

建议新增：

```powershell
conda run -n metadrive python -m mvr.scripts.evaluate_pilot ^
  --config mvr/configs/mvr_pilot.yaml ^
  --checkpoint results/mvr/pilot/outer.pt ^
  --output results/mvr/pilot/e2e_validation.json
```

选择：

```text
validation SUT
+
validation geometry
```

不要使用正式 g05 / late_response test set。

---

# 24. Framework Pilot 的最终判定

推荐最终只做三级判定。

## PASS-A：Execution

```text
三类场景均可执行
SUT / Base 稳定
checkpoint pipeline 完整
```

## PASS-B：Learning Signal

```text
Inner updates
Posterior updates
Latent calibration updates
Outer PPO updates
全部 finite
```

## PASS-C：Information Flow

```text
support → z changes
z → Outer changes
z → Inner changes
Outer → executable x0
Inner → executable residual behavior
rollout → valid semantic outcome
```

全部满足：

```text
FRAMEWORK_PILOT_PASS
```

## 33. 已执行的 framework pilot（seed 11）

已按本计划完成一次 80-episode、validation-only 的框架预实验。当前
当前控制契约下，三类 SUT-only 诊断均通过；Mini Inner、posterior、
z-conditioned Inner、MoE Outer 均完成有限更新并保存 checkpoint；g04/
fast_small_gap 上 K=0/K=1 的 24 个严格记账 episode 验证了 posterior 到
Inner 与 Outer 的信息流。完整可复核记录见
`results/mvr/pilot/assessment.md`、`manifest.json`、
`stage1_validation.json` 和 `e2e_validation.json`。

该结论仅为 `FRAMEWORK_PILOT_PASS`，不构成正式 Stage1 性能、跨几何迁移或
方法优越性的主张；正式验收仍须遵循本文件规定的独立固定预算与 transfer gates。

---

# 25. Framework Pilot 通过后，才恢复“正式 Stage1”

Pilot PASS 后，再使用：

```text
mvr_stage1.yaml
```

做正式 Stage1。

此时恢复旧计划中的科研要求：

```text
episodes_per_task = 5
36 × 5 = 180 episodes

cases_per_task = 16

Base
vs
Random Residual
vs
Trained Residual
```

并正式要求：

```text
validity >= threshold
trained > Base
trained > Random
positive gain in >= 2/3 families
validation SUT + validation geometry 上 positive transfer
```

完整 G3 也在这个阶段恢复。

---

# 26. 正式 Stage1 与 Framework Pilot 的区别

| 项目 | Framework Pilot | Formal Stage1 |
|---|---:|---:|
| 目的 | 跑通完整技术路线 | 验证 transferable Inner prior |
| Inner episodes/task | 1 | 5 起 |
| Inner episodes | 36 | 180+ |
| Stage1 validation cases/task | 1 | 16 |
| G3 | 跳过 / lite | 完整 |
| Seed | 1 | 多 seed 后续 |
| 要求 trained > baseline | 否 | 是 |
| Posterior/Outer | 立即继续 | Stage1 通过后继续 |
| 可以做论文性能结论 | 否 | 可以形成阶段性证据 |

---

# 27. 对 Stage1 研究问题的最终表述

正式 Stage1 应表述为：

> **Can a shared interaction-residual adversarial controller learn transferable challenge behaviors from multiple seen functional scenarios, SUT styles, and road geometries, and retain useful zero-shot controllability on held-out SUT/geometry combinations?**

中文：

> **在多个已见功能场景、不同 SUT 风格和多个道路几何上联合训练时，共享的交互残差对抗策略能否学习可复用的安全挑战行为，并在未参与训练的 SUT 与道路几何组合上保持有效的零样本对抗控制能力？**

---

# 28. Stage1 不应该被表述为什么

当前不要写：

```text
Stage1 验证 few-shot adaptation
```

因为 posterior 尚未训练。

不要写：

```text
Stage1 验证 unseen functional scenario
```

因为 Merge/Cut-in/Roundabout 全部参与训练。

不要写：

```text
Stage1 验证 unseen road topology
```

因为同 family 的 g01–g05 仍共享相同 map template。

不要写：

```text
Stage1 已证明完整 MVR superiority
```

因为 Outer/PEARL/MoE 尚未形成最终搜索闭环。

---

# 29. 整篇论文的快速验证顺序

当前最推荐的开发策略是：

```text
① 场景/SUT/Base correctness
        ↓
② Mini Inner learning
        ↓
③ Mini PEARL posterior
        ↓
④ Mini z-conditioned Inner
        ↓
⑤ Mini MoE Outer
        ↓
⑥ Validation-split K=0/K=1 end-to-end
        ↓
⑦ 确认框架成立
        ↓
⑧ 再扩大 Stage1
        ↓
⑨ 再扩大 Stage2–4
        ↓
⑩ 正式 R1–R4 + 多 seed
```

这样最符合当前研究阶段。

---

# 30. 当前最需要的代码修改只有两项

## 必须 1：增加 `mvr_pilot.yaml`

避免混用：

```text
stage1 smoke
formal Stage1
full-pipeline pilot
```

三种实验配置。

## 必须 2：增加 validation-only end-to-end evaluator

例如：

```text
PV_validation_sut_validation_geometry
```

它必须使用：

```text
fast_small_gap
+
g04
```

而不是：

```text
late_response
+
g05
```

这样在方法开发过程中可以重复跑 Pilot，而不会污染最终 test set。

---

# 31. 暂时不需要修改的内容

Framework Pilot 前不建议再修改：

```text
3D residual action definition
LaneStableNativeIDMPolicy
ScenarioActionAdapter
TrafficActionShield
ScenarioSemanticMonitor
PEARL architecture
MoE architecture
R1–R4 formal protocol
geometry/task schema
```

除非 Preflight 或 end-to-end pilot 暴露出明确 contract bug。

---

# 32. 最终建议

当前不要再把“Stage1 论文级 PASS”作为继续开发的前置条件。

应该改成：

\[
\boxed{
\text{Preflight}
\rightarrow
\text{Mini Stage1}
\rightarrow
\text{Mini Stage2}
\rightarrow
\text{Mini Stage3}
\rightarrow
\text{Mini Stage4}
\rightarrow
\text{E2E Pilot}
}
\]

只要证明：

\[
\boxed{
\text{场景正确}
+
\text{控制正确}
+
\text{各模块可学习}
+
\text{support 能改变 z}
+
\text{z 能改变搜索行为}
}
\]

就可以认为：

```text
FRAMEWORK_PILOT_PASS
```

然后再回到正式 Stage1，用 180+ episodes 和严格 transfer gate 做科研验证。

这比当前旧计划“先把 Stage1 做到论文级，再允许进入 Posterior”更适合现在的研究进度，也能最快判断 MoE-PEARL-ScenarioMiner 的完整 idea 是否真正可行。
