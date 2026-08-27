# MVR 第一阶段训练与验收计划
## Transferable Inner Adversarial Prior Pretraining

> 仓库：`SafeDL/META_LEARNING`  
> 审计基线：`8047284d37430898fb869bfc8945c3d0cb6a81fb`  
> 当前 active 方法：`mvr/`  
> 第一阶段：`inner_pretrain`  
> 本文核心最低目标：**unseen SUT + unseen road geometry + few-shot transferable scenario mining**

---

# 1. 最新代码审计结论

本次代码修改已经基本落实上一版重构计划的主体结构，当前 `mvr/` 已不再只是“held-out IDM profile + fixed map”的实现，而已经具备可迁移场景挖掘的主要架构骨架：

1. 主方法从 `meta_testing/` 独立迁移为 `mvr/`；
2. `pearl_learning` 与 `sac_scenario_mining` 已移动到 `archives/`；
3. Task 定义已经分离 `sut_split / geometry_split / functional_split`；
4. 已建立多 geometry catalog；
5. Taskbook 已显式记录 `geometry_hash`；
6. Outer 初始条件已改为 conflict-relative：
   - `adversary_distance_to_conflict_m`
   - `sut_distance_to_conflict_m`
7. 已新增 geometry-only `InteractionCandidate`；
8. 已新增 `InteractionEncoder`；
9. 已删除 family-specific neural policy heads；
10. 已新增 Universal MoE Outer Policy；
11. 已新增 learned MoE Router；
12. 已新增 PEARL-inspired Product-of-Gaussians context encoder；
13. 已新增 R1–R4 SUT/geometry OOD evaluation regimes；
14. concrete scenario manifest 已记录 geometry、conflict-relative initial state、latent、Inner policy hash、episode seed；
15. canonical pipeline 仍为：
   `inner_pretrain → posterior → inner_latent_calibration → outer`。

因此，**方法架构已经足以进入分阶段训练验证**。

但不建议立即执行完整四阶段训练。当前最合理的下一步是：

\[
\boxed{\text{先单独训练并验收 Inner transferable adversarial prior}}
\]

只有第一阶段通过，才进入 posterior/context training。

---

# 2. 第一阶段到底在训练什么？

第一阶段不是在训练完整 MVR，也不是在验证 few-shot adaptation。

第一阶段只回答一个基础问题：

> **一个共享的 Inner SAC 是否能够在不同 SUT、不同道路几何和不同 Functional Scenario 上，学习到具有迁移性的危险驾驶/对抗交互先验？**

形式化地，第一阶段训练：

\[
\pi_{\mathrm{inner}}
(a_t^{adv}
\mid
s_t,
h_{\mathrm{scene}},
x_0,
o,
z=0)
\]

其中：

- \(s_t\)：当前 SUT–adversary 的通用物理状态；
- \(h_{\mathrm{scene}}\)：geometry / interaction embedding；
- \(x_0\)：conflict-relative 初始条件；
- \(o\)：high-level adversarial option；
- \(z=0\)：PEARL posterior 的 unit prior。

第一阶段**不使用 learned SUT latent 进行适应**。

因此，第一阶段要验证的是：

\[
\boxed{\text{geometry-conditioned transferable adversarial control prior}}
\]

而不是 few-shot adaptation gain。

---

# 3. 第一阶段哪些网络被训练？

根据当前 `mvr/training/stages.py`：

## Trainable

```text
map_encoder
interaction_encoder
shared_feature_encoder
option_embedding
inner_sac
```

## Frozen

```text
episode_token_builder
context_encoder
outcome_decoder
universal_scene_policy / MoE Outer
```

特别是：

- PEARL Context Encoder 不应在第一阶段学习；
- MoE Router 不应在第一阶段学习；
- Outer PPO 不应在第一阶段学习。

这保证第一阶段的科学问题足够单纯：

> 是否能够先学到一个跨 task 的 adversarial driving prior？

---

# 4. 当前第一阶段训练集规模

当前 geometry catalog：

```text
Merge:      g01/g02/g03 train, g04 test
Cut-in:     g01/g02/g03 train, g04 test
Roundabout: g01/g02/g03 train, g04 test
```

SUT：

```text
Train:
    idm_cautious
    idm_defensive
    idm_normal
    idm_assertive

Validation:
    idm_fast_small_gap

Test:
    idm_late_response
```

因此 meta-train task 数：

\[
3\text{ families}
\times
3\text{ train geometries}
\times
4\text{ train SUTs}
=
\boxed{36\text{ tasks}}
\]

这 36 个 task 才是第一阶段 Inner pretrain 应真正覆盖的训练分布。

---

# 5. 当前默认配置不适合作为正式第一阶段预算

当前：

```yaml
inner:
  episode_budget: 20
```

但训练集有 36 tasks，同时 `MetaTaskSampler` 当前只是 `random.choice(tasks)`。

因此：

```text
20 episodes < 36 train tasks
```

甚至无法保证每个训练 task 被访问一次。

所以当前 `episode_budget=20` 只适合极小 smoke test，不适合作为正式 Inner pretraining。

---

# 6. 第一阶段正式训练前建议先完成的 5 个小修改

这些不是再次重构架构，而是让第一阶段实验具有科学可解释性。

## 6.1 必须修改 1：增加 validation geometry

当前 geometry 只有 `train/test`，虽然 `GeometrySpec` 已支持 `validation`，但 catalog 中没有真正 validation geometry。

如果训练期间用 `g04 test` 调参或判断是否继续训练，就会污染最终 unseen-geometry test set。

推荐每个 family 改为：

```text
g01 train
g02 train
g03 train
g04 validation
g05 test
```

训练 task 数仍保持 36。

第一阶段禁止使用 `geometry_split == test` 进行超参数选择、训练轮数选择或 checkpoint selection。

---

# 7. 必须修改 2：Task sampler 要保证均衡覆盖

当前 `random.choice(tasks)` 不能保证覆盖。

建议 `MetaTaskSampler` 增加：

```python
def shuffled_epoch(self):
    tasks = list(self.tasks)
    random.shuffle(tasks)
    return tasks
```

训练采用：

```text
每一轮：
    每个 train task 恰好出现一次
    只随机化 task 顺序
```

即：

\[
N_{episodes}=N_{tasks}\times N_{episodes/task}
\]

正式 pretraining 不要再用无覆盖保证的全局随机 `episode_budget`。

---

# 8. 必须修改 3：Inner pretrain 不应依赖随机初始化的 Outer MoE

当前 `train_inner()` 通过 `OnlineMetaTest.run()` 收集数据，而该流程会调用未训练的 `universal_scene_policy` 选择：

```text
candidate
x0
option
```

第一阶段 Outer 是冻结的随机初始化网络，因此训练数据分布会被一个未经训练的随机 neural policy 控制。

推荐第一阶段加入独立的：

```text
PretrainSceneSampler
```

或者让 `OnlineMetaTest.run()` 接受：

```python
scene_action_provider=
```

第一阶段 provider 使用：

```text
candidate: balanced cycle
option: balanced cycle
continuous x0: low-discrepancy / stratified sampling
```

目标是让 Inner 看到一个：

\[
\boxed{\text{广覆盖、可重复、与未训练 Outer 无关的场景分布}}
\]

---

# 9. 必须修改 4：pipeline 支持训练到 Stage 1 就停止

当前：

```bash
python -m mvr.scripts.train_mvr
```

会自动执行四个阶段，不适合 gate-based training。

建议加入：

```bash
--stop-after inner_pretrain
```

示例：

```powershell
python -m mvr.scripts.train_mvr ^
  --config mvr/configs/mvr_stage1.yaml ^
  --output results/mvr/stage1 ^
  --stop-after inner_pretrain
```

验收通过后再 `--resume inner_pretrain.pt` 进入 posterior。

---

# 10. 强烈建议修改 5：补充 Stage-1 metrics

当前 `train_inner()` 主要记录：

```text
simulator_episodes_consumed
transitions
optimizer_updates
last_loss
```

这不足以验收。

建议增加：

```text
task_episode_counts
family_episode_counts
geometry_episode_counts
sut_episode_counts

mean_inner_episode_return
mean_valid_rate
mean_failure_rate
mean_min_ttc
mean_min_distance
mean_max_closing_speed

actor_loss_mean / last
critic_loss_mean / last
alpha_loss_mean / last

action_abs_mean
action_saturation_rate
```

尤其不要只保存 `last_loss`。

---

# 11. 额外推荐：Interaction descriptor 归一化

当前 `InteractionCandidate.features()` 中：

```text
sin(angle)
cos(angle)
sut_distance_available_m
adversary_distance_available_m
sut_route_curvature
adversary_route_curvature
```

尺度差异较大。建议进入 `InteractionEncoder` 前统一 normalization，例如：

```text
distance / 100.0
curvature / π
```

或基于固定工程范围归一化。

这是低成本、值得在正式训练前完成的稳定性改进。

---

# 12. Geometry hash 再增加一个契约

当前 taskbook builder 已检查：

```text
train_hashes ∩ test_hashes = ∅
```

建议再加入：

```text
每个 family 的 train / validation / test geometry 必须具有预期数量的 unique hashes
```

同时建议 geometry hash payload 包含所有 model-visible static geometry 信息：

```text
lane points
lane index
lane width
speed limit
polyline type
```

保证模型看到的不同 geometry 不会被误记录成同一个 hash。

---

# 13. 第一阶段训练前 Pre-flight 验收

## 13.1 Code tests

```powershell
conda run -n metadrive python -m pytest mvr/tests -q
```

硬性要求：100% PASS。

## 13.2 Compile

```powershell
conda run -n metadrive python -m compileall -q mvr
```

要求 0 error。

## 13.3 重建 taskbook

```powershell
python -m mvr.scripts.build_taskbook ^
  --output mvr/configs/taskbook.json
```

验收：

```text
所有 geometry hash 可重建
train/validation/test hash 严格隔离
每个 family geometry 数量符合预期
```

---

# 14. Pre-flight G1：Geometry / Interaction Representation

验收目标：

1. HPTR map encoding 对 SE(2) 变换稳定；
2. Interaction encoder 不读取 candidate label；
3. 不读取 functional scenario ID；
4. 不读取 SUT ID；
5. scene/candidate embedding 全部 finite；
6. 不同 geometry 不应全部 collapse 为同一 scene embedding；
7. variable candidate count 可正确处理。

硬门槛：

```text
NaN/Inf = 0
```

---

# 15. Pre-flight G2：Simulator / Coordinate Contract

对 3 families × train/validation geometries 运行代表性 reset。

验收：

```text
runtime geometry hash == task geometry hash
SUT attachment success
adversary attachment success
candidate route valid
conflict zone finite
episode seed 正确记录
```

最关键：

requested：

```text
distance_to_conflict = d
```

实际生成：

```text
route distance to conflict ≈ d
```

建议容差：

```text
≤ 0.25 m
```

---

# 16. Pre-flight G3：SUT failure-landscape heterogeneity

执行：

```powershell
python -m mvr.scripts.validate_mvr ^
  --config mvr/configs/mvr.yaml ^
  --output results/validation/g3_pre_stage1.json
```

目标是证明不同 train SUT 在相同测试配置下存在不同 failure landscape。

保留当前 gate：

```text
valid_rate >= 0.80
failure_rate ∈ [0.02, 0.80]
mean_failure_disagreement >= 0.10
```

G3 不通过：禁止开始正式 Stage 1。

---

# 17. 第一阶段训练建议分两步

## Stage 1A：Coverage Smoke Run

目标：确认 36 个 train tasks 都能进入同一个 Inner training loop。

建议：

```text
episodes_per_task = 1
tasks = 36
```

共 36 simulator episodes。

可以临时：

```text
step_budget = 240  # a smoke rollout must still let the SUT complete its route
```

Smoke PASS：

```text
36/36 train tasks visited
3/3 families visited
9/9 train geometries visited
4/4 train SUTs visited
all candidates exercised
all options exercised
optimizer_updates > 0
all losses finite
no simulator contract error
no geometry hash mismatch
```

Smoke 不追求高 failure rate。

---

# 18. Stage 1B：正式 Pilot Pretraining

Smoke 全通过后：

```text
episodes_per_task = 5
```

规模：

\[
36\times5=\boxed{180\text{ simulator episodes}}
\]

保留：

```text
step_budget = 240
updates_per_episode = 8
batch_size = 64
```

理论最大 simulator steps：

\[
180\times240=43,200
\]

如果 180 episodes 后 validation 仍明显改善，再扩展：

```text
episodes_per_task = 10
```

即 360 episodes。

不要在没有 validation curve 时盲目扩大到上千 episode。

---

# 19. 第一阶段场景采样应保持 balanced

每个 task 的 episode 应覆盖：

## Candidate

各 candidate 尽量均衡。

## Option

```text
approach_conflict
yield_then_press
gap_close
```

全部覆盖。

## Continuous initial condition

对：

\[
[
d_{adv}^{C},
d_{sut}^{C},
v_{adv},
v_{sut}
]
\]

使用 deterministic low-discrepancy sequence。

目标是训练广泛场景空间下的通用对抗策略，而不是随机初始化 Outer 偏好的狭窄区域。

---

# 20. 第一阶段 checkpoint / artifacts

输出建议：

```text
results/mvr/stage1/
├── inner_pretrain.pt
├── inner_pretrain.json
├── manifest.json
├── coverage.json
└── validation.json
```

`inner_pretrain.json` 应记录：

```text
git commit
config hash
taskbook hash
checkpoint hash
stage seed
task coverage
optimizer updates
loss summary
training physical metrics
```

---

# 21. 第一阶段验收不能只看 training loss

SAC actor/critic/alpha loss 只能说明优化器是否工作，不能证明 adversarial policy 学会制造危险交互。

因此第一阶段必须有固定 validation protocol。

---

# 22. 第一阶段 Validation regime

推荐四个诊断 regime，都只评价 Inner policy，不运行 posterior adaptation。

## V1：Seen SUT + Seen Geometry

目的：判断基本训练能力。

## V2：Validation SUT + Seen Geometry

```text
idm_fast_small_gap + train geometry
```

判断 SUT-style transfer。

## V3：Seen SUT + Validation Geometry

判断 geometry transfer。

## V4：Validation SUT + Validation Geometry

这是第一阶段最重要的 validation regime：

\[
\boxed{\text{unseen validation SUT + unseen validation geometry}}
\]

它不会污染最终 R4，因为 final test 仍使用 `idm_late_response + test geometry`。

---

# 23. Validation 必须固定初始场景

为了单独评价 Inner SAC，所有 policy 必须共享完全相同：

```text
geometry
candidate
distance-to-conflict
initial speeds
option
episode seed
```

只改变：

\[
\pi_{adv}
\]

---

# 24. 第一阶段 Baseline

只需要两个：

## B0：Zero / No-op adversary

回答：初始条件本身有多危险？

## B1：Random continuous adversary

回答：学习到的 Inner 是否优于任意随机驾驶？

第一阶段暂时不需要 PEARL baseline、single-task SAC 或 MoE ablation。

---

# 25. Validation case 数

推荐每个 Functional Scenario：

```text
16 fixed initial-condition cases
```

由：

```text
low-discrepancy x0
+
balanced candidate
+
balanced option
```

产生。

V4：

\[
3\times16=48\text{ episodes/policy}
\]

比较 zero / random / trained Inner，总计约 144 episodes。

规模足够小。

---

# 26. 第一阶段核心验收指标

## A. Engineering correctness

硬门槛：

```text
pytest = PASS
compileall = PASS
runtime geometry hash mismatch = 0
NaN/Inf = 0
checkpoint can reload
```

## B. Training coverage

硬门槛：

```text
task coverage           = 100%
functional coverage     = 100%
train geometry coverage = 100%
train SUT coverage      = 100%
candidate coverage      = 100%
option coverage         = 100%
```

## C. Optimization health

要求：

```text
optimizer_updates > 0
actor_loss finite
critic_loss finite
alpha_loss finite
```

记录 action mean/std 与 saturation。

若 `|action| > 0.95` 的比例长期超过 95%，必须诊断 policy collapse。

## D. Physical validity

建议：

Hard stop：

```text
valid_rate < 0.80
```

则 Stage 1 FAIL。

Desired target：

```text
valid_rate >= 0.90
```

重点监控：

```text
non-target collision
adversary out-of-road
SUT out-of-road
wrong route
```

## E. Adversarial controllability

比较：

```text
Trained Inner
vs
Random
vs
Zero
```

固定 x0。

报告：

```text
valid critical rate
target collision rate
near-miss rate
median min TTC
median min distance
max closing speed
mean Inner episode return
invalid rate
```

---

# 27. 第一阶段 Hard PASS 标准

建议使用相对 baseline gate，而不是武断绝对 reward threshold。

必须同时满足：

### H1

整体 validation：

```text
Trained Inner 的有效危险发现能力 > Zero
```

### H2

整体 validation：

```text
Trained Inner 的有效危险发现能力 > Random
```

优先看：

```text
valid critical rate
```

若 failure 稀疏，则看：

```text
median min TTC 更低
且 invalid rate 不恶化
```

### H3

improvement 至少在：

```text
2 / 3 functional families
```

成立。

### H4

V4：

```text
validation SUT + validation geometry
```

上必须保持正 transfer gain。

### H5

V4：

```text
valid_rate >= 0.80
```

### H6

不能有任何 family 出现 policy 完全失控。

---

# 28. 推荐 Stage-1 Transfer Gain

定义：

\[
G_{inner}=S_{trained}-S_{random}
\]

其中 \(S\) 优先采用 valid critical rate；若事件稀疏，可采用连续 risk score。

分别报告：

```text
G_ID
G_SUT
G_GEO
G_JOINT
```

第一阶段最重要：

\[
\boxed{G_{JOINT}>0}
\]

暂时不需要统计显著性结论。

---

# 29. 不建议设置“failure rate 必须达到 X%”

第一阶段 Outer 尚未训练，x0 来自覆盖型 sampling，而不是危险初始状态搜索。

因此 failure rate 低不一定意味着 Inner 没学会。

更重要的是：

> 在同样 x0 上，learned Inner 是否比 random/no-op 更容易推动系统进入危险状态。

所以 Stage 1 看的是 relative controllability gain。

---

# 30. 第一阶段建议画的曲线

只需四类：

1. optimizer update vs critic loss；
2. training episode vs rolling Inner episode return；
3. checkpoint vs validation min TTC / valid critical rate；
4. family vs trained/random/zero performance。

不要第一阶段就做大量论文图。

---

# 31. 第一阶段不验证什么

Stage 1 不回答：

```text
K=0 vs K=1/2/4
```

因为 Context Encoder 尚未训练。

Stage 1 不回答：

```text
MoE 是否有效
```

因为 Outer 尚未训练。

Stage 1 不回答：

```text
R4 failure-discovery AUC 是否超过 baseline
```

因为完整 MVR 尚未形成。

Stage 1 也禁止使用 final test geometry / final test SUT 做 checkpoint selection。

---

# 32. Stage 1 成功意味着什么

Stage 1 PASS 只能支持：

> **The shared adversarial controller learned a geometry-conditioned control prior that transfers across the training task distribution and retains controllability on held-out validation SUT/geometry combinations.**

还不能支持 few-shot meta adaptation 或 MoE functional transfer。

---

# 33. Stage 1 失败的定位顺序

## A：所有 family 都没有优于 random

检查：

```text
Inner reward
SAC update
action scaling
replay
```

## B：seen geometry 有效，validation geometry 失效

检查：

```text
InteractionEncoder
MapEncoder
geometry descriptor scaling
distance-to-conflict coordinate
```

## C：seen SUT 有效，validation SUT 失效

说明 Inner overfit train SUT style，应优先增加 SUT behavior diversity。

## D：Merge 有效，Cut-in/Roundabout 无效

检查 shared physical state / interaction representation / reward semantics。

## E：危险度高但 invalid rate 高

说明在利用 simulator loophole，应优先检查 invalid penalty、spawn bounds、action smoothness、route constraints。

---

# 34. 第一阶段训练决策树

```text
pytest / compile
      │
      ▼
Task/Geometry contracts
      │
      ▼
G3 PASS?
 ┌────┴────┐
NO        YES
│          │
STOP    36-episode smoke
             │
             ▼
         smoke PASS?
        ┌────┴────┐
       NO        YES
       │          │
      FIX    180-episode pilot
                  │
                  ▼
           Stage-1 validation
                  │
                  ▼
       V4 transfer gain > 0
       + valid_rate >= 0.80?
             ┌────┴────┐
            NO        YES
            │          │
           STOP    STAGE_1_PASS
                         │
                         ▼
                   train posterior
```

---

# 35. 推荐第一阶段配置

建议建立：

```text
mvr/configs/mvr_stage1.yaml
```

概念配置：

```yaml
training:
  device: cuda
  step_budget: 240
  family_filter: all

inner:
  algorithm: option_conditioned_sac
  sampling: balanced_low_discrepancy
  episodes_per_task: 5
  updates_per_episode: 8
  batch_size: 64
  replay_capacity: 100000
  learning_rate: 0.0003
```

正式确定后建议只保留语义清楚的：

```text
episodes_per_task
```

不要长期同时维护 `episode_budget` 与 `episodes_per_task`。

---

# 36. 推荐执行命令

完成上述小修改后：

## Tests

```powershell
conda run -n metadrive python -m pytest mvr/tests -q
```

## Compile

```powershell
conda run -n metadrive python -m compileall -q mvr
```

## Build taskbook

```powershell
python -m mvr.scripts.build_taskbook ^
  --output mvr/configs/taskbook.json
```

## G3

```powershell
python -m mvr.scripts.validate_mvr ^
  --config mvr/configs/mvr_stage1.yaml ^
  --output results/validation/g3_stage1.json
```

## Smoke

```powershell
python -m mvr.scripts.train_mvr ^
  --config mvr/configs/mvr_stage1_smoke.yaml ^
  --output results/mvr/stage1_smoke_seed11 ^
  --stop-after inner_pretrain
```

## Pilot

```powershell
python -m mvr.scripts.train_mvr ^
  --config mvr/configs/mvr_stage1.yaml ^
  --output results/mvr/stage1 ^
  --stop-after inner_pretrain
```

## Validate Inner

建议扩展现有 `validate_mvr.py`：

```powershell
python -m mvr.scripts.validate_mvr ^
  --config mvr/configs/mvr_stage1.yaml ^
  --checkpoint results/mvr/stage1/inner_pretrain.pt ^
  --mode inner ^
  --output results/mvr/stage1/validation.json
```

---

# 37. 第一阶段最终验收表

| 类别 | 项目 | Hard Gate |
|---|---|---|
| Code | `pytest mvr/tests` | PASS |
| Code | `compileall mvr` | PASS |
| Data | train/test geometry hash | disjoint |
| Data | validation geometry | 独立存在 |
| Data | 每个 train family unique geometry hash | 达到预期数量 |
| Runtime | geometry hash mismatch | 0 |
| Runtime | conflict-relative spawn error | ≤ 0.25 m |
| Coverage | 36 train tasks | 100% |
| Coverage | 3 functional families | 100% |
| Coverage | candidate / option | 100% |
| Optimization | SAC actor/critic/alpha | finite |
| Optimization | optimizer updates | > 0 |
| Validity | V4 valid rate | ≥ 0.80 |
| Control | trained > random overall | YES |
| Control | trained > zero overall | YES |
| Transfer | positive gain in ≥2/3 families | YES |
| Transfer | V4 joint validation gain | > 0 |
| Provenance | checkpoint/config/taskbook hash | 完整 |
| Replay | checkpoint reload/reproduce | PASS |

全部通过后标记：

```text
STAGE_1_PASS
```

---

# 38. 第一阶段通过后才能进入第二阶段

Stage 1 PASS 后才进入：

```text
posterior
```

第二阶段的问题才是：

> 给定已经具备对抗控制能力的 Inner policy，PEARL-style context 是否能从 K 个测试 episode 中推断出对后续搜索有用的 latent task/SUT vulnerability？

第二阶段才验收：

```text
posterior identifiability
correct-z vs swapped-z
K-shot context utility
```

不要在 Stage 1 未通过时提前训练 Context Encoder。

---

# 39. 对当前最新代码的最终判断

与上一版相比，这次重构已经跨过最重要的架构门槛：

```text
fixed-map/family-head MVR
        ↓
multi-geometry
interaction-centric
universal MoE
PEARL-inspired context
R1–R4 task design
```

因此现在不需要再次大规模重构方法。

第一阶段之前真正需要处理的是训练实验工程：

1. 增加 validation geometry；
2. balanced task coverage；
3. Inner pretrain 使用独立覆盖型 initial-condition sampler；
4. pipeline 支持 stop-after-stage；
5. 补充 Stage-1 validation metrics；
6. 归一化 interaction descriptors。

完成这些后，就应该停止继续改网络，开始：

\[
\boxed{\text{Inner transferable prior 的训练与验收}}
\]

第一阶段最核心的成功标准只有一句话：

> **在完全固定相同的初始场景条件下，训练后的共享 Inner SAC 在未参与训练的 validation SUT + validation geometry 上，仍能比 random/no-op adversary 更稳定地诱导高风险且有效的交通交互。**

只要这一点成立，才值得继续训练 PEARL posterior。
