# MVR 当前实现与论文创新一致性审计及 Stage2 运行目标

## 0. 文档定位

- 项目：`SafeDL/META_LEARNING`
- 审计代码提交：`8c30520fd49e8885ed31492ef52a0ad7510354a0`
- 论文设计依据：`研究动机与总体技术框架.docx`
- 当前阶段：Stage1 `inner_pretrain` 已完成
- 下一阶段：Stage2 `posterior`
- 最低论文目标：**unseen SUT + unseen road geometry + few-shot transferable scenario mining**
- unseen Functional Scenario 属于后续理想扩展，不是当前 Stage2 的硬条件。

---

# 1. 总体判断

当前 `mvr/` 的主体架构已经与论文核心创新设计高度一致，不需要重新设计总体网络。

当前最准确的状态是：

```text
论文总体方法骨架            已实现
Stage1 transferable Inner   已正式通过
PEARL posterior 架构         已实现骨架，尚待 Stage2 正式训练/验收
MoE Outer 架构              已实现，尚未训练
最终 few-shot search gain    尚未验证
```

因此：

\[
\boxed{\text{Stage1 PASS} \Rightarrow \text{允许进入 Stage2}}
\]

但是**不建议直接按当前 `posterior.episode_budget=20` 运行现有 `train_posterior()`**。当前 posterior 流程仍是阶段骨架，存在几个会影响科学解释性的局部问题，应先修正后再做 formal Stage2。

---

# 2. 创新 1：可迁移的层次化场景搜索

论文将危险具体场景定义为：

\[
C=(M,x_0,\pi_{adv},\xi)
\]

并将搜索器拆为：

\[
\text{Outer}: (h_{scene},z)\rightarrow x_0
\]

\[
\text{Inner}: (s_t,h_{scene},z,x_0,o)\rightarrow a_t^{adv}
\]

当前代码已经具备：

- `UniversalScenePolicy`：episode-level Outer；
- `OptionConditionedSAC`：step-level Inner；
- 统一 conflict-relative continuous space；
- shared Inner SAC 跨 Merge / Cut-in / Roundabout 共用；
- `PhysicalStateExtractor` 使用统一物理状态，不读取 SUT identity；
- candidate / option 与 family-specific neural head 解耦。

### 判断

\[
\boxed{\text{符合}}
\]

Stage1 已证明其中的基础部分：shared Inner adversarial controller 能从多 SUT、多 geometry、多 functional scenario 训练任务中学习 transferable control prior。

Outer 尚未训练，所以目前只能证明层次化框架的 Inner 半部分，而不能声称完整 Outer+Inner 搜索器已经验证有效。

---

# 3. Interaction-centric representation

论文要求不同 Functional Scenario 被统一到“路线—冲突区域—相对状态”坐标系，而不是 family-specific 原始字段。

当前实现已经具有：

- HPTR-style map encoder；
- route-pair `InteractionCandidate`；
- conflict zone；
- crossing angle / route curvature；
- normalized geometry descriptor；
- route-relative progress；
- conflict timing；
- 统一四维初始条件：
  - adversary distance to conflict；
  - SUT distance to conflict；
  - adversary initial speed；
  - SUT initial speed；
- model 不直接输入 SUT identity；
- Outer 不直接输入 Functional Scenario identity。

因此当前结构已满足论文最低 R4 泛化目标的表示基础。

---

# 4. 创新 2：场景结构条件化的 PEARL + MoE

## 4.1 PEARL

当前已经实现：

```text
EpisodeTokenBuilder
    ↓
PearlContextEncoder
    ↓
Product-of-Gaussians q(z|C)
    ↓
VulnerabilityOutcomeDecoder
```

Evidence token 当前包括：

```text
scene embedding
continuous x0
option
full interaction trajectory
outcome
```

这与论文把 \(z\) 定义为“task vulnerability / failure-landscape representation”的方向一致。

### Stage2 前建议做一个小修正

当前 token 使用 global scene embedding，但没有显式表示本 episode 实际选择的 interaction candidate。

建议 `OnlineEpisode` 增加：

```text
selected_interaction_embedding
```

取：

```python
encoding.candidate_embeddings[scene.candidate_index]
```

然后在 `EpisodeTokenBuilder` 与 held-out target decoder 中使用该 selected interaction embedding。

这样：

- 不增加 SUT / family label；
- 不改变 128-D scene input 维度；
- 不修改 Stage1 Inner；
- 不需要重训 Stage1；
- 更符合论文强调的 interaction-centric evidence。

---

## 4.2 MoE

当前实现：

```text
TaskAwareMoERouter(h_scene, z)
→ expert probabilities
→ sample/select one expert
→ InitialConditionExpert
```

优点：

- router 同时读取 scene representation 与 \(z\)；
- expert 不硬绑定 Functional Scenario；
- candidate / option 是 universal heads；
- experts 主要建模 continuous initial-condition proposal。

这与论文 MoE 主体思想一致。

### 一个后续需处理的差异

论文草稿写的是 strict soft mixture：

\[
w=softmax(g(h,z)),\qquad \pi=\sum_i w_i\pi_i
\]

当前代码则是 probabilistic hard routing：根据 categorical router 选择一个 expert。

这不影响 Stage2，因为 Stage2 完全冻结 Outer。

但在 Outer formal training 前必须二选一：

1. 实现真正 mixture distribution；或
2. 将论文措辞收敛为 `task-aware probabilistic expert routing`。

当前不建议为 Stage2 修改 MoE。

---

# 5. 创新 3：critical / valid / diverse dangerous scenario mining

当前代码已经实现：

## Criticality

- target collision；
- valid critical near miss；
- TTC；
- minimum distance；
- closing speed；
- severity vector。

## Validity

invalid episode 包括：

```text
non-target collision
adversary out of road
SUT out of road
wrong route
```

并且 invalid episode 不能记为有效 failure；Inner/Outer reward 均有 invalid penalty。

## Diversity

已经有：

```text
FailureSignature
NoveltyTracker
```

Outer reward 对 novel failure 提供额外奖励。

## 仍未完全闭合

强版本论文表述还包括：

```text
RSS / explicit responsibility attribution
explicit action smoothness / plausibility penalty
```

当前尚未完整实现。因此目前代码可以支持：

> criticality + validity + novelty-aware scenario mining

但还不宜声称已经完整实现：

> RSS-responsibility-aware multi-objective mining

这一点不阻塞 Stage2，应该在最终 Outer / formal search 前闭合或收敛论文措辞。

---

# 6. R4 与 R5/R6 的范围

论文将：

```text
R4 = unseen SUT + unseen geometry + seen Functional Scenario
```

定义为最低核心贡献。

当前 split：

```text
Geometry:
  train      g01/g02/g03
  validation g04
  test       g05

SUT:
  train       cautious / defensive / normal / assertive
  validation  fast_small_gap
  test        late_response
```

已经支持 R1–R4。

R5/R6 的 unseen Functional Scenario 是理想扩展，不需要为了 Stage2 提前加入 leave-one-family-out。

---

# 7. Stage1 formal 结果

Formal Stage1：

```text
36 train tasks
× 5 episodes/task
= 180 simulator episodes
```

实际：

```text
simulator_episodes_consumed = 180
balanced_sampling_epochs    = 5
transitions                 = 21,508
optimizer_updates           = 1,432
requested_optimizer_updates = 1,440
warmup_skipped_updates      = 8
```

覆盖：

```text
36/36 train tasks
Merge      60 episodes
Cut-in     60 episodes
Roundabout 60 episodes

4 train SUTs:
45 episodes each

4 options:
45 episodes each
```

训练：

```text
valid rate              = 1.0
failure rate            = 0.6167
action saturation rate  = 0.6086
```

action saturation 低于原 Stage1 collapse diagnostic 的 0.95，最终 losses 均 finite。

---

# 8. Stage1 V4 formal 验收

V4：

```text
validation SUT = idm_fast_small_gap
validation geometry = g04
3 families × 16 cases = 48 episodes/policy
```

## Overall

| Policy | Valid Critical Rate | Target Collision Rate | Valid Rate |
|---|---:|---:|---:|
| Zero | 0.5208 | 0.0000 | 1.0 |
| Random | 0.4792 | 0.1458 | 1.0 |
| **Trained Inner** | **0.6875** | **0.5000** | **1.0** |

所以：

\[
G_{random}=0.6875-0.4792=\boxed{+0.2083}
\]

\[
G_{zero}=0.6875-0.5208=\boxed{+0.1667}
\]

即：

```text
+20.83 percentage points over random
+16.67 percentage points over zero
```

trained Inner median min TTC：

```text
1.713 s
```

低于：

```text
random = 5.488 s
zero   = 3.959 s
```

## Family-level

| Family | Zero | Random | Trained |
|---|---:|---:|---:|
| Cut-in | 0.0625 | 0.1875 | **0.4375** |
| Merge | 0.8750 | 0.8125 | **0.9375** |
| Roundabout | 0.6250 | 0.4375 | **0.6875** |

即：

```text
trained > random : 3/3 families
trained > zero   : 3/3 families
```

之前 smoke 中 Roundabout 的负迁移已经在 formal training 后消失。

---

# 9. Stage1 最终状态

原 hard gates：

```text
formal 180 episodes             PASS
task coverage                   PASS
loss finite                     PASS
valid rate >= 0.80              PASS
trained > random overall        PASS
trained > zero overall          PASS
positive gain >= 2/3 families   PASS (actual 3/3)
V4 joint gain > 0               PASS
```

因此：

\[
\boxed{\texttt{STAGE\_1\_PASS}}
\]

Stage1 已足以支持：

> The shared adversarial controller learns a geometry-conditioned transferable control prior and retains adversarial controllability on held-out validation SUT/geometry combinations.

但仍不能支持：

```text
few-shot adaptation improves search
posterior z is useful
MoE improves cross-task mining
```

这些属于 Stage2–Stage4。

---

# 10. 一个不阻塞 Stage2、但应修复的 provenance 问题

formal artifact 中记录：

```text
git_commit = 45050d16...
```

但产生 balanced schedule、scene deduplication 等 formal 行为的代码最终提交在：

```text
8c30520f...
```

这说明 formal run 很可能发生在：

```text
HEAD = 45050d16
+ uncommitted working-tree changes
```

这不构成科学结果失败，但最终复现链不够严格。

从 Stage2 开始建议 formal run 前强制：

```powershell
git status --porcelain
```

为空，并记录：

```text
git_commit
git_dirty
model_contract_hash
stage_config_hash
taskbook_hash
source_diff_hash (only if dirty)
```

正式 checkpoint 原则上禁止由 dirty worktree 产生。

---

# 11. 当前 posterior 代码为什么不能直接作为 formal Stage2

## 11.1 Budget 不足

当前：

```yaml
posterior:
  episode_budget: 20
  support_episodes: [1, 2, 4]
```

而 meta-train 有 36 tasks。

20 episodes 连每个训练 task 一次都覆盖不了，只适合作为 smoke。

---

## 11.2 Task / K sampling 不保证 coverage

当前 posterior 使用随机 task 与随机 support K，因此：

```text
task coverage
SUT coverage
geometry coverage
K coverage
```

都没有 formal guarantee。

Stage2 必须改为 balanced task/K sampling。

---

## 11.3 当前 posterior evidence 会由未训练 Outer 控制

当前 `train_posterior()` 调用 `OnlineMetaTest.run()` 时没有传 `scene_action_provider`。

因此场景初始条件来自尚未训练的：

```text
UniversalScenePolicy
```

这会让 Stage2 evidence distribution 被随机冻结 Outer 控制。

正确流程应为：

\[
\boxed{
\text{Stage1 frozen Inner}
+
\text{balanced deterministic scene probes}
\rightarrow
\text{posterior evidence}
}
\]

而不是：

\[
\text{random frozen Outer}
\rightarrow
\text{posterior evidence}
\]

这是 Stage2 前最重要的修正。

---

## 11.4 Full config hash 过度绑定 future-stage hyperparameters

当前 checkpoint hash 同时绑定：

```text
inner
posterior
inner_latent_calibration
outer
training
```

所以一旦把 posterior budget 改成 formal 参数，Stage1 checkpoint 可能因为 config hash mismatch 无法 resume。

建议拆成：

```text
model_contract_hash
stage_config_hash
taskbook_hash
```

`model_contract_hash` 只描述跨阶段必须兼容的模型和数据语义；`stage_config_hash` 记录当前 stage 训练超参数。

对已有 Stage1 checkpoint，可以从其 `provenance_config` 重新计算 contract hash，不需要重训 Stage1。

---

# 12. Stage2 唯一核心科学问题

Stage2 只回答：

> 给定已经具备 transferable adversarial control prior 的 Stage1 Inner，PEARL-style posterior 能否仅利用 K 个 support episodes，在不输入 SUT identity 的条件下，推断出对当前 task failure landscape 有用的 vulnerability latent \(z\)？

定义：

\[
\mathcal C_K=\{e_1,\ldots,e_K\}
\]

\[
z\sim q_\phi(z|\mathcal C_K)
\]

目标是让 held-out target outcome：

\[
p_\psi(
y_{target}
|
h_{interaction,target},
x_{0,target},
o_{target},
z
)
\]

使用正确 context 时优于：

```text
z = 0
swapped z
```

Stage2 不要求 Outer 搜索性能提高。

---

# 13. Stage2 的冻结/训练边界

## Frozen

必须冻结：

```text
map_encoder
interaction_encoder
shared_feature_encoder
option_embedding
inner_sac
universal_scene_policy
```

Stage2 结束后要求 Stage1-owned parameter hash：

```text
before == after
```

## Trainable

仅训练：

```text
episode_token_builder
context_encoder
outcome_decoder
```

当前 `TrainingStage.POSTERIOR` 的 ownership 是正确的。

---

# 14. Stage2 前最小代码修改

## P0-1 selected interaction evidence

`OnlineEpisode` 增加：

```text
selected_interaction_embedding
```

取：

```python
encoding.candidate_embeddings[scene.candidate_index]
```

posterior evidence 与 target decoder 使用它。

---

## P0-2 独立 PosteriorProbeSampler

Stage2 禁止未训练 Outer 参与 simulator evidence collection。

Sampler 输出：

```text
candidate
option
conflict-relative x0
episode seed
probe_id
```

并要求：

```text
balanced candidate
balanced option
low-discrepancy x0
reproducible
```

---

## P0-3 aligned probes

同一个 geometry 下，不同 SUT 使用完全相同：

```text
probe_id
candidate
x0
option
episode seed
```

只改变 SUT。

这样 task latent 的差异更容易解释为 SUT response / vulnerability，而不是初始化条件差异。

---

## P0-4 stage-aware checkpoint compatibility

增加：

```text
model_contract_hash
stage_config_hash
taskbook_hash
```

允许已有 Stage1 checkpoint 在保持模型/任务语义一致的前提下使用新的 Stage2 超参数。

---

## P0-5 balanced posterior meta-batch

不要继续：

```text
one random task
one random K
batch size = 1 task group
```

支持：

```text
K ∈ {1,2,4}
```

近似均衡采样，padding 到 K=4：

```text
support_tokens [B,4,D]
support_mask   [B,4]
```

---

# 15. Stage2 建议改成“两步式”

\[
\boxed{
\text{Simulator Evidence Collection}
\rightarrow
\text{Offline Posterior Training}
}
\]

不要把 MetaDrive rollout 与 posterior gradient update 绑在同一循环。

原因：

- Stage1 Inner 在 Stage2 frozen；
- map/interaction encoder frozen；
- 同一 episode 可反复重组 support/target；
- 大幅减少 MetaDrive 成本；
- fixed validation 更容易；
- support-target leakage 更容易审计。

---

# 16. Posterior Episode Bank

每条 episode 建议保存：

```text
metadata:
  task_id
  SUT split
  geometry split
  functional scenario
  geometry hash
  probe_id

model-visible evidence:
  selected_interaction_embedding
  continuous x0
  option
  trajectory [T,12]
  outcome [5]

provenance:
  Stage1 checkpoint hash
  Inner policy hash
  episode seed
  concrete scenario
```

其中：

```text
SUT ID
family ID
candidate label
```

只作为 metadata，不进入网络。

---

# 17. Stage2 formal simulator budget

建议默认：

```text
8 probes/task
```

## Meta-train bank

36 tasks：

\[
36\times8=\boxed{288}
\]

## Validation P2

```text
validation SUT × 9 train geometries
= 9 tasks
```

\[
9\times8=72
\]

## Validation P3

```text
4 train SUT × 3 validation geometries
= 12 tasks
```

\[
12\times8=96
\]

## Validation P4

```text
validation SUT × 3 validation geometries
= 3 tasks
```

\[
3\times8=24
\]

总 validation bank：

\[
72+96+24=\boxed{192}
\]

Stage2 一次性 simulator evidence 总量：

\[
288+192=\boxed{480}
\]

后续 posterior epoch 均 offline，不再重复 MetaDrive。

可先用 30–60 episodes 做 data-pipeline smoke；formal 再按上述预算执行。

---

# 18. Stage2 严禁使用 final test split

Stage2 禁止使用：

```text
idm_late_response
g05 test geometries
```

包括：

```text
training
early stopping
hyperparameter tuning
checkpoint selection
threshold selection
```

最终 test R4 必须保持 untouched。

---

# 19. Support/target protocol

每个 task：

```text
K ∈ {1,2,4}
target ∉ support
```

必须：

\[
support\_ids\cap target\_id=\varnothing
\]

K=0 定义为 prior：

\[
z_0=0
\]

训练时随机重组 support-target；validation 使用固定组合。

---

# 20. Training objective

保留当前 outcome：

```text
failure
invalid
normalized min TTC
normalized min distance
normalized max closing speed
```

训练：

\[
L_{posterior}
=
L_{binary}
+
L_{severity}
+
\beta D_{KL}
\]

初始继续使用：

\[
\beta=10^{-3}
\]

不要在 Stage2 一开始增加复杂 loss。

---

# 21. Formal 配置建议

概念配置：

```yaml
posterior_data:
  sampler: aligned_balanced_low_discrepancy
  train_episodes_per_task: 8
  validation_episodes_per_task: 8
  deterministic_inner: true
  use_outer_policy: false
  posterior_support_limit: 0

posterior:
  support_episodes: [1, 2, 4]
  batch_size: 32
  learning_rate: 0.0003
  max_epochs: 80
  validation_interval_epochs: 2
  early_stop_patience: 10
  kl_weight: 0.001
```

这里的 epoch 是 offline bank epoch，不重新运行 simulator。

---

# 22. 为什么 formal bank 建议 deterministic Stage1 Inner

Stage2 的问题是 task inference，而不是 behavior-policy stochasticity。

建议收集 posterior bank 时：

```text
Stage1 Inner deterministic=True
```

这样相同 probe 在不同 SUT 上的差异主要来自：

```text
SUT response
geometry × SUT coupling
```

而不是 SAC exploration noise。

最终 meta-test 再恢复正常 stochastic protocol。

---

# 23. Stage2 validation：correct / zero / swapped

## Correct

\[
z_{correct}=q_\phi(\mathcal C_K^{\mathcal T})
\]

## Zero

\[
z_{zero}=0
\]

回答 support 是否提供有效信息。

## Swapped

优先交换：

```text
same geometry
same Functional Scenario
different SUT
```

得到：

\[
z_{swap}=q_\phi(\mathcal C_K^{\mathcal T'})
\]

这样固定 geometry，只交换 SUT behavior evidence。

回答 posterior 是否学到 task-specific vulnerability，而不是通用常量。

---

# 24. Stage2 主指标

定义 predictive loss，不含 KL：

\[
L_{pred}
=
BCE_{failure}
+
BCE_{invalid}
+
Huber_{severity}
\]

## Correct-vs-zero

\[
\Delta_0
=
L_{zero}-L_{correct}
\]

要求：

\[
\boxed{\Delta_0>0}
\]

## Correct-vs-swapped

\[
\Delta_{swap}
=
L_{swapped}-L_{correct}
\]

要求：

\[
\boxed{\Delta_{swap}>0}
\]

## K-shot gain

\[
G_K=L_{K=0}-L_K
\]

最重要：

\[
\boxed{G_4>0}
\]

需要画：

```text
K=0 / 1 / 2 / 4
```

held-out predictive loss curve。

不必强制每一步严格单调，但总体应该体现更多有效 support 带来更好的 task prediction。

---

# 25. Posterior identifiability

同一 task 两个不重叠 support set：

\[
d_{within}
=
E\|\mu_A-\mu_B\|_2
\]

相同 geometry、不同 SUT：

\[
d_{between}
=
E\|\mu_U-\mu_{U'}\|_2
\]

最低要求：

\[
\boxed{d_{between}>d_{within}}
\]

不要把 SUT classification accuracy 作为论文主指标。\(z\) 应表示 vulnerability / failure landscape，而不是 SUT ID。

---

# 26. P4 是 Stage2 最重要 validation

P4：

\[
\boxed{
U_{validation}+M_{validation}
}
\]

Functional Scenario 已见。

Stage2 必须单独报告：

```text
P4 K=0
P4 K=1
P4 K=2
P4 K=4

correct-z
zero-z
swapped-z
```

---

# 27. Stage2 Hard PASS

## S1 Engineering

```text
pytest                 PASS
compileall              PASS
NaN/Inf                 0
checkpoint reload       PASS
support-target leakage  0
```

## S2 Stage ownership

```text
Stage1-owned parameter hash before == after
```

## S3 Data coverage

```text
36/36 train tasks
3/3 families
9/9 train geometries
4/4 train SUTs
candidate coverage 100%
option coverage 100%
K={1,2,4} all covered
```

## S4 Overall posterior utility

在 P2+P3+P4：

\[
L_{correct,K=4}<L_{zero}
\]

且：

\[
L_{correct,K=4}<L_{swapped}
\]

## S5 P4 joint-OOD utility

\[
\boxed{
L_{correct,K=4}
<
\min(L_{zero},L_{swapped})
}
\]

这是 Stage2 最关键 gate。

## S6 Family robustness

P4 至少：

```text
2/3 families positive correct-z gain
```

理想：

```text
3/3
```

## S7 K-shot usefulness

P4：

\[
L_{K=4}<L_{K=0}
\]

并且 overall K curve 呈改善趋势。

## S8 Identifiability

\[
d_{between}>d_{within}
\]

## S9 Statistical robustness

利用相同 target episode 的 paired baseline 做 paired bootstrap。

对：

\[
\Delta_0,\quad \Delta_{swap}
\]

报告 95% CI。

推荐 formal PASS：

```text
overall validation CI lower bound > 0
P4 point estimate > 0
```

若 P4 样本充足，最好 P4 CI lower bound 也 > 0。

---

# 28. 不应作为 Stage2 PASS 的指标

不要仅凭：

```text
posterior train loss 很低
latent cluster 很漂亮
SUT classification accuracy 很高
KL 很大
```

判定成功。

真正需要证明的是：

\[
\boxed{
\text{few-shot task evidence}
\Rightarrow
\text{better held-out failure-landscape prediction}
}
\]

---

# 29. Checkpoint selection

不要按 train loss 选。

建议：

```text
selection metric
=
mean correct-z predictive loss
over P2 + P3 + P4
```

固定 validation bank。

P4 单独保留 hard gate。

建议：

```text
validation every 2 epochs
early stopping patience = 10 validations
```

---

# 30. 建议新增测试

```text
test_posterior_selected_interaction_input.py
test_posterior_support_target_disjoint.py
test_posterior_support_permutation.py
test_posterior_balanced_task_sampler.py
test_posterior_k_mask.py
test_posterior_swapped_context.py
test_posterior_episode_bank_replay.py
test_stage_aware_checkpoint_hash.py
test_stage2_freezes_stage1_components.py
test_no_sut_or_family_identity_in_posterior_input.py
```

---

# 31. Stage2 artifacts

```text
results/mvr/stage2_seed11/
├── stage1_source.json
├── posterior_bank_train_manifest.json
├── posterior_bank_validation_manifest.json
├── posterior.pt
├── posterior.json
├── validation.json
├── latent_diagnostics.json
├── stage2_gate.json
└── manifest.json
```

大的 episode bank 可以本地保存，不一定提交 GitHub。

GitHub 保留：

```text
config
manifest
metrics
gate
checkpoint hash
```

即可。

---

# 32. 推荐执行流程

## Step 0 clean tree

```powershell
git status --porcelain
```

formal run 必须为空。

## Step 1 tests

```powershell
conda run -n metadrive python -m pytest mvr/tests -q
```

## Step 2 compile

```powershell
conda run -n metadrive python -m compileall -q mvr
```

## Step 3 verify Stage1 source

要求：

```text
stage = inner_pretrain
180 simulator episodes
taskbook hash match
checkpoint hash match
Stage1 gate PASS
```

## Step 4 Stage2 data smoke

建议 30–60 simulator episodes。

检查：

```text
episode bank save/reload
selected interaction embedding finite
support-target leakage = 0
posterior loss finite
correct/zero/swapped validation executable
```

## Step 5 collect formal bank

```text
train      288 episodes
validation 192 episodes
```

## Step 6 offline posterior training

概念命令：

```powershell
python -m mvr.scripts.train_mvr ^
  --config mvr/configs/mvr_stage2.yaml ^
  --resume results/mvr/stage1_seed11/inner_pretrain.pt ^
  --output results/mvr/stage2_seed11 ^
  --stop-after posterior
```

运行前 pipeline 应完成 stage-aware compatibility 和 offline bank 接口。

## Step 7 posterior validation

建议扩展：

```powershell
python -m mvr.scripts.validate_mvr ^
  --config mvr/configs/mvr_stage2.yaml ^
  --checkpoint results/mvr/stage2_seed11/posterior.pt ^
  --mode posterior ^
  --output results/mvr/stage2_seed11/validation.json
```

输出：

```text
P2/P3/P4
K=0/1/2/4
correct / zero / swapped
latent distance
paired bootstrap CI
```

---

# 33. Stage2 决策树

```text
Stage1 PASS
    │
    ▼
Fix posterior data contract
    │
    ├── selected interaction evidence
    ├── aligned probe sampler
    ├── stage-aware checkpoint hash
    └── offline episode bank
    │
    ▼
Stage2 smoke
    │
    ▼
Formal posterior bank
    │
    ▼
Offline posterior training
    │
    ▼
P2/P3/P4 validation
    │
    ▼
correct-z > zero/swapped ?
    ├── NO → diagnose posterior
    └── YES
          │
          ▼
P4 K=4 gain > 0 ?
          ├── NO → STOP
          └── YES
                │
                ▼
between-task > within-task ?
                ├── NO → diagnose latent collapse
                └── YES
                      │
                      ▼
                 STAGE_2_PASS
```

---

# 34. Stage2 失败定位

## correct ≈ zero

优先检查：

```text
selected interaction embedding
trajectory encoder
outcome evidence
KL weight
support diversity
```

不要先扩大 latent dimension。

## correct ≈ swapped

优先检查：

```text
aligned probe protocol
SUT response heterogeneity
support/target construction
```

## P1/P2 有效但 P3/P4 失败

说明 geometry shift 下 posterior 不稳。

优先检查：

```text
interaction representation
selected candidate representation
map / route scaling
```

不要增加 family label。

## K=1 好、K=4 变差

检查 Product-of-Gaussians：

```text
factor logvar calibration
overconfident evidence factors
duplicate/correlated support
```

只有真实出现后再考虑 precision clipping / temperature。

## latent 可分但 prediction 不提高

说明 latent 不是 task-useful representation。

不要把 latent visualization 当成功，回到 held-out outcome objective。

---

# 35. Stage2 PASS 后能支持的论文结论

Stage2 PASS 后可以支持：

> A PEARL-style product-of-Gaussians posterior can infer a task-dependent vulnerability representation from a few support interactions, and the inferred context improves held-out failure-landscape prediction under unseen SUT and unseen road geometry.

但仍不能支持：

> few-shot adaptation improves scenario-search efficiency.

因为：

```text
Inner 尚未针对 non-zero z calibration
Outer 尚未训练
MoE 尚未训练
```

---

# 36. Stage2 后的路线

Stage2 PASS 后进入：

```text
Stage3 = inner_latent_calibration
```

Stage3 回答：

> Stage1 Inner 原本主要在 z=0 prior 下训练；Stage2 得到有意义的 non-zero posterior z 后，Inner 能否稳定利用 z，而不破坏 Stage1 transferable adversarial controllability？

随后：

```text
Stage4 = MoE Outer PPO
```

最后在正式 R1–R4 fixed-budget meta-test 中比较：

```text
K=0 / K=1 / K=2 / K=4
```

真正证明：

\[
\boxed{
\text{few-shot scenario-mining efficiency gain}
}
\]

---

# 37. 最终一致性判断

## 创新 1：可迁移层次化场景搜索

```text
Architecture               YES
Stage1 Inner evidence      YES
Full Outer+Inner evidence  pending
```

## 创新 2：interaction-conditioned PEARL + MoE

```text
Interaction encoder        YES
PEARL architecture         YES
Task-aware router          YES
Posterior evidence         Stage2 pending
Strict soft MoE            pending before Outer formal
```

## 创新 3：critical / valid / diverse mining

```text
Criticality                YES
Validity                   YES
Novelty/diversity          YES
Explicit RSS responsibility pending
Explicit smoothness         pending
```

所以当前最合理的研究决策是：

\[
\boxed{
\text{保持总体架构不变}
\rightarrow
\text{正式完成 Stage2 posterior evidence}
}
\]

---

# 38. Stage2 一句话目标

\[
\boxed{
\text{用固定 Stage1 transferable Inner 产生少量、对齐、可复现的交互证据，}
}
\]

\[
\boxed{
\text{训练 PEARL posterior，使正确 few-shot context 在 unseen SUT + unseen geometry 上}
}
\]

\[
\boxed{
\text{比 zero context 和 swapped context 更准确地预测 held-out failure landscape。}
}
\]

只要这一点通过，才有科学依据进入 `inner_latent_calibration`，并最终让 \(z\) 真正参与危险场景搜索。
