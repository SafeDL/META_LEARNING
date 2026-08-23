# META_LEARNING：面向“可迁移元强化学习场景挖掘”的代码审计与重构计划

gpu环境在系统的：
conda activate metadrive


> 最低必须实现的主实验：**unseen SUT + unseen road geometry + few-shot adaptation**。  
> 理想扩展：**unseen SUT + unseen functional scenario + few-shot adaptation**。

---

# 1. 结论先行

当前 `mvr/` **已经具备一个很好的分层场景挖掘骨架，但尚未实现论文原始的“MoE-PEARL 可迁移场景搜索器”**。

当前代码最准确的技术定位是：

> **Hierarchical map-conditioned few-shot scenario miner for held-out IDM profiles on seen functional scenarios and fixed per-family geometries.**

而目标论文希望实现的是：

> **A meta-learned transferable scenario mining policy that rapidly adapts to an unseen SUT and unseen scene domain using a few test interactions.**

二者的主要差距不在 MetaDrive 执行、SAC 或 failure oracle，而在“**什么被定义成 meta-RL task，以及策略如何跨 task 共享**”。

## 1.1 当前满足程度

| 论文目标组件 | 当前状态 | 判断 |
|---|---|---|
| 危险具体场景 = 初始状态 + adversarial continuous policy | Outer 选初始位置/速度，Inner SAC 输出连续动作 | **基本满足** |
| 分层搜索：Outer 初始条件 + Inner 对抗驾驶 | 已完整实现 | **满足** |
| 显式地图/几何编码 | HPTR-style map encoder 已实现 | **满足表示层，未满足数据分布层** |
| unseen SUT few-shot | held-out IDM profile + posterior | **部分满足** |
| 不同风格/不同算法 SUT | registry 有 `rule_based` adapter，但正式 taskbook 仍全部为 IDM | **尚未验证** |
| unseen road geometry | 每个 family 只有一个 `map_hash` | **不满足** |
| unseen functional scenario | train/test 均包含 merge/cutin/roundabout | **不满足** |
| 统一跨场景搜索空间 | 仍以 family-specific candidate + lane-start `spawn_m` 表示 | **不满足** |
| PEARL | active path 使用 `SetPosterior` 均值池化，不是 PEARL PoG + RL-coupled inference | **不满足** |
| MoE Router | 无 learned router；只有 `parameter_space_id -> scene_policy` 硬路由 | **不满足** |
| learned experts | 每个 parameter space 一个独立 Outer head，更接近 hard-coded heads | **不满足原始 MoE 定义** |
| RSS validity | 当前是 physical validity / invalid event penalty，不是 RSS responsibility model | **未实现** |
| diversity | novelty-based failure signature 已存在 | **基本满足** |
| OOD 2×2/多维实验协议 | evaluation 只有单一 `meta_test` split | **不满足** |

因此：

**现在不建议直接进行最终大规模正式训练。**  
应先把“可迁移 task 定义”和“shared meta-policy”改正确，再进行正式实验。

---

# 2. 论文中应固定的三个定义

后续所有代码设计建议围绕以下三个定义展开。

## 2.1 Scenario Domain

对于功能场景 \(F\)，定义：

\[
\mathcal D_F = (\mathcal M_F,\mathcal X_F,\Pi_{adv})
\]

其中：

- \(\mathcal M_F\)：该功能场景允许的道路/拓扑/几何集合；
- \(\mathcal X_F\)：允许的初始交通状态空间；
- \(\Pi_{adv}\)：对抗车允许采用的闭环行为策略空间。

`merge / cut-in / roundabout / intersection` 应理解为 **Functional Scenario Family**，而不是一个具体 logical/concrete scenario。

## 2.2 Scenario-Mining Task

建议正式定义：

\[
\boxed{\mathcal T=(U,F,M)}
\]

其中：

- \(U\)：待测 SUT；
- \(F\)：功能场景；
- \(M\)：该功能场景中的一个具体道路几何实例。

每一个 \(\mathcal T\) 对应一个场景挖掘任务。

传统 RL 的问题是：

\[
\pi_{\mathcal T}^{*}
=
\arg\max_\pi
\mathbb E[J \mid \mathcal T]
\]

换一个 \(U\) 或 \(M\) 往往需要重新训练。

本文的元强化学习目标是：

\[
\mathcal T_{train}
\rightarrow
\text{learn transferable search prior}
\rightarrow
K\text{-shot adapt to }\mathcal T_{new}
\]

## 2.3 Concrete Adversarial Scenario

建议定义：

\[
\boxed{C=(M,x_0,\pi_{adv},\xi)}
\]

其中：

- \(M\)：确定道路几何；
- \(x_0\)：确定初始状态；
- \(\pi_{adv}\)：确定的对抗车闭环策略；
- \(\xi\)：执行 seed / background conditions 等复现信息。

完整测试实例：

\[
Q=(U,C)
\]

最终搜索目标：

\[
C^*
=
\arg\max_C
J(\operatorname{Sim}(U,C))
\]

这与当前代码中：

- Outer：选初始位置、初始速度、route/conflict candidate；
- Inner SAC：逐 timestep 输出 steering / throttle-brake；

的总体结构是兼容的，因此现有分层架构应保留。

---

# 3. 当前代码中值得保留的部分

以下模块不应推翻重写。

## 3.1 `mvr/map/`

保留：

- `map/metadrive_tokenizer.py`
- `map/schema.py`
- `map/relations.py`
- `map/hptr_encoder.py`

优点：

1. 已将 MetaDrive road network 转换为 lane polyline tokens；
2. HPTR-style encoder 使用相对位姿与 lane relation；
3. 不依赖 `"merge"` / `"roundabout"` one-hot 作为模型输入；
4. 天然具有 geometry-aware representation 的潜力。

需要增强的是 **interaction-specific encoding**，而不是替换现有 map encoder。

---

## 3.2 `mvr/state.py`

当前 `PhysicalStateExtractor` 是整个代码中最符合“跨几何迁移”目标的模块之一。

已经使用：

- relative longitudinal / lateral；
- relative heading；
- vehicle speeds；
- closing speed；
- pair distance；
- route progress；
- conflict timing。

特别是它已经计算：

- `RoutePolyline`
- `conflict_s`
- distance/time-to-conflict

因此后续应进一步把 **Outer 初始状态空间也改成 conflict-relative 坐标**，与 Inner 的语义保持一致。

---

## 3.3 `mvr/policy/adversarial_sac.py`

保留单一共享的 `OptionConditionedSAC` 基础结构。

原因：

- action 是 2-D continuous driving control；
- actor/critic 不绑定 Merge/Cut-in/Roundabout；
- 可以通过 `scene_embedding + latent + initial condition` 做 task conditioning；
- 与“危险场景 = 初始状态 + adversarial policy”定义完全一致。

后续重点是调整其 conditioning input，而不是重新实现 SAC。

---

## 3.4 `mvr/scenario/executor.py`

保留其“Outer action -> verified executable episode”的严格契约。

尤其保留：

- map hash verification；
- actual lane/spawn/speed assertion；
- SUT registry；
- AppliedScenario provenance；
- simulator fail-fast。

需要改的是 task / geometry / interaction 的抽象层，不应降低这些物理执行检查。

---

## 3.5 `mvr/failure/`

保留：

- `FailureCriteria`
- `FailureSignature`
- invalid episode filtering
- failure / severity / novelty

当前 failure 逻辑已经具备统一 threshold source，不需要为了新 meta-RL 架构重写。

---

## 3.6 `training/pipeline.py`

保留：

- canonical stage enforcement；
- checkpoint compatibility；
- taskbook hash；
- stage-level reproducible seed；
- manifest。

架构变化后修改 stage 内容即可，不需要重新制造复杂训练框架。

---

# 4. 当前最核心的七个结构性问题

---

## P0：当前 taskbook 没有 geometry distribution

### 当前实现

`configs/idm_taskbook.json` 中：

- Merge 全部使用 `merge-default`；
- Cut-in 全部使用 `cutin-default`；
- Roundabout 全部使用 `roundabout-default`；
- 同一 family 的 train / validation / meta-test 使用相同 `map_hash`。

因此当前只验证：

\[
U_{test}\notin U_{train}
\]

而：

\[
M_{test}=M_{train}
\]

### 后果

即使 HPTR map encoder 很强，也无法证明其 geometry transfer 能力。

模型可能把每个 family 的唯一 map embedding 当作固定 task identifier。

### 必须修改

增加真正的 geometry task distribution：

```text
Merge:
    geometry_001
    geometry_002
    geometry_003
    ...
Cut-in:
    geometry_001
    geometry_002
    ...
Roundabout:
    geometry_001
    geometry_002
    ...
```

并保证：

```text
train geometry_hash ∩ test geometry_hash = ∅
```

---

# 5. P1：Outer 搜索坐标目前不具备跨 geometry 语义一致性

这是当前最值得优先修改的技术问题之一。

## 5.1 当前 Outer 连续参数

`scenario/catalog.py`：

```python
adversary_spawn_m
sut_spawn_m
adversary_initial_speed_mps
sut_initial_speed_mps
```

其中 `spawn_m` 是相对 lane 起点的纵向位置。

### 问题

对于不同道路几何：

```text
Geometry A:
lane start ------------------- conflict

Geometry B:
lane start ----------------------------------------- conflict
```

同样：

```text
spawn_m = 10
```

并不意味着车辆距离 conflict point 相同。

因此策略学到的：

> 在 lane start 后 10 m 生成车辆

无法自然迁移成：

> 在新几何中距冲突区约某个危险距离生成车辆。

## 5.2 目标表示

Outer 统一搜索：

\[
x_0=
[
d_{sut}^{C},
d_{adv}^{C},
v_{sut},
v_{adv}
]
\]

其中：

- `sut_distance_to_conflict_m`
- `adversary_distance_to_conflict_m`
- `sut_initial_speed_mps`
- `adversary_initial_speed_mps`

全部是 interaction-centric 参数。

adapter 再执行：

```python
spawn_s = conflict_s - distance_to_conflict_m
```

将其反解为 MetaDrive 的 lane `spawn_longitude`。

## 5.3 文件修改

### 修改

`mvr/scenario/catalog.py`

从：

```python
"adversary_spawn_m"
"sut_spawn_m"
```

改为：

```python
"adversary_distance_to_conflict_m"
"sut_distance_to_conflict_m"
```

### 修改

`mvr/scenario/adapters/base.py`

新增类似：

```python
def spawn_from_conflict_distance(
    route: RoutePolyline,
    conflict_xy: tuple[float, float],
    distance_to_conflict_m: float,
) -> float:
    conflict_s = route.conflict_s(conflict_xy)
    return conflict_s - distance_to_conflict_m
```

并进行边界验证。

### 修改

`mvr/scenario/executor.py`

执行顺序应变为：

```text
build map
→ enumerate/resolve candidate interaction
→ construct route polylines
→ compute conflict-relative spawn positions
→ instantiate SUT/adversary
→ verify actual physical state
```

### 修改

`AppliedScenario`

正式保存：

```text
sut_distance_to_conflict_m
adversary_distance_to_conflict_m
```

而不仅仅保存 simulator-specific `spawn_m`。

---

# 6. P2：需要把“family-specific candidate index”改成可迁移的 Interaction Candidate

当前：

```text
merge:
    main_conflict
    downstream_merge

cutin:
    left_target_lane
    right_target_lane

roundabout:
    entry_0_exit_1
    ...
```

这些字符串适合 provenance，但不适合成为 universal model action semantics。

## 6.1 新增核心抽象

新增：

`mvr/scenario/interaction.py`

建议：

```python
@dataclass(frozen=True)
class InteractionCandidate:
    candidate_id: str
    sut_route: tuple[LaneIndex, ...]
    adversary_route: tuple[LaneIndex, ...]
    conflict_xy: tuple[float, float]

    # geometry descriptors
    crossing_angle_rad: float
    sut_distance_available_m: float
    adversary_distance_available_m: float
    sut_route_curvature: float
    adversary_route_curvature: float
```

模型不需要知道：

```text
candidate_id = "main_conflict"
```

模型需要看到：

```text
two routes
+
conflict zone
+
relative geometry
```

## 6.2 修改 adapter contract

当前：

```python
resolve_layout(..., candidate_strings)
```

建议拆成：

```python
enumerate_interactions(env, task) -> tuple[InteractionCandidate, ...]
instantiate_interaction(env, task, candidate, initial_state)
```

family adapter 的职责只剩：

> 将通用 interaction 定义翻译成具体 MetaDrive 几何。

而不是让 learning policy 通过 family-specific index 学习。

---

# 7. P3：Map Encoder 后应新增 Interaction Encoder

当前 `HPTRMapEncoder` 返回：

```text
local lane embeddings H_map
global map embedding h_map
```

但 Outer 最终只使用全局：

```text
h_map + z
```

这不足以完成 variable-size conflict candidate selection。

## 7.1 新增

`mvr/map/interaction_encoder.py`

输入：

```text
H_map
InteractionCandidate
```

输出：

\[
h_i^{candidate}\in\mathbb R^d
\]

每个候选冲突点有一个 embedding。

建议接口：

```python
@dataclass(frozen=True)
class SceneEncoding:
    global_embedding: torch.Tensor
    candidate_embeddings: torch.Tensor
    candidate_mask: torch.Tensor
```

## 7.2 Outer 最终输入

从：

\[
[h_{map},z]
\]

变成：

\[
[h_{scene},z,H_{candidate}]
\]

其中：

\[
h_{scene}=f(h_{map},route_{sut},route_{adv},conflict)
\]

---

# 8. P4：当前 Outer 是 hard-routed family heads，不是 MoE

## 8.1 当前代码

`model.py`：

```python
self.scene_policies = nn.ModuleDict({
    identifier: HybridScenePolicy(...)
    for identifier, space in parameter_spaces.items()
})
```

运行时：

```python
scene_policies[task.parameter_space_id]
```

也就是：

```text
merge_v1       -> Merge policy
cutin_v1       -> Cut-in policy
roundabout_v1  -> Roundabout policy
```

这是 **hard routing by known family**。

它不等价于：

```text
Task-aware MoE Router
→ learned expert mixture
```

更重要的是：

如果未来：

```text
F_test = intersection
```

且训练时没有 `intersection_v1` head，则当前模型不存在可直接迁移的 policy head。

---

# 9. 目标：Universal MoE Outer Search Policy

建议新增：

```text
mvr/policy/moe_router.py
mvr/policy/universal_scene_policy.py
```

不要把 expert 硬绑定：

```text
Expert 0 = Merge
Expert 1 = Cut-in
Expert 2 = Roundabout
```

否则仍然无法支持 unseen functional scenario。

## 9.1 Router

\[
w = \operatorname{softmax}(g(h_{scene},z))
\]

建议：

```python
class TaskAwareMoERouter(nn.Module):
    def forward(
        self,
        scene_embedding: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

输出：

```text
[B, num_experts]
```

## 9.2 Experts

每个 expert 都工作在**同一个 interaction-centric initial-state space**：

\[
[d_{sut}^{C}, d_{adv}^{C}, v_{sut}, v_{adv}]
\]

例如：

```python
class InitialConditionExpert(nn.Module):
    def forward(self, shared_features):
        return mean, log_std
```

expert 的含义应通过训练自动分化，而不是人工命名为 Merge/Roundabout。

## 9.3 Candidate selection

使用共享 pointer/scorer：

\[
score_i
=
f(h_i^{candidate},h_{scene},z)
\]

再：

\[
p(c=i)=softmax(score_i)
\]

因此 candidate 数量可以随场景变化：

```text
Merge: 2
Roundabout: 3
Intersection: 4
```

而无需新增 neural head。

## 9.4 最简 MoE PPO 实现

为降低实现复杂度，建议首先使用 **discrete expert selection**，而不是连续 weighted-mixture action：

```text
router Categorical -> expert_id
shared candidate Categorical -> candidate_id
expert Normal -> continuous x0
optional option Categorical -> adversarial_option
```

总 log-prob：

\[
\log p(a)
=
\log p(e)
+
\log p(c)
+
\log p(x_0|e)
+
\log p(o)
\]

这样可以直接复用现有 PPO update 框架。

---

# 10. P5：`OuterRolloutBuffer` 必须同步升级

当前 Outer replay 只保存：

```text
map_embedding
latent
candidate index
continuous action
option
log_prob
value
```

Universal MoE 后需要至少保存：

```text
scene_embedding
candidate_embeddings / candidate features
candidate_mask
latent
expert_index
router_prob/logprob
candidate_index
initial_condition
option (若保留)
old_logprob
value
reward
```

修改：

`mvr/training/replay.py`

新增字段：

```python
@dataclass(frozen=True)
class OuterRolloutStep:
    scene_embedding: torch.Tensor
    candidate_embeddings: torch.Tensor
    candidate_mask: torch.Tensor
    latent: torch.Tensor
    expert_index: torch.Tensor
    candidate_index: torch.Tensor
    initial_condition: torch.Tensor
    ...
```

---

# 11. P6：当前 few-shot posterior 不是 PEARL

这是论文命名必须解决的问题。

## 11.1 当前 active path

`context/set_posterior.py`：

```text
episode token
→ MLP
→ mean pooling across episodes
→ Gaussian mean/logvar
```

`online_meta_test.py` 又直接：

```python
latent_after, _ = model.infer_posterior(...)
```

实际 policy 使用的是 posterior mean。

这与 PEARL 的核心设计不同。

## 11.2 仓库中真正的 PEARL-style PoG

当前冻结 baseline：

```text
archives/pearl_learning/src/context_encoder.py
```

已经有：

```text
transition evidence
→ Gaussian factors
→ Product-of-Gaussians
→ q(z|c)
```

但它属于 baseline，不应被 active `mvr` runtime import。

---

# 12. 如果论文继续叫 “MoE-PEARL-ScenarioMiner”，必须做的改动

新增：

`mvr/context/pearl_context.py`

实现主方法自己的：

```python
class PearlContextEncoder(nn.Module):
    def prior(...)
    def forward(context, prior=None)
    def sample(mu, logvar)
    def kl_to_prior(...)
```

可以借鉴：

```text
archives/pearl_learning/src/context_encoder.py
```

的数学实现，但**不要建立运行时依赖**：

```python
from archives.pearl_learning...
```

不推荐。

---

# 13. PEARL context 应输入什么？

建议不要直接把 SUT ID、family ID 放进去。

可以定义单次测试 evidence：

\[
c_i =
[
h_{scene},
x_{0,i},
trajectory_i,
outcome_i
]
\]

其中：

- `h_scene`：显式 geometry / interaction encoding；
- `x0`：实际初始条件；
- `trajectory`：SUT-adversary interaction；
- `outcome`：criticality / validity / severity。

当前已有：

- `TrajectoryEncoder`
- `EpisodeTokenBuilder`
- `OutcomeSchema`

因此可以继续保留 **episode-level evidence token**。

如果论文严格声称“PEARL 原算法”，则应实现 transition-wise Gaussian factors；如果只使用 episode token PoG，应在论文中称：

> **PEARL-inspired probabilistic context inference**

而不要写成“严格复现 PEARL”。

---

# 14. 更关键：posterior 应与 RL objective 发生联系

当前 posterior 主要通过：

```text
support token
→ predict held-out target outcome
```

训练。

这是一种合理的 latent outcome model，但不等价于标准 PEARL meta-RL。

如果论文坚持 PEARL，应至少让：

\[
z
\]

进入 SAC critic / policy learning，并允许 context encoder 通过 RL objective 接收梯度。

## 14.1 推荐折中实现

保留当前辅助 `OutcomeDecoder`，但 posterior 总 loss 改为：

\[
L_{context}
=
L_{RL}
+
\lambda_{pred}L_{outcome}
+
\beta KL(q(z|c)||p(z))
\]

其中：

- `L_RL`：posterior-conditioned Inner SAC critic objective 或 meta query return；
- `L_outcome`：当前 held-out outcome prediction；
- `KL`：PEARL regularization。

这样既保留当前代码优势，又能合理称为 meta-RL context inference。

---

# 15. P7：当前 posterior task sampling 需要升级

当前 `train_posterior()`：

```text
选一个 task
→ 在同一个 task 中收 K support + 1 target
→ train posterior
```

但 task distribution 本身只有：

```text
4 IDM profiles × 3 fixed maps/families
```

因此它不可能学会 geometry transfer。

## 15.1 新增

`mvr/training/meta_sampler.py`

建议：

```python
@dataclass
class MetaTaskBatch:
    tasks: list[ScenarioMiningTaskSpec]
```

支持按轴采样：

```text
SUT
Functional Scenario
Geometry
```

训练 batch 应覆盖大量：

\[
(U_i,F_j,M_k)
\]

组合。

## 15.2 第一阶段不必做复杂 cross-task posterior

最简、理论清晰的 PEARL 方式仍然可以：

```text
support/query 来自同一个 T=(U,F,M)
```

关键是：

> meta-training 时 T 本身必须来自广泛的 SUT×geometry×family 分布。

如果后续希望 latent 更接近稳定的 SUT vulnerability，可再加入：

```text
same SUT, different geometry
```

的 latent consistency / cross-context query 约束。

这应作为后续增强，不要第一版就引入太多正则项。

---

# 16. P8：SUT abstraction 已有接口，但 task 分布仍太窄

当前：

`SUTRegistry` 已注册：

```text
IDMSUTAdapter
RuleBasedSUTAdapter
```

但 profile 实际只注册了多个 IDM。

因此当前 held-out test：

```text
idm_late_response
```

仍然是 **同一算法族内参数外推**。

这可以作为 SUT-OOD level 1，但不足以支持：

> different-style SUT

## 16.1 建议增加两级 SUT generalization

### Level S1：parametric unseen SUT

保留：

```text
train:
idm_cautious
idm_defensive
idm_normal
idm_assertive

test:
idm_late_response
```

### Level S2：behavior-family unseen SUT

增加：

```text
rule_yielding
rule_gap_accepting
```

或其他独立 black-box controller adapter。

---

# 17. `ControllerProfile` 建议泛化

当前 `ControllerProfile` 字段明显偏 IDM：

```text
target_speed
distance_wanted
time_headway
acceleration_factor
deceleration_factor
```

为了支持未来 external ADS / rule-based controller，建议改成：

```python
@dataclass(frozen=True)
class SUTProfile:
    profile_id: str
    adapter_name: str
    parameters: Mapping[str, object]
```

由具体 adapter 自己验证：

```python
adapter.validate_profile(profile)
```

模型仍禁止读取这些参数。

这样以后无需修改核心 schema 即可增加：

```text
IDM
rule-based
learned policy
external ADS
```

---

# 18. P9：TaskSpec 必须从单一 `split` 升级为多 OOD 轴

当前：

```python
split = meta_train / meta_validation / meta_test
```

无法描述：

```text
unseen SUT but seen geometry
seen SUT but unseen geometry
unseen SUT + unseen geometry
unseen functional family
```

## 18.1 建议改名

`MetaTestTaskSpec`

→

`ScenarioMiningTaskSpec`

## 18.2 推荐字段

```python
@dataclass(frozen=True)
class ScenarioMiningTaskSpec:
    task_id: str

    sut_ref: str
    functional_scenario: str

    geometry_id: str
    geometry_hash: str
    geometry_seed: int

    adapter_id: str
    interaction_schema_id: str

    sut_split: str
    geometry_split: str
    functional_split: str
```

不要再把：

```text
scenario_family
parameter_space_id
map_id
seed
```

混成一个整体 split。

### 特别重要

当前 `task.seed` 同时被用于：

- MetaDrive `start_seed`
- SUT attach/reset
- episode execution

建议拆开：

```text
geometry_seed
episode_seed
training_seed
```

否则“改变几何”可能与“改变 SUT/control randomness”耦合。

---

# 19. 不要继续硬编码 `SUPPORTED_FAMILIES`

当前：

```python
SUPPORTED_FAMILIES = {"merge", "cutin", "roundabout"}
```

会让 unseen functional family 在 schema 层直接失败。

建议：

- task schema 只检查字符串非空；
- 是否可执行由 `ScenarioAdapterRegistry` 判断；
- learning model 不读取 functional scenario ID。

例如：

```python
adapter_registry.get(task.adapter_id)
```

而不是：

```python
if task.scenario_family not in SUPPORTED_FAMILIES:
    fail
```

---

# 20. `ScenarioAdapter` 应成为 simulator translator，而不是 learning taxonomy

推荐新增：

`mvr/scenario/registry.py`

统一：

```python
ADAPTERS = {
    "merge": MergeScenarioAdapter(),
    "cutin": CutInScenarioAdapter(),
    "roundabout": RoundaboutScenarioAdapter(),
}
```

并从：

- `training/trainers.py`
- `scripts/validate_mvr.py`

删除重复 mapping。

adapter 只负责：

```text
functional specification
→ MetaDrive geometry
→ interaction candidates
→ concrete vehicle spawning
```

不决定 neural policy head。

---

# 21. Geometry 数据如何生成

建议新增：

`mvr/scripts/build_taskbook.py`

或：

`mvr/scenario/taskbook_builder.py`

流程：

```text
for functional scenario
    for geometry seed/config
        build MetaDrive map
        tokenize map
        compute map_hash
        enumerate interaction candidates
        save geometry manifest
```

必须 fail-fast：

```python
assert train_hashes.isdisjoint(test_hashes)
```

以及：

```python
if len(unique_map_hashes) < requested_geometries:
    raise RuntimeError(...)
```

避免“换 seed 但实际地图没变”。

---

# 22. 第一版 Geometry 实验不需要很多地图

不建议立即生成几十或几百张地图。

最低足够：

```text
每个 Functional Scenario:
    3 train geometries
    1 validation geometry
    2 test geometries
```

若 3 个 families：

```text
18 geometries total
```

已经足以验证：

- seen geometry
- unseen geometry
- joint OOD

等协议。

---

# 23. P10：Inner SAC 的改动应尽可能小

当前 Inner 已是较好的共享 policy。

建议继续：

\[
a_t^{adv}
\sim
\pi_{SAC}
(
s_t,
h_{scene},
z,
x_0,
o
)
\]

## 23.1 修改 `SharedFeatureEncoder`

当前：

```text
state
map_embedding
latent
option
config
```

建议：

```text
physical_state
scene_embedding
latent
option/expert_embedding
initial_condition
```

这里 `initial_condition` 使用统一 conflict-relative x0。

## 23.2 Map input

Inner 不必每一步读取所有 map lane tokens。

使用：

```text
scene_embedding
```

即可。

这样计算量小且具有几何语义。

---

# 24. `AdversarialOption` 是否删除？

第一阶段 **不要删除**。

当前：

```text
approach_conflict
yield_then_press
gap_close
route_block
```

具有跨场景的高层行为语义，并且：

- `InnerRiskReward`
- option embedding
- Inner SAC

已经依赖它。

推荐将其重新解释为：

> **latent/high-level adversarial behavior primitive**

而不是 concrete scenario 的必要显式组成。

最终：

\[
C=(M,x_0,\pi_{adv})
\]

其中 option 只是定义 \(\pi_{adv}\) 的内部 conditioning 变量。

### 后期可选

如果 MoE expert 已经形成稳定的 behavior specialization，可以实验：

```text
remove option
```

作为结构简化，而不是现在同时重写两个机制。

---

# 25. P11：MoE 与 Option 不应形成重复的 family hard-coding

推荐：

```text
MoE Expert
    = transferable search mode / initialization prior

Adversarial Option
    = Inner SAC high-level driving intention
```

不要定义：

```text
Expert 1 = Merge
Expert 2 = Cut-in
Expert 3 = Roundabout
```

否则 MoE 只是把现有 hard routing 换名字。

推荐 expert 数：

```text
4
```

先固定，不做 expert-number 大网格实验。

---

# 26. P12：训练 pipeline 如何调整

当前：

```text
inner_pretrain
→ posterior
→ inner_latent_calibration
→ outer
```

这套 staged bootstrap 可以保留。

推荐改成：

```text
1. inner_pretrain
2. context_pretrain
3. inner_latent_calibration
4. outer_moe
5. optional light_joint_meta
```

## Stage 1：Inner pretrain

目标：

> 在多 SUT × 多 geometry × 多 family 中学到通用的 adversarial driving prior。

训练组件：

```text
map/scene encoder
shared feature encoder
option embedding
inner SAC
```

## Stage 2：Context pretrain

目标：

> 从 K 个 interaction episodes 推断当前 task residual / vulnerability latent。

训练组件：

```text
trajectory encoder
PEARL context encoder
outcome decoder
```

如果严格 PEARL，引入 RL-coupled loss。

## Stage 3：Inner latent calibration

保留。

目标：

> 让 Inner SAC 真正利用 posterior z，而不是只在 prior z=0 下工作。

## Stage 4：Outer MoE

训练：

```text
router
candidate pointer
experts
value head
```

冻结或低学习率更新基础编码器。

## Stage 5：Light joint meta（建议最终增加，但先不要作为第一版 blocker）

联合微调：

```text
context encoder
router
experts
inner SAC
```

小学习率、少量预算。

目的：

> 让 context 推断真正服务于 failure-discovery return。

当前 `TrainingStage.LIGHT_JOINT` 已存在概念，可正式启用，而不是另造 stage 名称。

---

# 27. 如果坚持论文名中的 “PEARL”，`light_joint` 很重要

原因：

当前 context encoder 与 search policy 是分阶段独立优化的。

这更像：

```text
latent outcome model
+
latent-conditioned RL
```

而不是严格的：

```text
meta-RL task inference optimized for return
```

如果最终论文写：

> PEARL-based meta-RL

建议至少通过 joint meta fine-tuning 让 context posterior 与 RL objective 建立梯度/优化联系。

否则建议论文术语改成：

> PEARL-inspired probabilistic task inference

---

# 28. P13：Evaluation 必须从一个 `meta_test` split 改成 OOD regime matrix

新增：

`mvr/evaluation/regimes.py`

建议定义：

```python
class EvalRegime(str, Enum):
    ID = "seen_sut_seen_geometry"
    SUT_OOD = "unseen_sut_seen_geometry"
    GEO_OOD = "seen_sut_unseen_geometry"
    JOINT_OOD = "unseen_sut_unseen_geometry"
    FUNC_OOD = "unseen_functional"
    FULL_OOD = "unseen_sut_unseen_functional"
```

---

# 29. 论文主实验矩阵

建议：

| Regime | SUT | Geometry | Functional Scenario | 角色 |
|---|---|---|---|---|
| R1 | seen | seen | seen | sanity |
| R2 | unseen | seen | seen | SUT transfer |
| R3 | seen | unseen | seen | geometry transfer |
| **R4** | **unseen** | **unseen** | seen | **最低主贡献** |
| R5 | seen/unseen | unseen | unseen | functional transfer |
| R6 | unseen | unseen | unseen | strongest extension |

正式论文至少必须完成：

```text
R1
R2
R3
R4
```

R5/R6 在 Universal Outer 通过后再做。

---

# 30. Few-shot 评估保持 fixed all-in budget

当前：

```text
total budget = 20
K = 0 / 1 / 2 / 4
```

这个设计很好，应保留。

不要改成：

```text
20 query + K free support
```

否则 K 越大预算越高，不公平。

推荐主指标：

1. Failure Discovery AUC
2. Unique Valid Failures @ B
3. Tests-to-first-valid-failure
4. Invalid Rate
5. Adaptation Gain

新增：

\[
\Delta AUC(K)
=
AUC(K)-AUC(0)
\]

它直接回答：

> few-shot context 是否真的提高场景挖掘效率？

---

# 31. P14：当前 reward 不等于 RSS

当前 Outer reward：

```text
failure
+ novelty
+ severity
- invalid episode
```

当前 validity 由：

```text
non-target collision
out-of-road
wrong-route
...
```

构成。

这不是 RSS responsibility model。

## 建议

### 第一版论文

不要写：

> RSS validity

写：

> **validity-constrained failure discovery**

即可。

### 如果 RSS 是必须贡献

再新增：

```text
mvr/safety/rss.py
```

显式计算：

```text
safe longitudinal distance
safe lateral distance
proper response
responsibility-valid dangerous interaction
```

并在 `FailureSignature` 中记录：

```text
rss_valid
rss_violation_type
```

不要为了匹配原始框架图而加入一个没有完整定义的“RSS reward”。

---

# 32. plausibility / diversity 如何处理

## Diversity

当前：

```text
NoveltyTracker
FailureSignature
unique failures
```

已经提供相对合理的 diversity proxy。

无需单独增加复杂 diversity network。

## Plausibility

第一版建议主要通过：

- MetaDrive vehicle dynamics；
- valid lane / route；
- bounded steering/acceleration；
- invalid event penalty；

保证。

若需要论文指标，再增加：

```text
action jerk
max lateral acceleration
max longitudinal acceleration
route validity
```

而不是先设计一个难解释的 plausibility reward。

---

# 33. `archives/pearl_learning/` 与 `archives/sac_scenario_mining/` 的定位

两者均不应成为 active method runtime dependency。

## `archives/pearl_learning/`

定位：

> frozen PEARL baseline / implementation reference

可参考：

```text
archives/pearl_learning/src/context_encoder.py
```

中的：

```text
Product-of-Gaussians
exact empty-context prior
KL
```

但主方法应在：

```text
mvr/context/
```

独立实现。

禁止：

```python
from archives.pearl_learning.src.context_encoder import ...
```

作为正式 active method。

## `archives/sac_scenario_mining/`

定位：

> frozen single-task SAC scenario mining baseline

它适合说明传统问题：

```text
specific SUT + specific geometry + specific functional scenario
→ train SAC from scratch
```

正好可作为本文 motivation baseline。

当前 active Inner SAC 位于：

```text
mvr/policy/adversarial_sac.py
```

二者不是同一个模块。

---

# 34. 推荐目标文件结构

建议最终 `mvr/`：

```text
mvr/
├── README.md
├── model.py
│
├── configs/
│   ├── mvr.yaml
│   ├── taskbook.json
│   └── geometry_catalog.json
│
├── scenario/
│   ├── task_spec.py
│   ├── geometry.py                 # NEW
│   ├── interaction.py              # NEW
│   ├── concrete.py                 # NEW
│   ├── registry.py                 # NEW
│   ├── catalog.py
│   ├── executor.py
│   ├── layout.py
│   ├── route_geometry.py
│   ├── parameter_space.py
│   ├── option.py
│   └── adapters/
│       ├── base.py
│       ├── merge.py
│       ├── cutin.py
│       └── roundabout.py
│
├── map/
│   ├── schema.py
│   ├── metadrive_tokenizer.py
│   ├── hptr_encoder.py
│   ├── interaction_encoder.py      # NEW
│   └── relations.py
│
├── context/
│   ├── trajectory_features.py
│   ├── trajectory_encoder.py
│   ├── outcome_schema.py
│   ├── episode_token.py
│   ├── pearl_context.py            # NEW
│   └── outcome_decoder.py          # 可从 set_posterior.py 拆出
│
├── policy/
│   ├── adversarial_sac.py
│   ├── shared_features.py
│   ├── moe_router.py               # NEW
│   ├── universal_scene_policy.py   # NEW
│   └── scene_policy.py             # 过渡后删除/替换
│
├── sut/
│   ├── base.py
│   ├── registry.py
│   ├── idm.py
│   └── rule_based.py
│
├── failure/
│   └── ...
│
├── training/
│   ├── pipeline.py
│   ├── stages.py
│   ├── trainers.py
│   ├── meta_sampler.py             # NEW
│   ├── online_meta_test.py
│   ├── replay.py
│   └── updates.py
│
├── evaluation/
│   ├── regimes.py                  # NEW
│   ├── run.py
│   ├── metrics.py
│   └── baselines.py
│
└── scripts/
    ├── build_taskbook.py            # NEW
    ├── train_mvr.py
    ├── evaluate_mvr.py
    └── validate_mvr.py
```

不要把 `archives/pearl_learning/` 或 `archives/sac_scenario_mining/` 搬入 `mvr/`。

---

# 35. `model.py` 的目标结构

当前：

```text
map_encoder
SetPosterior
scene_policies[parameter_space_id]
shared_feature_encoder
inner_sac
```

目标：

```python
class TransferableScenarioMiner(nn.Module):
    map_encoder
    interaction_encoder

    context_encoder        # PEARL-style
    outcome_decoder

    moe_router
    universal_scene_policy

    option_embedding
    shared_feature_encoder
    inner_sac
```

示意：

```text
Map / Geometry
     ↓
HPTR Map Encoder
     ↓
Interaction Encoder
     ↓
h_scene + H_candidates
     │
     ├───────────────┐
     │               │
     │          support episodes
     │               ↓
     │        PEARL Context Encoder
     │               ↓
     │              z
     │               │
     └──────┬────────┘
            ↓
        MoE Router
            ↓
 Universal Outer Policy
            ↓
 candidate + x0 (+ option)
            ↓
        MetaDrive
            ↓
    Shared Inner SAC
            ↓
 continuous adversarial actions
            ↓
 trajectory/outcome
            ↓
 next posterior update
```

---

# 36. `SetPosterior` 如何迁移

不建议直接删除其思想。

当前 `SetPosterior` / outcome decoder 可以暂时保留为：

```text
ablation: SetPosterior
```

正式主方法切到：

```text
PearlContextEncoder
```

这样论文可以做一个非常有价值、且不需要大量额外实验的消融：

```text
PEARL-style probabilistic context
vs
simple set-mean context
```

如果篇幅不够，可以不放主文，只保留工程验证。

---

# 37. Taskbook v2 示例

建议：

```json
{
  "task_id": "merge-g03-idm_late_response",
  "sut_ref": "idm_late_response",
  "functional_scenario": "merge",

  "geometry_id": "merge-g03",
  "geometry_hash": "...",
  "geometry_seed": 103,

  "adapter_id": "merge",
  "interaction_schema_id": "two_route_conflict_v1",

  "sut_split": "test",
  "geometry_split": "test",
  "functional_split": "train"
}
```

联合 OOD：

```text
sut_split      = test
geometry_split = test
functional_split = train
```

即：

```text
R4: unseen SUT + unseen geometry
```

---

# 38. 配置文件建议

`mvr.yaml` 不要塞每个 geometry。

只定义算法：

```yaml
schema: transferable_scenario_miner_v2

taskbook: mvr/configs/taskbook.json

model:
  state_dim: 10
  scene_dim: 128
  latent_dim: 16
  num_experts: 4

context:
  type: pearl_product
  kl_weight: 0.001
  support_shots: [1, 2, 4]

outer:
  algorithm: hybrid_moe_ppo
  ppo_epochs: 4
  batch_size: 64

inner:
  algorithm: sac
  action_dim: 2

evaluation:
  total_episode_budget: 20
  support_shots: [0, 1, 2, 4]
  regimes:
    - seen_sut_seen_geometry
    - unseen_sut_seen_geometry
    - seen_sut_unseen_geometry
    - unseen_sut_unseen_geometry
```

不要把每个 task 细节写入 YAML。

---

# 39. 必须新增的单元/契约测试

## Task / split

新增：

`test_task_generalization_contracts.py`

检查：

```python
train_suts.isdisjoint(test_suts)
train_geometry_hashes.isdisjoint(test_geometry_hashes)
```

对于 functional OOD：

```python
train_functionals.isdisjoint(test_functionals)
```

至少一个专用 fold。

---

## Geometry

新增：

`test_interaction_coordinates.py`

验证同样：

```text
distance_to_conflict = 20 m
```

在两个不同 map geometry 中实例化后，真实沿 route 距 conflict 都约为 20 m。

这是 unseen geometry 能否成立的关键测试。

---

## Candidate

验证：

```text
2 candidates
3 candidates
4 candidates
```

均可通过同一个 `UniversalScenePolicy.forward()`。

---

## MoE

新增：

`test_moe_router.py`

检查：

```text
router weights sum to 1
expert indices valid
all experts can receive gradients
load-balance regularizer finite
```

并明确禁止：

```text
router input contains functional_scenario ID
```

---

## PEARL context

新增：

`test_pearl_context.py`

检查：

1. empty context -> exact unit Gaussian prior；
2. context permutation 不改变 posterior；
3. evidence 增加后 posterior 发生变化；
4. `sample()` 与 mean 行为区分；
5. KL finite；
6. SUT ID 不进入 input。

---

## Online adaptation

新增：

`test_online_meta_adaptation.py`

验证：

```text
K=0:
latent stays prior

K=2:
latent changes after first 2 evidence episodes
then stays fixed if support limit reached
```

---

## Concrete scenario replay

新增：

`test_concrete_scenario_replay.py`

一个正式输出的危险场景应能够由：

```text
map/geometry hash
x0
adversarial policy checkpoint/hash
option/internal mode
seed
```

重建并回放。

---

# 40. 推荐的代码修改顺序

不要同时改所有模块。

---

## Commit 1 — `Define transferable scenario tasks`

修改：

```text
scenario/task_spec.py
sut/base.py
sut/registry.py
configs/taskbook
```

新增：

```text
scenario/geometry.py
scenario/registry.py
scripts/build_taskbook.py
```

目标：

- SUT / geometry / functional split 显式分离；
- 生成多 geometry；
- 无训练算法变化。

### 验收

```text
pytest passes
train/test map hash disjoint
existing three adapters still executable
```

---

## Commit 2 — `Use conflict-relative initial conditions`

修改：

```text
scenario/catalog.py
scenario/adapters/base.py
scenario/executor.py
scenario/applied.py
```

目标：

```text
spawn_m
→
distance_to_conflict_m
```

### 验收

两个不同 geometry 上：

```text
requested 20m to conflict
≈ actual 20m to conflict
```

---

## Commit 3 — `Add interaction-centric scene encoding`

新增：

```text
scenario/interaction.py
map/interaction_encoder.py
```

修改：

```text
adapters/base.py
model.py
```

目标：

```text
MapTokens
+
route pair
+
conflict zone
→
scene embedding
+
candidate embeddings
```

### 验收

- same model handles variable candidate count；
- SE(2) transform 后 embedding 稳定；
- family ID 不作为输入。

---

## Commit 4 — `Replace family heads with universal MoE policy`

新增：

```text
policy/moe_router.py
policy/universal_scene_policy.py
```

修改：

```text
model.py
training/replay.py
training/updates.py
training/trainers.py
training/online_meta_test.py
```

删除/停用：

```text
scene_policies[parameter_space_id]
```

### 验收

同一个 checkpoint 可以 forward：

```text
merge
cut-in
roundabout
```

且不查找：

```text
scene_policies["merge_v1"]
```

---

## Commit 5 — `Add active PEARL context inference`

新增：

```text
context/pearl_context.py
training/meta_sampler.py
```

修改：

```text
model.py
training/posterior_data.py
training/trainers.py
training/updates.py
online_meta_test.py
```

目标：

- probabilistic posterior；
- exact prior；
- few-shot sample/mean protocol；
- active path 不依赖 `archives/pearl_learning/`。

---

## Commit 6 — `Add OOD evaluation regimes`

新增：

```text
evaluation/regimes.py
```

修改：

```text
evaluation/run.py
configs/mvr.yaml
README.md
```

正式支持：

```text
R1/R2/R3/R4
```

此时才值得开始论文正式训练。

---

## Commit 7 — `Optional functional-scenario OOD`

前 6 步跑通后才做。

目标：

```text
leave-one-functional-family-out
```

例如：

```text
train: merge + cutin
test: roundabout
```

同一个 universal model 必须可以 forward / execute。

如果做不到，不影响 R4 主论文目标。

---

# 41. 第一轮不要做的事情

为了避免再次膨胀：

1. 不要新增 10 个 Functional Scenario；
2. 不要新增大量 SUT 算法；
3. 不要做 expert 数量大网格；
4. 不要同时改 PPO/SAC 为全新 RL 算法；
5. 不要立即引入真实 ADS；
6. 不要引入复杂 RSS，除非它是论文明确贡献；
7. 不要让 `mvr` import legacy baseline；
8. 不要为了 “PEARL” 名称复制整个 `archives/pearl_learning/`；
9. 不要用 family one-hot 帮 Universal Policy；
10. 不要把 functional scenario holdout 和 geometry holdout 混成一个指标。

---

# 42. 第一阶段建议只追求 R4

论文最低可发表且与 motivation 高度一致的主问题：

> **在一个未见过的道路几何和行为风格不同的 SUT 上，能否利用已经从多个场景挖掘任务中学习到的搜索先验，通过少量测试交互，快速发现危险初始状态和 adversarial driving policy？**

形式化：

\[
U_{test}\notin U_{train}
\]

\[
M_{test}\notin M_{train}
\]

\[
F_{test}\in F_{train}
\]

这是：

\[
\boxed{\text{SUT OOD + Geometry OOD + Few-shot Adaptation}}
\]

如果这一层成立，已经能有力证明：

> 场景搜索策略不再依赖于一个固定 SUT 和固定道路实例。

---

# 43. Functional Scenario OOD 应作为更强扩展

只有在以下条件满足后：

```text
Universal candidate representation
Universal x0 coordinates
MoE router
shared Inner SAC
multi-geometry training
```

再验证：

\[
F_{test}\notin F_{train}
\]

推荐 leave-one-family-out：

```text
Fold A:
train cutin + roundabout
test merge

Fold B:
train merge + roundabout
test cutin

Fold C:
train merge + cutin
test roundabout
```

若三种 family 数量太少，可以之后增加 `intersection`，而不是一开始就扩场景 zoo。

---

# 44. 最小正式实验集合

架构改完后，主实验仍保持克制。

## 方法

1. Random / low-discrepancy initial-condition search
2. Single-task SAC / non-meta search
3. Meta search K=0
4. Meta search K=1
5. Meta search K=2
6. Meta search K=4

如果 MoE 是贡献：

7. Meta search without MoE

如果 geometry encoder 是贡献：

8. Meta search without geometry interaction encoder

只保留与贡献直接对应的消融。

---

# 45. `sac_scenario_mining` 最适合承担什么 baseline

它最适合代表：

> **传统单场景、单 SUT RL 场景挖掘需要在新的 task 上重新训练。**

但当前它是：

```text
merge-only
fixed SrS
IDM
different failure criterion
```

所以不能直接把归档百分比与新 MVR 主表横向比较。

正式比较时要：

- 同一 geometry；
- 同一 failure criteria；
- 同一 testing budget；
- 同一 SUT；
- 同一 initial-state search bounds。

不需要把 legacy SAC 扩到所有 family。

---

# 46. `pearl_learning` 最适合承担什么 baseline

它可代表：

> **经典 latent-context meta-RL，但不具备本文的 geometry-aware universal scene abstraction / hierarchical adversarial scenario definition。**

不应该成为当前主方法的一部分。

如果算力有限，可以先不跑正式 PEARL baseline，先证明：

```text
R4 joint OOD
+
K-shot gain
```

主假设成立后再决定。

---

# 47. 最终建议的论文-代码对应关系

| 论文组件 | 代码目标 |
|---|---|
| Scenario Mining Task \((U,F,M)\) | `scenario/task_spec.py` |
| Functional Scenario Adapter | `scenario/adapters/` |
| Geometry instance | `scenario/geometry.py` |
| Interaction abstraction | `scenario/interaction.py` |
| Concrete adversarial scenario | `scenario/concrete.py` |
| Topology encoder | `map/hptr_encoder.py` |
| Interaction encoder | `map/interaction_encoder.py` |
| Few-shot task inference | `context/pearl_context.py` |
| MoE router | `policy/moe_router.py` |
| Universal initial-condition policy | `policy/universal_scene_policy.py` |
| Continuous adversarial policy | `policy/adversarial_sac.py` |
| Safety oracle | `failure/` |
| Meta-training | `training/` |
| OOD protocol | `evaluation/regimes.py` |

---

# 48. 最关键的设计约束

后续改代码时必须始终满足：

### C1

模型永远不能接收：

```text
sut_ref
profile_id
controller_id
```

当前 guard 应保留。

### C2

Universal learning policy 不应接收：

```text
functional_scenario = "merge"
```

作为 one-hot shortcut。

### C3

functional adapter 可以知道场景类型，因为它属于 simulator translator，不属于 learned policy。

### C4

Outer continuous action 必须具有跨 geometry 一致物理语义：

```text
distance-to-conflict
speed
```

而不是：

```text
lane-start spawn coordinate
```

### C5

最终危险场景必须可复现：

```text
C = geometry + x0 + adversarial policy + seed
```

### C6

K-shot support 必须计入固定测试预算。

### C7

unseen geometry 必须通过 `map_hash` disjoint 明确验证，而不是只换 task 名称。

### C8

如果论文继续使用 “PEARL”，active method 必须真正使用 probabilistic context inference，并与 RL objective 建立联系。

### C9

如果论文继续使用 “MoE”，必须存在 learned Router + multiple shared experts，而不是 `family -> hard policy head`。

---

# 49. 当前最建议立即执行的工作

按优先级：

## Priority 1

**先不要改 PEARL/MoE。**

先完成：

```text
TaskSpec v2
+
multiple geometries
+
conflict-relative x0
+
R1/R2/R3/R4 split
```

因为如果 geometry distribution 都不存在，后面的 meta-RL/MoE 无法被科学验证。

## Priority 2

新增：

```text
InteractionCandidate
+
InteractionEncoder
```

让 model 真正获得跨 geometry / candidate 的统一 representation。

## Priority 3

将：

```text
family-specific scene policies
```

替换为：

```text
universal MoE scene policy
```

## Priority 4

再把：

```text
SetPosterior
```

替换/升级为：

```text
PEARL-style context inference
```

## Priority 5

最后做：

```text
joint meta fine-tuning
+
functional scenario OOD
```

---

# 50. 最终判断

当前代码**不是错误方向**。

相反，当前已经完成了较难的底层工作：

- MetaDrive execution contract；
- route/conflict-aware physical state；
- map tokenization；
- hierarchical Outer/Inner interaction；
- latent-conditioned Inner；
- fixed-budget few-shot protocol；
- failure/validity contract；
- reproducible training pipeline。

真正的问题是：

> **目前 meta-learning task distribution 仍然被设计成“不同 IDM profile × 三个固定 family map”，而不是“不同 SUT × 不同 geometry × 不同 functional scenario 的场景挖掘任务分布”。**

因此最核心的代码转变不是“换一个更复杂网络”，而是：

\[
\boxed{
\text{family-specific scenario search}
\rightarrow
\text{interaction-centric transferable scenario search}
}
\]

以及：

\[
\boxed{
\text{fixed-map held-out SUT}
\rightarrow
\text{SUT × geometry task distribution}
}
\]

在此基础上，再引入：

\[
\boxed{
\text{PEARL probabilistic task inference}
+
\text{learned MoE routing}
}
\]

才真正与原始论文 idea 对齐。

---

# 51. 建议的下一里程碑

在进行任何正式 3-seed 大实验之前，代码应达到以下验收状态：

```text
[ ] 至少每个 seen functional family 有 >=3 train geometries
[ ] test geometry_hash 与 train 完全不相交
[ ] 至少一个 held-out SUT
[ ] 至少一个不同 behavior-family SUT 测试
[ ] Outer 使用 conflict-relative initial-condition coordinates
[ ] family ID 不进入 model input
[ ] 一个 Universal Policy 能处理 2/3/4 个 candidate
[ ] 不再存在 scene_policies[parameter_space_id]
[ ] learned MoE Router 存在并可训练
[ ] active PEARL-style context 存在于 mvr/
[ ] K=0/1/2/4 仍在同一 20-episode all-in budget
[ ] evaluation 明确输出 R1/R2/R3/R4
[ ] concrete dangerous scenario 可由 manifest 回放
[ ] mvr 不依赖 pearl_learning 或 sac_scenario_mining runtime
[ ] pytest/compileall 全通过
```

达到这个状态后，再运行正式训练才具有论文意义。
