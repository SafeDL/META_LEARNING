# META_LEARNING：异构道路拓扑编码补充设计与未闭环模块修改建议

> **仓库**：`https://github.com/SafeDL/META_LEARNING`  
> **审阅基线**：`main@f85f892948d79e2be556336048091990558a0708`  
> **日期**：2026-08-21  
> **范围**：`meta_testing/` active MVR

---

# 0. 结论先行

当前代码已经完成了技术方向切换，但仍属于“architecture contract 基本建立、end-to-end semantic integration 尚未完成”的阶段。

下一阶段不建议继续增加 MoE、map-conditioned prior、更大 latent 或 information-gain objective。应优先闭合三条链路：

\[
\boxed{\pi_{scene}\rightarrow\text{真实场景配置}\rightarrow\text{真实 SUT/Adversary 初始状态}}
\]

\[
\boxed{\text{真实 trajectory}\rightarrow E_{traj}\rightarrow q(z|C_K)\rightarrow\pi_{scene},\pi_{adv}}
\]

\[
\boxed{z\text{ 改变}\rightarrow\text{Outer/Inner 决策改变}\rightarrow\text{固定预算下失效发现效率改变}}
\]

地图部分需要从当前的 **polyline + relative-pose attention** 升级为真正的：

```text
typed lane nodes
+ explicit interaction/conflict-zone nodes
+ typed relation-aware message passing
+ map-aware candidate pointer
+ vehicle-centric local map attention
```

---

# Part I. 异构道路拓扑编码器补充设计

## 1. 当前实现的主要不足

当前地图链路已经具备：

- 固定长度 lane polyline；
- 局部 XY、heading cos/sin、curvature、lane width、speed limit；
- relative-pose attention；
- successor/predecessor/left/right；
- local lane tokens `H_G`；
- global map embedding `g_G`；
- SE(2) invariance 测试；
- raw/frozen embedding cache。

但还存在六个根本缺口。

### 1.1 关系类型只“声明”，没有完整构造

当前虽然声明：

```text
successor / predecessor / left / right /
merge / split / conflict / crossing / route_membership
```

但实际 topology builder 只产生前四类。因此 `merge/split/conflict/crossing` 目前并没有真实进入模型。

### 1.2 `polyline_type` 没有进入 encoder

当前 `MapPolyline.polyline_type` 被保存，但没有 type embedding，也没有：

```text
lane semantic type
road/block type
turn direction
entry/exit role
circulating lane role
merge arm/mainline role
```

因此当前更准确地说是“同构 polyline Transformer”，还不是 heterogeneous road encoder。

### 1.3 没有显式 interaction/conflict-zone token

对于 Merge、Roundabout、Intersection，真正关键的是：

> 哪些 lane 在哪个区域产生 interaction/conflict，以及各自以什么方向进入。

当前没有 `InteractionZone` 节点，导致 Outer 和 Inner 都只能通过全局 lane token 间接推断冲突结构。

### 1.4 Inner 没有真正消费 local `H_G`

当前 Inner 更接近：

\[
[state,g_G,z,option,config]
\]

而我们需要的是：

\[
\pi_{adv}(a_t|s_t,H_G,z,o,c)
\]

即根据当前车辆位置动态查询局部 lane/conflict token。

### 1.5 Outer candidate 还是固定 categorical index

不同地图 candidate 数量可变，不能长期依赖固定 `candidate_count`。应改为对当前地图候选 token 做 masked pointer selection。

### 1.6 map hash 没覆盖完整拓扑/语义

最终 hash 应覆盖：

```text
polyline geometry
polyline semantic type
typed relations
interaction/conflict zones
schema version
tokenizer version
```

否则几何相同、语义不同的地图可能无法被严格区分。

---

# 2. 推荐的异构图结构

建议定义：

\[
\mathcal G=(V_L\cup V_Z, E)
\]

其中：

- `V_L`：Lane / Polyline Nodes；
- `V_Z`：Interaction / Conflict Zone Nodes。

第一版只实现这两类节点即可，不必立即引入 traffic light/control-element node。

---

# 3. Lane Node 设计

## 3.1 几何 point feature

保留当前 7 维：

\[
[x_{local},y_{local},\cos\Delta\theta,\sin\Delta\theta,\kappa,w,v_{limit}]
\]

可选补充：

```text
normalized longitudinal progress
left/right boundary type
```

## 3.2 Lane-level semantic feature

每条 lane 增加：

```text
lane_type
road/block_type
turn_direction
is_entry_lane
is_exit_lane
is_circulating_lane
is_merge_arm
is_mainline
is_ramp
lane_number / normalized lane index
lane_length
in_degree
out_degree
```

使用 embedding，而不是把这些 one-hot 重复拼到每一个 polyline point：

\[
h_i^0=PointPool(E_{point}(P_i))+E_{type}+E_{role}+E_{topology}
\]

---

# 4. Interaction / Conflict Zone Node

建议新增：

```python
@dataclass(frozen=True)
class InteractionZone:
    zone_id: str
    zone_type: str
    center_xy: np.ndarray
    heading: float
    extent: tuple[float, float]
    incident_lanes: tuple[int, ...]
    entry_lanes: tuple[int, ...]
    exit_lanes: tuple[int, ...]
    attributes: Mapping[str, Any]
```

`zone_type` 至少支持：

```text
merge_zone
split_zone
crossing_zone
roundabout_entry_zone
roundabout_exit_zone
lane_change_target_zone
```

其中 `lane_change_target_zone` 可以作为 scenario overlay 动态生成，而非永久 static map node。

Zone feature 建议包含：

```text
zone type embedding
center / orientation
extent
incident lane count
entry/exit count
conflict angle
```

显式 zone token 可直接支持：

- Outer 选择 conflict zone；
- Inner 计算当前车辆到 zone 的局部关系；
- trajectory encoder 计算 conflict timing；
- roundabout entry/exit 的语义区分。

---

# 5. Heterogeneous Edge 设计

## 5.1 Lane → Lane

至少实现：

```text
successor
predecessor
left_neighbor
right_neighbor
merge_into
split_to
geometric_crossing
```

### merge_into

当多个上游 lane/road 汇入同一 downstream lane/road：

\[
L_i\xrightarrow{merge\_into}L_j
\]

### split_to

单一路段分成多个 downstream branch：

\[
L_i\xrightarrow{split\_to}L_j
\]

### geometric_crossing

非直接 predecessor/successor 的 lane corridor 在物理上发生交叉时建立；必须排除连续 route 接缝。

## 5.2 Lane ↔ Zone

建立：

```text
lane_enters_zone
lane_exits_zone
lane_participates_in_zone
zone_has_lane
```

使冲突结构能通过 heterogeneous attention 显式传播。

## 5.3 Zone → Zone

MVR 第一版可以不实现。未来 multi-conflict intersection 再增加：

```text
zone_successor
zone_adjacent
```

---

# 6. Topology Builder 文件设计

建议把当前 `relations.py` 拆分为：

```text
meta_testing/map/
├── relations.py
├── topology_builder.py
├── conflict_zones.py
└── scenario_overlay.py
```

## 6.1 `relations.py`

只负责：

```text
relation schema
relation id
node type id
TypedEdge dataclass
```

## 6.2 `topology_builder.py`

负责：

1. 解析 MetaDrive `road_network.graph`；
2. successor/predecessor；
3. left/right；
4. 根据 in-degree/out-degree 和 road connectivity 推断 merge/split；
5. 读取 MetaDrive block/type 信息产生 lane semantic role；
6. 输出 typed edge list。

推荐：

```python
@dataclass(frozen=True)
class TypedEdge:
    source_type: str
    source_index: int
    relation: str
    target_type: str
    target_index: int
```

## 6.3 `conflict_zones.py`

负责：

- 非连续 lane pair 的几何相交检测；
- 使用 lane corridor / OBB envelope，而不是单点 centerline intersection；
- 聚类相邻 intersection samples；
- 计算 conflict angle；
- 标记 entry/exit lanes；
- roundabout entry/exit zone。

优先复用 legacy 已审计的 `RoutePolyline` 和 conflict geometry，不重新发明另一套冲突判定。

## 6.4 `scenario_overlay.py`

用于叠加 task-specific 信息：

```text
selected route candidate
selected conflict candidate
cut-in target lane
spawn candidate
SUT route
adversary route
```

这些信息不属于 static map，不应写入永久 map hash。

---

# 7. 新的 MapTokens Schema

建议升级为：

```python
@dataclass(frozen=True)
class HeteroMapTokens:
    schema: str
    map_hash: str
    lane_polylines: tuple[MapPolyline, ...]
    interaction_zones: tuple[InteractionZone, ...]
    lane_lane_edges: tuple[TypedEdge, ...]
    lane_zone_edges: tuple[TypedEdge, ...]
    tokenizer_version: str
```

Tensor 化后至少有：

```text
lane_points:      [N_lane, P, F_point]
lane_semantics:   [N_lane, F_lane_sem]
lane_centres:     [N_lane, 2]
lane_headings:    [N_lane]
zone_features:    [N_zone, F_zone]
zone_centres:     [N_zone, 2]
zone_headings:    [N_zone]
typed edge indices + relation ids
```

---

# 8. 真正的 Heterogeneous HPTR Encoder

当前 relation 只作为 attention bias。完整设计应至少让 relation 同时影响：

```text
attention score
message/value transformation
```

推荐：

\[
score_{ij}^{(r)}=
\frac{(W_Q^{t_i}h_i)^T(W_K^{r,t_j}h_j)}{\sqrt d}
+b_{pose}(\Delta x,\Delta y,\Delta\theta)+b_r
\]

消息：

\[
v_j^{(r)}=W_V^{r,t_j}h_j
\]

为避免 relation 参数爆炸，可使用：

```text
node-type-specific Q
shared K/V
relation embedding / FiLM modulation
shared FFN
```

而不是每种 relation 一整套 Transformer。

建议新增：

```python
class HeteroRelativeAttention(nn.Module):
    ...
```

---

# 9. Sparse Local Attention

当前 full attention 为 `N×N`。复杂 roundabout/大地图后成本会上升。

建议：

\[
\mathcal N(i)=\mathcal N_{typed-edge}(i)\cup\mathcal N_{spatial-kNN}(i)
\]

推荐：

```text
spatial KNN = 8~16
topology hop = 1~2
```

全局 `g_G` 再通过 attention pooling 获得。

---

# 10. Outer：从固定 candidate index 改为 Map Pointer

当前：

```text
Linear(256, candidate_count)
```

建议：

\[
q_{outer}=MLP[g_G,z,H_K]
\]

候选来自：

```text
candidate lane token
candidate conflict-zone token
candidate route token
```

计算：

\[
\ell_j=q_{outer}^{T}W_ch_j^{candidate}
\]

然后：

\[
p(j)=MaskedSoftmax(\ell_j)
\]

新增：

```text
meta_testing/policy/candidate_pointer.py
```

接口：

```python
class CandidatePointer(nn.Module):
    def forward(self, query, candidate_tokens, candidate_mask): ...
```

这样同一 Outer 可以自然处理不同地图中 2/3/6/... 个合法候选。

---

# 11. Inner：动态查询局部地图

新增：

```text
meta_testing/policy/local_map_attention.py
```

先形成 query：

\[
q_t=MLP[s_t,z,o,c]
\]

再做：

\[
h_t^{map}=CrossAttn(q_t,H_{lane}\cup H_{zone})
\]

mask 只保留：

- adversary 周围 lane；
- SUT 周围 lane；
- 当前选定 conflict zone；
- 当前 routes 前方若干 token。

最终：

\[
f_t=MLP[s_t,h_t^{map},g_G,z,o,c]
\]

再进入 SAC。

必须增加测试：在车辆瞬时 state 相同但前方 topology 不同的情况下，局部 map 表示必须不同。

---

# 12. Map Encoder 的训练策略

当前 `TrainingStage` 没有 `map_encoder`，这是 P0 问题。

如果没有外部预训练 checkpoint，地图 encoder 可能保持随机参数。

建议新增：

## MAP_PRETRAIN

结构自监督任务：

1. relation classification；
2. masked polyline reconstruction；
3. conflict-zone membership prediction。

这一步不需要 SUT rollout。

## INNER_PRETRAIN

允许：

```text
map_encoder (low LR)
local_map_attention
inner_feature_encoder
inner_sac
```

一起训练。

建议地图 encoder LR 为 policy LR 的 `0.1~0.3`。

## POSTERIOR / OUTER

先冻结 map encoder，减少 representation drift。

## JOINT

只做 very slow update 或继续冻结，作为消融。

---

# 13. Map Hash / Cache 升级

## map hash

必须包含：

```text
schema
tokenizer version
lane geometry
lane semantic type
lane-lane typed edges
interaction-zone geometry/type
lane-zone edges
```

## embedding cache

当前只按 `map_hash` 保存 embedding 不够。

增加：

```text
encoder_fingerprint
```

例如由：

```text
architecture schema
checkpoint state hash
embedding dim
relation schema
```

共同 hash。

缓存建议：

```text
embeddings/{map_hash}/{encoder_fingerprint}.pt
```

---

# 14. 地图 Gate：G1 应拆成 G1a–G1f

## G1a Topology Completeness

golden map 的 typed edges 与人工预期完全一致。

## G1b Conflict-Zone Correctness

检查：

```text
incident lanes
entry lanes
exit lanes
conflict angle
zone center/extent
```

## G1c SE(2) Invariance

保留现有测试。

## G1d Relation Sensitivity

保持几何相同，只改变 relation type，要求 token 输出有显著变化。

## G1e Candidate Pointer Validity

非法 candidate：

\[
P=0
\]

## G1f Local Map Utility

不同局部 topology 必须产生不同 Inner local-map context。

只有 G1a–G1f 全部通过，才建议称为：

> **heterogeneous topology-aware map encoder**

---

# Part II. 当前未改进彻底的地方

# 15. P0：Outer action 必须真正改变物理仿真

## 当前问题

`ParameterSpace.decode()` 已产生：

```text
route_or_conflict_candidate
option
adversary_spawn_m
sut_spawn_m
adversary_initial_speed_mps
sut_initial_speed_mps
```

但三个 family adapter 主要仍固定创建：

```text
Merge      -> SrS
Cut-in     -> S
Roundabout -> O
```

当前 runtime validation 主要检查 config dictionary 被记录，不等价于 config 被物理应用。

## 修改

新增：

```python
@dataclass
class AppliedScenario:
    adversary_vehicle_id: str
    sut_vehicle_id: str
    adversary_lane: LaneIndex
    sut_lane: LaneIndex
    adversary_spawn_m: float
    sut_spawn_m: float
    adversary_speed_mps: float
    sut_speed_mps: float
    selected_candidate: str
    selected_option: str
```

`ScenarioExecutor.reset()` 返回：

```text
env
initial_observation
AppliedScenario
```

并逐项验证实际值与 Outer 输出一致。

新增测试：

```text
test_outer_spawn_applied_exactly.py
test_outer_initial_speed_applied_exactly.py
test_outer_route_candidate_applied.py
test_family_candidate_mask.py
```

---

# 16. P0：SUTAdapter 必须进入真实 rollout

当前已经有：

```text
SUTRegistry
IDMSUTAdapter
RuleBasedSUTAdapter
ExternalSUTAdapter
spawn_sut()
```

但主 rollout 仍未形成明确链：

```text
task.sut_ref
→ registry.create()
→ spawn_sut()
→ attach controller
→ simulate
```

建议 `ScenarioExecutor.reset()` 显式接入 SUTRegistry，并返回：

```python
@dataclass
class ExecutableEpisode:
    env: Any
    adversary: Any
    sut: Any
    sut_adapter: SUTAdapter
    sut_profile: ControllerProfile
    applied_scenario: AppliedScenario
    map_tokens: HeteroMapTokens
```

runner 不再依赖隐式角色约定。

---

# 17. P0：新增真正的模型级 wrapper

建议新增：

```text
meta_testing/model.py
```

结构：

```python
class HierarchicalMetaTester(nn.Module):
    map_encoder
    trajectory_encoder
    episode_token_builder
    posterior
    outcome_decoder
    outer_state_encoder
    scene_policy
    candidate_pointer
    local_map_attention
    inner_feature_encoder
    inner_sac
```

统一提供：

```text
encode_map()
infer_posterior()
select_scene()
act_inner()
```

所有 training/evaluation 都只调用这一套模型数据契约。

---

# 18. P0：Trajectory Feature Extractor 必须实现

当前 `TrajectoryEncoder` 要求 12 维：

```text
relative_x
relative_y
relative_speed
sut_acceleration
sut_speed
sut_lateral_offset
adversary_progress
sut_progress
ttc
pet
pair_distance
conflict_timing
```

但 runner 只保存 raw observation/action/info。

新增：

```text
meta_testing/context/trajectory_features.py
```

接口：

```python
class TrajectoryFeatureExtractor:
    def reset(...): ...
    def step(env, adversary, sut, info) -> np.ndarray: ...
    def finalize() -> Tensor: ...
```

其中 route progress、TTC、PET、conflict timing 优先复用 legacy 审计后的 route/critical geometry。

---

# 19. P0：Posterior 训练必须防 outcome leakage

当前 EpisodeToken 包含 outcome，同时 decoder 又预测 outcome。

必须使用：

```text
support episodes → z
held-out target episode → target outcome
```

不能让同一个 episode 的 outcome 同时进入 support token 又成为自己的 decoder target。

新增：

```python
@dataclass
class PosteriorTrainingBatch:
    support_tokens
    support_mask
    support_episode_ids
    target_episode_id
    target_map
    target_config
    target_option
    target_outcome
```

trainer 必须 assert：

```text
target_episode_id not in support_episode_ids
```

---

# 20. P0：明确 5 维 outcome schema

当前 decoder 固定输出 5 维，并对 5 维全部使用 BCE，但没有明确字段定义。

推荐：

```text
failure_logit                  binary
invalid_logit                  binary
normalized_min_ttc             continuous
normalized_min_distance        continuous
normalized_max_closing_speed   continuous
```

损失：

\[
L=L_{BCE}^{failure}+L_{BCE}^{invalid}+\lambda_s L_{Huber}^{severity}+\beta KL
\]

新增：

```text
meta_testing/context/outcome_schema.py
```

---

# 21. P0：Outer PPO 不能使用普通长期 Replay

当前 `OuterReplay.sample()` 可随机抽历史 transition，而 Outer 配置为 PPO。

PPO 需要 on-policy / near-on-policy rollout，至少保存：

```text
old_logprob
value
reward
done
advantage
return_target
policy_version
```

建议用：

```text
PPORolloutBuffer
```

流程：

```text
collect fresh outer episodes
→ compute GAE
→ PPO epochs
→ discard buffer
```

Inner SAC 继续使用长期 off-policy replay。

---

# 22. P0：TrainingStage 必须加入 Map/Shared 模块

当前 stage 中没有：

```text
map_encoder
local_map_attention
shared_feature_encoder
```

建议组件统一注册：

```text
map_encoder
trajectory_encoder
episode_token_builder
posterior
outcome_decoder
outer_state_encoder
outer_policy
candidate_pointer
local_map_attention
inner_feature_encoder
inner_sac
```

建议阶段：

```text
MAP_PRETRAIN
INNER_PRETRAIN
POSTERIOR
OUTER
JOINT
```

---

# 23. P0：Inner option 必须有行为实现 reward

Inner 虽接收 option，但当前主要使用 `env.step()` reward，policy 可能完全忽略 option。

建议增加：

```text
meta_testing/policy/option_reward.py
```

例如：

### APPROACH_CONFLICT

奖励降低到目标 conflict zone 的 arrival time，并保持 route validity。

### YIELD_THEN_PRESS

早期保持 gap，触发条件后快速缩小 gap。

### GAP_CLOSE

奖励降低 longitudinal/arrival-time gap。

### ROUTE_BLOCK

奖励在合法 route 内占据 SUT 目标 interaction region。

G4 不只看 action 差异，而要验证：

\[
P(option\ realized|o)\ge0.8
\]

---

# 24. P0：Outer state 必须包含 history/coverage

Outer reward 有 novelty，但当前没有真正实现 `H_K`。

新增：

```text
meta_testing/policy/outer_state.py
```

建议：

\[
H_K=[
\text{budget fraction},
\text{valid failure count},
\text{unique failure count},
\text{max/mean severity},
\text{recent failure rate},
\text{posterior uncertainty},
\text{coverage summary}
]
\]

第一版 coverage 可以使用：

```text
candidate visitation counts
option visitation counts
severity histogram
failure-signature histogram
```

Outer 输入：

\[
u_K=[g_G,z_K,H_K]
\]

---

# 25. P0：Fixed-budget evaluator 必须真正执行 adaptation

当前 evaluator 主要是执行 B 次 episode 然后统计结果，不等价于 K-shot adaptation。

正式流程：

```python
for K in [0,1,2,4]:
    reset unseen SUT task
    z = prior
    context = []

    for episode in range(B):
        z = posterior(context) if context else prior
        scene = outer(map, z, history)
        rollout = execute(scene, z)
        context.append(make_token(rollout))
        metrics.add(rollout.signature)
```

若做 frozen-K 对照：前 K 个 episode 形成 posterior，之后 posterior freeze；但前 K 个 episode 必须计入总预算。

---

# 26. P0：G1–G8 应自己执行实验，而不是只读取 scalar JSON

建议每个 Gate 拆成：

```text
run.py
metrics.py
decision.py
```

例如 G3：

```text
same map/config samples
× SUT A/B
× seeds
→ raw outcomes
→ within/cross-SUT metrics
→ transfer matrix
→ decision
```

Gate 产物必须保存 raw measurements 和 provenance，而不是只保存一个手工 scalar。

---

# 27. P1：当前 SUT 仍是 controller-profile POC

当前 IDM + rule-based profiles 可以用于工程 POC，但不足以支持：

> unseen autonomous-driving algorithm adaptation

正式实验至少加入：

```text
PPO policy
SAC policy
imitation/BC policy
不同 planner-controller stack
external black-box controller
```

论文应严格区分：

```text
unseen controller profile
vs
unseen algorithm family
```

---

# 28. P1：Failure novelty 不应只依赖离散 signature hash

当前 signature 基于：

```text
failure type
family
conflict zone
severity bins
```

建议改成 hierarchical novelty：

```text
Level 1: failure type / conflict zone
Level 2: severity cluster
Level 3: trajectory behavior embedding distance
```

减少“行为不同但同 bin”以及“几乎一样但跨 bin”的误判。

---

# 29. P1：Runtime map hash 必须实际校验

每次 environment 构建后：

```text
tokenize(env.current_map)
→ map_hash_actual
```

必须：

\[
map\_hash_{actual}=task.map\_hash
\]

否则 TaskSpec 和真实执行地图可能不一致。

---

# 30. P1：Posterior → policy 的梯度策略

建议第一版：

### Posterior stage

正常训练 `trajectory encoder + posterior + outcome decoder`。

### Inner/Outer stage

使用：

```python
z_policy = z.detach()
```

先让 policy 学会使用稳定 latent。

### Joint stage

再做：

```text
A: z detached
B: critic/RL gradients → posterior
```

消融。

---

# 31. P1：Outer candidate family mask

在完整 Candidate Pointer 前，至少实现：

```text
candidate_mask [B,M_max]
```

例如：

```text
Merge:      [1,1,0]
Cut-in:     [1,1,0]
Roundabout: [1,1,1]
```

禁止非法 candidate 获得概率。

---

# 32. P1：增加真实训练入口

建议：

```text
meta_testing/scripts/
├── pretrain_map.py
├── train_inner.py
├── train_posterior.py
├── train_outer.py
├── train_joint.py
└── evaluate_fixed_budget.py
```

每个入口统一：

- config schema；
- `HierarchicalMetaTester`；
- stage checkpoint；
- trainable/frozen manifest；
- seed/hash/provenance。

---

# 33. 推荐的完整端到端数据流

```text
MetaTestTaskSpec
      │
      ├── scenario_family / map_id
      └── sut_ref (runner-only)
      │
      ▼
ScenarioExecutor + SUTRegistry
      │
      ▼
真实 Map + SUT + Adversary
      │
      ▼
Hetero Map Tokenizer
      │
      ▼
Map Encoder
      ├── H_lane
      ├── H_zone
      └── g_map
      │
      ▼
SetPosterior(C_K)
      │
      └── z_K
      │
      ▼
OuterStateEncoder(g_map,z_K,H_K)
      │
      ▼
CandidatePointer + Continuous Head + Option Head
      │
      ├── candidate
      ├── scene config
      └── option
      │
      ▼
真实 apply 到 simulator
      │
      ▼
Inner LocalMapAttention
      │
      ▼
Inner SAC
      │
      ▼
Full rollout
      ├── inner replay
      ├── trajectory feature extractor
      ├── failure signature
      ├── outer reward
      └── context token
      │
      ▼
C_{K+1}
```

---

# 34. 修改优先级

## P0-A：地图拓扑完整性

```text
typed topology builder
conflict/interaction zones
heterogeneous attention
candidate pointer
local map attention
map encoder training stage
```

## P0-B：物理执行闭环

```text
Outer config → real spawn/speed/route
SUTAdapter → real SUT
runtime map hash
role validation
```

## P0-C：Meta-learning 闭环

```text
trajectory feature extractor
support/target leakage guard
explicit outcome schema
posterior → Outer/Inner
```

## P0-D：训练契约修复

```text
PPO on-policy buffer
map/shared module training
option reward
outer history state
```

## P0-E：真实评估

```text
adaptive fixed-budget evaluator
G1–G8 executable experiments
```

## P1

```text
heterogeneous SUT algorithms
behavior-based novelty
posterior gradient ablations
structural map pretraining
```

## P2

只有 P0/P1 稳定后再考虑：

```text
information-gain objective
map-conditioned prior
MoE
unseen scenario-family claim
multi-adversary
full external ADS integration
```

---

# 35. 下一轮提交的最小验收标准

下一次提交不应看“新增文件数量”，而应通过一个最小真实闭环。

## Test 1：Outer 真实作用于环境

\[
(g_G,z_0,H_0)\rightarrow(c_1,o_1)
\]

确认 spawn/speed/route/candidate 全部真实进入 simulator。

## Test 2：Inner 真正 map-/option-conditioned

\[
(s_t,H_G,z_0,o_1,c_1)\rightarrow a_t
\]

验证：

```text
不同 option → 可控行为不同
不同局部 topology → local map context 不同
```

## Test 3：真实 trajectory 产生 posterior

\[
\tau_1\rightarrow x_{1:T}^{12D}\rightarrow e_1\rightarrow q(z|\tau_1)=z_1
\]

## Test 4：第二次测试发生 adaptation

\[
(g_G,z_1,H_1)\rightarrow(c_2,o_2)
\]

且 posterior intervention 下：

\[
\pi_{scene}(\cdot|z_1)\neq\pi_{scene}(\cdot|z_0)
\]

和/或：

\[
\pi_{adv}(\cdot|z_1)\neq\pi_{adv}(\cdot|z_0)
\]

## Test 5：固定预算盈利

在 `B=20` 下比较：

```text
random
BO/CEM
zero-z
swapped-z
correct-z
full method
```

只有：

\[
AUC_{full}>AUC_{baseline}
\]

并且 probe 全计入预算，才能进入正式性能结论。

---

# 36. 最终判断

当前路线不需要推翻。

地图部分应从：

```text
polyline + relative-pose Transformer
```

提升为：

```text
typed lane nodes
+ explicit interaction/conflict-zone nodes
+ relation-conditioned attention/message
+ map candidate pointer
+ vehicle-centric local map attention
```

其余代码则应优先完成：

```text
定义好的模块
→ 真实 simulator 数据流
→ 可训练闭环
→ 可执行 Gate
→ fixed-budget few-shot 证据
```

整个下一阶段最重要的科学条件仍是：

\[
\boxed{
\text{Map structurally correct}
\land
\text{SUTs distinguishable}
\land
z\text{ causally changes decisions}
\land
\text{all-in budget profitable}
}
\]

任一条件不成立，都不应通过继续堆网络复杂度来补救。
