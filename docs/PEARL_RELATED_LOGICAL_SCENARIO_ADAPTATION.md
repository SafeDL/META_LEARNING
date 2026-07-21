# 第二阶段实现目标：PEARL 用于相关逻辑场景间的少样本快速适配

> **文档用途**：作为 Codex 实现第二阶段代码的工程与实验规格。  
> **目标仓库**：`SafeDL/META_LEARNING`  
> **阶段编号**：Stage 2 / P1  
> **当前基线**：`sac_scenario_mining/` 中已经完成的 MetaDrive `on_ramp_merge + SAC`。  
> **核心研究问题**：面对一个训练时未见、但与元训练场景共享交通交互结构的逻辑场景模板，PEARL 能否只使用少量完整场景交互，比统一多任务 SAC 和等预算从头训练更快地生成有效安全关键场景？  
> **Task 的主要来源**：道路拓扑、冲突区结构、参与者路线和通行权逻辑的变化。  
> **明确不作为主要 Task 来源**：被测 ADS/SUT 的内部参数变化。  
> **本阶段不包含**：MoE、RSS、外部 ADS、视觉输入、完全无关场景簇的大规模统一训练。

---

## 0. Codex 执行约束

在开始第二阶段实现前，必须先阅读并理解：

- `README.md`
- `docs/SAC_METADRIVE_SCENARIO_MINING.md`
- `sac_scenario_mining/configs/merge_sac.yaml`
- `sac_scenario_mining/src/env.py`
- `sac_scenario_mining/src/metadrive_compat.py`
- `sac_scenario_mining/src/casebook.py`
- `sac_scenario_mining/src/observation.py`
- `sac_scenario_mining/src/reward.py`
- `sac_scenario_mining/src/metrics.py`
- `sac_scenario_mining/scripts/train_sac.py`
- `sac_scenario_mining/scripts/evaluate.py`
- `ref_code/metadrive-scenario/metadrive_scenario/examples/run_scenarios.py`
- `ref_code/metadrive-scenario/metadrive_scenario/utils/env_create_utils.py`
- PEARL 官方参考实现 `oyster` 中的：
  - `PEARLAgent`
  - context posterior inference
  - task-indexed replay buffer
  - PEARL-SAC update

实施时必须遵守：

1. **Task 必须表示逻辑场景模板，而不是隐藏的 SUT 参数。**
2. 所有正式 task 使用同一类、同一参数配置的固定 SUT。
3. `task_id`、场景类型名称和 one-hot topology label 不得直接输入 Actor、Critic 或 Context Encoder。
4. 允许输入通用、连续、可解释的道路和冲突结构特征，但不允许输入“on_ramp / bottleneck / y_merge”等类别标签。
5. 本阶段只处理具有共享交互机制的 `merge-like` 逻辑场景族。
6. 不在第二阶段直接混合 roundabout、crossing、pedestrian、parking 等明显不同的逻辑场景簇。
7. 本阶段不实现 MoE；单一 PEARL Actor 的能力边界是本阶段需要测量的结果。
8. 不修改和覆盖第一阶段正式结果目录。
9. 对公共环境代码的任何修改都必须通过 Stage 1 回归测试。
10. `ref_code/metadrive-scenario` 只读借鉴，不直接修改，也不强制下载大型数据集。
11. 主实验必须使用真实 MetaDrive，不得静默回退到 toy environment。
12. toy task 只允许用于 PEARL 数学模块的单元测试，不能计入研究结果。
13. 先完成多逻辑场景环境、统一观测和异质性审计，再实现完整 PEARL。
14. 不得通过给 PEARL 更多 query 数据、更多环境步或未限制的梯度更新制造不公平优势。
15. 元测试时只更新任务后验 `q(z|C)`，不得更新网络权重。
16. 所有 taskbook、casebook、support/query split 和随机种子必须可保存、可哈希、可回放。
17. Codex 不得为通过验收硬编码地图、策略动作或测试成功结果。

---

## 1. 第一阶段结论与第二阶段定位

### 1.1 第一阶段已经支持的结论

当前第一阶段可以作为：

> **单一 `on_ramp_merge` 逻辑任务中的工程可行性与 SAC 场景挖掘有效性初步结果。**

它已经说明：

1. MetaDrive 中 adversary 和 SUT 的固定角色可以稳定建立；
2. SAC 可以持续控制入口匝道 adversary；
3. 安全关键场景可以按 case 保存、评估和回放；
4. SAC 在 held-out initial-condition cases 上明显优于随机连续动作；
5. 多个优化随机种子得到一致结论。

### 1.2 第一阶段尚未支持的结论

第一阶段不能证明：

1. 对未见逻辑场景的适配；
2. 对不同道路拓扑的迁移；
3. 对不同冲突区结构的泛化；
4. PEARL 的 few-shot 价值；
5. MoE 的场景簇专门化；
6. RSS 责任有效性；
7. 对真实 ADS 或真实道路的泛化。

### 1.3 第二阶段的正确目标

第二阶段不是：

```text
固定 on-ramp 拓扑
+ 改变 SUT/ADS 内部参数
+ 推断驾驶风格
```

第二阶段应当是：

```text
固定 SUT
+ 多个相关 merge-like 逻辑场景
+ 对未见逻辑场景模板进行 K-shot 适配
```

---

## 2. Scenario Family、Task 与 Case 的定义

这是第二阶段最重要的概念约束。

### 2.1 Scenario Family

`scenario_family` 表示共享核心交互机制的逻辑场景簇。

第二阶段固定：

```text
scenario_family = merge_like
```

共享的交互原语包括：

- conflict-point arrival timing
- gap acceptance
- yield / non-yield competition
- merging pressure
- relative-speed control
- merge-window selection

### 2.2 Task

一个 PEARL task 是一个具体逻辑场景模板：

\[
T_i =
(G_i,\; Z_i,\; U_i,\; \Gamma_i,\; \pi_{\mathrm{SUT}},\; R)
\]

其中：

- \(G_i\)：道路和车道连接图；
- \(Z_i\)：冲突区、汇入区和关键几何结构；
- \(U_i\)：adversary 与 SUT 的路线及语义角色；
- \(\Gamma_i\)：通行权、让行关系和场景规则；
- \(\pi_{\mathrm{SUT}}\)：固定的 SUT 策略；
- \(R\)：跨任务统一的危险场景奖励。

Task 的差异来源包括：

- on-ramp 汇入；
- lane-drop 汇入；
- bottleneck 汇入；
- Y-shaped merge；
- 可选 zipper merge；
- 不同 merge 长度；
- 不同入口角度；
- 不同车道连接图；
- 不同角色路线；
- 不同优先权结构。

Task **不以 SUT 内部参数为主要差异来源**。

### 2.3 Case

一个 case 是 task 下的一次具体初始条件实例：

\[
c_{ij} =
(x^{\mathrm{adv}}_0,\;
x^{\mathrm{SUT}}_0,\;
x^{\mathrm{bg}}_0,\;
\rho,\;
seed)
\]

它可以改变：

- adversary 初始速度；
- SUT 初始速度；
- 两车初始间隔；
- 出生位置；
- 背景车密度；
- 背景车状态；
- policy seed。

因此：

```text
Task = 逻辑场景模板
Case = 该模板中的一次参数化实例
```

### 2.4 推荐层次结构

```text
Merge-like Scenario Family
├── Task: on_ramp_merge_A
│   ├── support case 000
│   ├── support case 001
│   ├── query case 000
│   └── ...
├── Task: lane_drop_merge_A
│   ├── support cases
│   └── query cases
├── Task: bottleneck_merge_A
│   ├── support cases
│   └── query cases
└── Task: y_merge_A
    ├── support cases
    └── query cases
```

---

## 3. 第二阶段科学假设

主假设：

> 在统一 SUT、统一动作空间和统一奖励下，不同但相关的 merge-like 逻辑场景需要不同的危险场景生成策略。PEARL 能利用元训练中学到的共享交互知识，在未见逻辑场景模板上通过少量支持集交互推断任务潜变量，并比统一多任务 SAC 和等预算从头训练更快地挖掘有效安全关键场景。

PEARL 后验：

\[
z_i \sim q_\phi(z \mid C_i)
\]

Context：

\[
C_i =
\{(o_t,a_t,r_t,o_{t+1},d_t)\}
\]

条件策略：

\[
a_t \sim \pi_\theta(a_t \mid o_t,z_i)
\]

在本阶段中，\(z_i\) 应主要表达：

- 当前场景的交互逻辑；
- 冲突区风险模式；
- 哪种动作序列在该逻辑场景中更有效；
- 到达冲突点时序是否比横向位置更关键；
- 需要优先控制速度、间隙还是汇入时机；
- 当前场景中高风险但有效的控制模式。

---

## 4. 一个关键方法学问题：拓扑可见，PEARL 仍然必须证明必要性

逻辑场景拓扑通常是可观测的。

因此第二阶段不能假设：

```text
PEARL 的唯一作用是猜出地图类型
```

道路结构应通过通用拓扑/冲突特征直接进入 observation。

建议结构：

\[
h_G = f_{\mathrm{topology}}(G)
\]

\[
z = q_\phi(z \mid C)
\]

\[
a \sim \pi_\theta(a \mid h_{\mathrm{dynamic}},h_G,z)
\]

其中：

- \(h_G\)：回答“当前场景的道路和冲突结构是什么”；
- \(z\)：回答“在这个场景中，怎样挖掘风险更有效”。

必须使用强基线：

```text
Topology-conditioned Multi-task SAC
```

如果它在未见逻辑场景上已经零样本饱和，则 PEARL 没有必要性。

---

## 5. 第二阶段范围

### 5.1 必须完成

- 至少 3 种相关 merge-like 逻辑场景；
- 一个完全 held-out 的 merge-like 逻辑子类型；
- 通用场景适配器接口；
- 通用 conflict-centric observation；
- 固定 SUT；
- 统一动作与奖励；
- taskbook 与 casebook；
- per-task SAC；
- topology-conditioned pooled SAC；
- PEARL；
- scratch SAC；
- support/query few-shot 评估；
- Stage 1 回归；
- 关键场景回放；
- 结果审计。

### 5.2 不做

- SUT 参数变化作为主要 task；
- MoE；
- RSS；
- 外部 ADS；
- RGB 或视觉模型；
- roundabout；
- signalized intersection；
- pedestrian crossing；
- parking；
- 多个完全不相关场景簇；
- 多 adversary 协同；
- 真实数据大规模训练；
- 从零生成完整轨迹和地图。

### 5.3 可选扩展

- 使用 `metadrive-scenario` synthetic dataset 增加同类模板；
- 参数化场景编辑动作；
- OOD 几何外推；
- 更多 SUT 类型作为附加鲁棒性实验；
- Context Encoder 的 recurrent 版本。

---

## 6. 第一阶段冻结与回归保护

在修改公共环境代码前：

1. 创建 Stage 1 Git tag：

```text
stage1-on-ramp-sac-v1
```

2. 保留：

```text
results/sac_scenario_mining/
```

3. 新结果写入：

```text
results/pearl_learning/
```

4. Stage 2 环境不能破坏：

```text
on_ramp_merge_obs_v2
```

5. 若新增通用 observation，使用新的 schema：

```text
logical_merge_obs_v1
```

6. 旧 manifest 和旧 action trace 必须继续可回放。

7. Stage 1 正式模型不要求重新训练，但其测试和回放审计必须通过。

---

## 7. 进入第二阶段前补做的 Stage 1 审计

### 7.1 Pairwise target collision

不能仅依赖：

```python
adversary.crash_vehicle and sut.crash_vehicle
```

新增：

```python
def pairwise_target_contact(env, adversary, sut) -> bool:
    ...
```

优先级：

1. 物理引擎 contact pair；
2. MetaDrive collision object；
3. 双 crash flag + 两车包围盒接触；
4. 记录 fallback 方法。

结果记录：

```text
target_contact_method
```

### 7.2 Strict validity

同时保留：

```text
valid_critical_legacy
valid_critical_strict
```

正式 Stage 2 使用：

```text
valid_critical_strict =
    critical
    and no invalid event before or at termination
```

无效事件包括：

- adversary 出界；
- SUT 出界；
- 逆行；
- 非目标碰撞；
- 目标角色丢失；
- 不合法路线；
- 仿真异常。

### 7.3 更强的非学习基线

至少增加：

- `zero_action`
- `smooth_random`
- `constant_throttle`
- `brake_then_accelerate`

可选：

- grid search
- CEM

---

## 8. 逻辑场景拓扑审计

在正式编码所有 Adapter 前，先新增：

```text
python -m pearl_learning.scripts.audit_topologies
```

该脚本必须：

1. 枚举 MetaDrive 当前版本可用的候选程序化地图；
2. 渲染 top-down 图；
3. 输出 lane graph；
4. 输出 lane index；
5. 标记候选 conflict point；
6. 验证 adversary 和 SUT 路线；
7. 检查两条路线是否产生真实 merge interaction；
8. 检查角色是否可以稳定 spawn；
9. 检查相同 task/case 能否回放；
10. 生成 `topology_audit.json`。

不得在未验证地图几何的情况下，凭猜测硬编码 lane ID。

---

## 9. 推荐的 merge-like 逻辑场景集合

### 9.1 必须实现的类型

建议：

1. `on_ramp_merge`
2. `lane_drop_merge`
3. `bottleneck_merge`
4. `y_merge`

如果 MetaDrive 当前程序化 block 无法可靠构建某一类型，可使用功能等价、仍属于 merge-like 的模板替代，但必须在文档中说明。

### 9.2 Stage 2A：模板级测试

用于工程调试和基本泛化验证：

```text
训练：
- on_ramp variants
- lane_drop variants
- bottleneck variants

测试：
- 上述类型中的 held-out geometry templates
```

这验证：

```text
unseen template adaptation
```

### 9.3 Stage 2B：逻辑子类型级测试

这是跨逻辑场景 motivation 的主结果：

```text
Meta-train:
- on_ramp_merge
- lane_drop_merge
- bottleneck_merge

Meta-test:
- y_merge
```

`y_merge` 整个逻辑子类型不得参加：

- 元训练；
- checkpoint 选择；
- 超参数调节。

只有 Stage 2B 成立，才能主张：

```text
few-shot adaptation to an unseen logical scenario type
```

---

## 10. LogicalScenarioTaskSpec

新增：

```text
pearl_learning/src/task_spec.py
```

建议数据结构：

```python
@dataclass(frozen=True)
class LogicalScenarioTaskSpec:
    task_id: str
    split: str
    family: str
    logical_type: str
    map_source: str
    map_config: dict
    conflict_spec: dict
    adversary_route: dict
    sut_route: dict
    priority_spec: dict
    spawn_regions: dict
    case_seed: int
    schema_version: str = "logical_merge_task_v1"
```

要求：

- `logical_type` 只用于环境构建、日志和分组；
- 不得进入 Actor、Critic 或 Context Encoder；
- task spec 必须可 JSON 序列化；
- taskbook 必须有 hash；
- taskbook 一旦用于正式测试，不得修改。

### 10.1 示例

```json
{
  "schema_version": "logical_merge_task_v1",
  "task_id": "meta_train_on_ramp_00",
  "split": "meta_train",
  "family": "merge_like",
  "logical_type": "on_ramp_merge",
  "map_source": "procedural",
  "map_config": {
    "template": "verified_on_ramp_template_00"
  },
  "conflict_spec": {
    "conflict_point": [0.0, 0.0],
    "conflict_radius_m": 6.0,
    "merge_length_m": 42.0
  },
  "adversary_route": {
    "route_id": "ramp_to_mainline"
  },
  "sut_route": {
    "route_id": "mainline_through"
  },
  "priority_spec": {
    "sut_has_priority": true
  },
  "spawn_regions": {
    "adversary": [0.0, 8.0],
    "sut": [8.0, 20.0]
  },
  "case_seed": 10001
}
```

---

## 11. Taskbook 与划分

新增：

```text
pearl_learning/src/taskbook.py
```

### 11.1 推荐任务数量

MVP 建议：

| Split | 数量 | 内容 |
|---|---:|---|
| meta-train | 12 | 3 种 seen logical types，每种 4 个模板 |
| meta-validation | 3 | 每种 seen type 1 个 held-out 模板 |
| meta-test-template | 3 | seen logical types 的新模板 |
| meta-test-logical | 4 | 完全 held-out 的 `y_merge` 模板 |

### 11.2 主结果划分

论文主结果使用：

```text
meta-test-logical
```

`meta-test-template` 只用于：

- 调试；
- ID 泛化；
- 检查 PEARL 实现是否能够工作。

### 11.3 每个 task 的 case

建议：

```yaml
cases_per_task:
  train_pool: 32
  validation_support: 10
  validation_query: 20
  test_support: 10
  test_query: 20
```

要求：

- support 和 query 不重叠；
- case 参数空间跨 task 尽量统一；
- 每个 task 使用同一 SUT；
- case 只改变初始条件；
- case 不能改变逻辑拓扑；
- support/query case 表保存为 JSON。

---

## 12. LogicalScenarioAdapter 接口

新增：

```text
pearl_learning/src/adapters/
```

定义协议：

```python
class LogicalScenarioAdapter(Protocol):
    def build_env(
        self,
        task: LogicalScenarioTaskSpec,
        case: dict,
        config: dict,
    ) -> Any:
        ...

    def establish_roles(
        self,
        env: Any,
        task: LogicalScenarioTaskSpec,
        case: dict,
    ) -> tuple[Any, Any]:
        ...

    def conflict_frame(
        self,
        env: Any,
        task: LogicalScenarioTaskSpec,
    ) -> dict:
        ...

    def topology_features(
        self,
        env: Any,
        task: LogicalScenarioTaskSpec,
    ) -> dict[str, float]:
        ...

    def target_contact(
        self,
        env: Any,
        adversary: Any,
        sut: Any,
    ) -> bool:
        ...

    def validate_episode_roles(
        self,
        env: Any,
        adversary: Any,
        sut: Any,
    ) -> None:
        ...
```

具体实现：

```text
OnRampMergeAdapter
LaneDropMergeAdapter
BottleneckMergeAdapter
YMergeAdapter
```

所有 MetaDrive 内部 API 和 lane graph 访问必须集中在 Adapter 或 compatibility layer。

---

## 13. 统一角色和 SUT

跨 task 必须保持：

```text
adversary = 唯一由 RL 控制的车辆
SUT       = 固定规则策略控制的目标车辆
```

SUT 要求：

- 相同控制器类型；
- 相同控制器参数；
- 相同 lane-change 配置；
- 相同目标速度规则；
- task 不通过修改 SUT 参数制造差异。

允许因道路速度限制而执行相同规则下的自然速度约束，但必须记录。

---

## 14. 动作空间

第二阶段继续使用：

```text
action[0] = steering
action[1] = throttle / brake
```

空间：

```python
spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
```

理由：

- 所有 merge-like task 中物理语义一致；
- 可以复用 Stage 1 工程经验；
- 避免同时修改 task、action 和算法；
- 便于 per-task SAC 和 pooled SAC 公平比较。

每个 Adapter 必须保证：

- adversary 的目标路线合法；
- 相同动作只控制 adversary；
- SUT 不接受外部动作；
- 角色不因背景交通变化而改变。

---

## 15. 统一 conflict-centric observation

当前 `on_ramp_merge_obs_v2` 不能直接扩展到所有逻辑场景。

新增：

```text
logical_merge_obs_v1
```

建议固定 56 维。

### 15.1 Adversary conflict-frame state：8 维

```text
signed_distance_to_conflict
route_speed
route_acceleration
lateral_offset
heading_error
time_to_conflict
route_progress
on_route_flag
```

### 15.2 SUT conflict-frame state：8 维

字段同上。

### 15.3 Pairwise interaction：8 维

```text
arrival_time_difference
euclidean_distance
relative_route_speed
closing_speed
ttc
conflict_angle
adversary_priority
sut_priority
```

### 15.4 背景车辆：18 维

最多 3 辆，每辆 6 维：

```text
relative_x_in_conflict_frame
relative_y_in_conflict_frame
relative_vx
relative_vy
distance
mask
```

### 15.5 Generic topology descriptor：14 维

```text
num_incoming_branches
num_outgoing_branches
adversary_lane_count
sut_lane_count
merge_length
adversary_route_remaining
sut_route_remaining
conflict_radius
adversary_route_curvature
sut_route_curvature
adversary_speed_limit
sut_speed_limit
traffic_density
num_conflict_zones
```

总计：

```text
8 + 8 + 8 + 18 + 14 = 56
```

### 15.6 禁止的输入

不得输入：

- task ID；
- logical type one-hot；
- `on_ramp` 标签；
- `bottleneck` 标签；
- `y_merge` 标签；
- train/test split；
- query/support 标志；
- 模板文件名。

### 15.7 允许的输入

允许输入：

- 可由当前道路结构计算的连续特征；
- 冲突点和路线几何；
- 通行权标志；
- 当前动态车辆状态；
- 背景交通状态。

### 15.8 观测合同

要求：

- 所有 task 返回相同 shape；
- 所有值有限；
- 所有值规范化到 `[-1, 1]`；
- task 切换不改变字段含义；
- schema 版本写入结果；
- 单元测试逐字段验证。

---

## 16. 统一奖励与指标

Stage 2 必须跨 task 使用同一奖励结构。

\[
r_t =
w_{\mathrm{ttc}} r_{\mathrm{ttc}}
+
w_{\mathrm{prox}} r_{\mathrm{prox}}
+
w_{\mathrm{col}} I_{\mathrm{target\ collision}}
-
w_{\mathrm{invalid}} C_{\mathrm{invalid}}
-
w_a \|a_t\|^2
-
w_{\Delta a}\|a_t-a_{t-1}\|^2
\]

要求：

- 各 task 的 TTC 和距离均在 conflict-centric frame 中计算；
- collision bonus 一致；
- 无效行为惩罚一致；
- 不得为不同 task 手工调不同 reward；
- 若因 episode horizon 不同产生 return 偏差，必须统一 horizon 或使用归一化指标；
- 主结果不以 raw return 跨 task 比较。

主指标：

```text
valid_critical_strict
target_collision
critical
invalid
min_ttc
min_distance
```

---

## 17. PEARL 实现前的任务异质性门禁

新增：

```text
python -m pearl_learning.scripts.audit_task_heterogeneity
```

### 17.1 Per-task SAC

在若干 anchor tasks 上分别训练：

```text
SAC_on_ramp_A
SAC_lane_drop_A
SAC_bottleneck_A
```

交叉评价：

\[
M_{ij}
=
J(\pi_i,T_j)
\]

输出：

```text
policy_task_matrix.csv
policy_task_matrix.png
```

### 17.2 Stage 1 SAC 跨场景评估

把第一阶段 best SAC 直接评估到其他逻辑 task：

```text
stage1_nominal_sac_cross_task.csv
```

### 17.3 Topology-conditioned pooled SAC

训练：

```text
pi(a | logical_merge_obs_v1)
```

特征中包含 generic topology descriptor，但不包含 task ID。

它是 PEARL 最重要的强基线。

### 17.4 Oracle task-conditioned SAC

只作为上界：

```text
pi(a | obs, task_one_hot)
```

Oracle 可以读取 task ID，其他方法不可以。

### 17.5 Go 条件

进入正式 PEARL 前，至少满足以下两项：

1. per-task SAC 平均对角线性能比平均非对角线性能高至少 10 个百分点；
2. per-task SAC 与 pooled SAC 的平均差距至少 10 个百分点；
3. Stage 1 nominal SAC 在至少一半非 on-ramp tasks 上低于 80% 有效危险率；
4. 不同 task 的高性能动作轨迹、冲突点到达时序或速度控制模式明显不同；
5. pooled SAC 在 `meta-test-logical` 上未达到饱和。

### 17.6 No-Go

若 topology-conditioned pooled SAC 在 held-out logical type 上已经接近 100%：

1. 不要直接增加 PEARL 网络复杂度；
2. 检查是否 task 太简单；
3. 检查 observation 是否泄漏模板 ID；
4. 扩大 merge-like 逻辑结构差异；
5. 增加有效性约束；
6. 重新运行异质性审计。

---

## 18. PEARL 的作用边界

PEARL 不负责：

- 识别显式 task label；
- 生成道路拓扑；
- 解决完全无关的任务簇；
- 自动增加模型容量；
- 保证责任有效性。

PEARL 在本阶段负责：

- 从 support 交互中提取逻辑场景的任务表征；
- 更新任务 posterior；
- 让共享策略针对新逻辑场景调整控制模式；
- 减少每个新场景从头训练的环境步数。

当引入 roundabout、intersection 等差异更大的簇时，再进入 MoE 阶段。

---

## 19. `shot`、support 与 query

### 19.1 Shot

```text
1-shot = 在一个 held-out logical task 上完成 1 个完整 support episode
```

主要评估：

```text
K ∈ {0, 1, 2, 5, 10}
```

同时报告：

```text
support transitions
```

### 19.2 Support

Support episode：

- 进入 context；
- 可使用 posterior sampling；
- 不进入 query 指标；
- case 表预先冻结。

### 19.3 Query

Query episode：

- 只用于评价；
- 不进入 context；
- 不参与梯度更新；
- 每个 K 使用相同 query cases；
- 默认 deterministic action。

---

## 20. PEARL 网络结构

### 20.1 总体结构

```text
Current logical-scene observation
        │
        ├── dynamic/conflict features
        └── generic topology descriptor
                     │
                     ▼
                state feature h

Support transitions
        │
        ▼
Context Encoder q_phi(z | C)
        │
        ▼
Task latent z

[h, z] ──► Actor pi(a | h, z)
[h, a, z] ► Q1 / Q2
```

第一版不使用 MoE。

### 20.2 Context 输入

使用：

\[
c_t =
[o_t,\;a_t,\;\tilde r_t,\;o_{t+1},\;terminated_t,\;truncated_t]
\]

若 observation 为 56 维：

```text
56 + 2 + 1 + 56 + 1 + 1 = 117
```

代码不应硬编码 117，应从 environment spaces 推导。

### 20.3 Reward scaling

```text
tilde_r = reward / context_reward_scale
```

Q-learning 使用原始 reward。

### 20.4 Context Encoder

建议：

```text
context_dim → 200 → 200 → 2 * latent_dim
```

初始：

```yaml
latent_dim: 5
```

输出：

```text
mu_t
variance_t
```

使用 Product of Gaussians 聚合。

### 20.5 Prior

空 context：

\[
q(z\mid\emptyset)=\mathcal N(0,I)
\]

### 20.6 Actor

```text
input = [observation, z]
hidden = [256, 256]
output = mean, log_std
distribution = tanh-squashed Gaussian
```

### 20.7 Critic

```text
Q1 input = [observation, action, z]
Q2 input = [observation, action, z]
hidden = [256, 256]
```

使用：

- twin Q；
- target Q；
- automatic entropy tuning；
- soft target update。

### 20.8 梯度边界

- Context Encoder 由 Q loss 和 KL loss 更新；
- Actor 使用 `z.detach()`；
- alpha 不更新 Context Encoder；
- target Q 不更新 Context Encoder；
- 元测试无网络梯度。

---

## 21. PEARL 损失

### 21.1 KL

\[
L_{KL}
=
\beta
D_{KL}
\left(
q_\phi(z\mid C)
\Vert
\mathcal N(0,I)
\right)
\]

### 21.2 Critic

\[
y =
r +
\gamma(1-d)
\left[
\min_j Q_j^{target}(o',a',z)
-
\alpha\log\pi(a'\mid o',z)
\right]
\]

\[
L_Q =
(Q_1-y)^2+(Q_2-y)^2
\]

### 21.3 Encoder

\[
L_{\mathrm{encoder}}
=
L_Q+L_{KL}
\]

### 21.4 Actor

\[
L_\pi =
\mathbb E
\left[
\alpha\log\pi(a\mid o,z)
-
\min_jQ_j(o,a,z)
\right]
\]

Actor 中使用 detached \(z\)。

---

## 22. Replay Buffer

新增：

```text
pearl_learning/src/replay.py
```

结构：

```text
replay_buffers[task_id]
context_buffers[task_id]
```

存储：

```text
obs
action
reward
next_obs
terminated
truncated
task_id
case_id
episode_id
logical_type
```

`task_id` 和 `logical_type` 只用于：

- buffer 索引；
- 日志；
- 统计；
- 审计。

不得作为网络输入。

Context batch 与 RL batch 独立采样。

---

## 23. 数据收集器

新增：

```text
pearl_learning/src/collector.py
```

核心接口：

```python
rollout = collect_episode(
    env=env,
    task=task,
    case=case,
    agent=agent,
    z=z,
    deterministic=False,
)
```

模式：

```text
prior_support
posterior_support
deterministic_query
```

MVP 规定：

- 一个完整 support episode 内保持 z 不变；
- episode 结束后更新 posterior；
- query 不更新 context。

---

## 24. PEARL 元训练流程

```python
initialize context_encoder
initialize actor
initialize q1, q2
initialize target_q1, target_q2
initialize alpha
initialize task replay buffers

for task in meta_train_tasks:
    collect bootstrap episodes from prior
    store transitions in task-specific buffers

for meta_iteration in range(M):
    sampled_tasks = sample_tasks(meta_batch_size)

    for task in sampled_tasks:
        clear_context()

        z_prior = sample_prior()
        prior_episode = collect_support(task, z_prior)
        add_to_buffers(task, prior_episode)

        context = sample_context(task)
        z_post = infer_posterior(context)

        posterior_episode = collect_support(task, z_post)
        add_to_buffers(task, posterior_episode)

    for update in range(updates_per_iteration):
        task_batch = sample_tasks(meta_batch_size)

        context_batch = sample_context_per_task(task_batch)
        z = infer_posterior(context_batch)

        rl_batch = sample_rl_batch_per_task(task_batch)

        update_critics_and_context(
            rl_batch,
            z,
            q_loss + beta * kl_loss,
        )

        update_actor(
            rl_batch,
            z.detach(),
        )

        update_alpha()
        soft_update_targets()

    periodically_evaluate_meta_validation()
    save_checkpoint()
```

推荐起点：

```yaml
meta_training:
  meta_batch_size: 6
  bootstrap_episodes_per_task: 5
  prior_episodes_per_sampled_task: 1
  posterior_episodes_per_sampled_task: 1
  gradient_updates_per_iteration: 100
  rl_batch_size: 256
  context_batch_size: 128
  total_environment_steps: 1500000
  validation_interval_steps: 100000
```

---

## 25. 元测试流程

```python
freeze_all_parameters()
hash_before = parameter_hash(model)

context = []
z = prior()

evaluate_queries(k=0, z=z)

for k in range(1, 11):
    support_case = fixed_support_cases[k - 1]

    trajectory = collect_episode(
        held_out_task,
        support_case,
        z,
        deterministic=False,
    )

    context.extend(trajectory)
    z = infer_posterior(context)

    if k in {1, 2, 5, 10}:
        evaluate_queries(
            k=k,
            z=posterior_mean(z),
        )

hash_after = parameter_hash(model)
assert hash_before == hash_after
```

要求：

- 每个 task 从 prior 开始；
- task 间不共享 context；
- query 不进入 context；
- query 不做 optimizer step；
- 每个 K 使用同一 query case 表。

---

## 26. 必须实现的基线

### 26.1 非学习基线

- Random
- Smooth Random
- Zero
- Constant throttle
- Brake then accelerate

### 26.2 Stage 1 nominal SAC

将 Stage 1 best SAC 直接跨场景评估：

```text
stage1_sac_zero_shot
```

### 26.3 Per-task SAC

每个逻辑 task 单独训练。

作用：

- 性能上界参考；
- 构建 cross-task matrix；
- 证明任务异质性。

### 26.4 Topology-conditioned pooled SAC

```text
pi(a | logical_merge_obs_v1)
```

不输入 task ID。

这是 PEARL 最重要的非元学习基线。

### 26.5 Scratch SAC

在 held-out logical task 上从随机初始化开始，以相同 support 环境步预算训练。

### 26.6 Fine-tuned pooled SAC

```text
pooled SAC
+ support-data gradient fine-tuning
```

### 26.7 Oracle task-conditioned SAC

```text
pi(a | obs, task_one_hot)
```

只作为上界。

### 26.8 PEARL

报告：

```text
PEARL@0
PEARL@1
PEARL@2
PEARL@5
PEARL@10
```

### 26.9 推荐消融

- PEARL without topology descriptor
- PEARL without context
- PEARL deterministic z
- PEARL without next observation in context
- PEARL without KL
- PEARL shared buffer instead of task buffers

---

## 27. 公平比较

1. 所有方法使用相同 held-out tasks；
2. 使用相同 support cases；
3. 使用相同 query cases；
4. query 不计入适配预算；
5. 同时报告 support episodes 和 transitions；
6. Scratch/Fine-tune 的梯度更新预算固定；
7. PEARL 元测试无梯度更新；
8. query 使用 deterministic action；
9. 至少 3 个训练随机种子；
10. 按 task 进行配对统计；
11. 不得选择性删除 PEARL 表现差的 task；
12. 不得用 meta-test-logical 调超参数。

---

## 28. 主指标

每个 \(K\)：

```text
valid_critical_strict_rate@K
target_collision_rate@K
critical_rate@K
invalid_rate@K
median_min_ttc@K
median_min_distance@K
```

适配效率：

```text
adaptation_auc
episodes_to_first_valid_critical
transitions_to_first_valid_critical
support_steps_to_target_rate
```

泛化：

```text
meta_test_template performance
meta_test_logical performance
```

潜变量诊断：

```text
posterior_mean
posterior_variance
KL
z_shift_from_prior
```

---

## 29. Adaptation AUC

评估点：

```text
K = [0, 1, 2, 5, 10]
```

对：

```text
valid_critical_strict_rate
```

做归一化梯形积分。

所有方法必须使用相同 K 点。

---

## 30. 统计要求

至少输出：

- 每个 held-out task 的原始结果；
- 每个训练 seed 的结果；
- 均值和标准差；
- paired bootstrap 95% CI；
- PEARL 与 pooled SAC 的配对差值；
- PEARL 与 scratch SAC 的配对差值；
- meta-test-template 与 meta-test-logical 分开统计。

---

## 31. 第二阶段验收标准

### 31.1 工程验收

- [ ] Stage 1 回归测试通过；
- [ ] 至少 4 个 LogicalScenarioAdapter 可运行；
- [ ] 所有 task 使用同一 SUT；
- [ ] 所有 task 使用同一动作空间；
- [ ] 所有 task 使用同一 observation schema；
- [ ] 所有 task 使用同一 reward；
- [ ] task ID 未进入网络；
- [ ] support/query 严格隔离；
- [ ] task replay buffer 严格隔离；
- [ ] 空 context 对应标准正态 prior；
- [ ] posterior 随 context 更新；
- [ ] 元测试参数 hash 不变；
- [ ] held-out logical type 未参与训练和验证；
- [ ] 关键场景可回放。

### 31.2 任务设计验收

- [ ] per-task SAC cross matrix 显示任务差异；
- [ ] pooled SAC 未在全部 task 上饱和；
- [ ] Stage 1 SAC 在至少部分非 on-ramp tasks 上明显退化；
- [ ] logical_merge_obs_v1 不含 task label；
- [ ] generic topology descriptor 可以跨 task 计算；
- [ ] 每个 task 存在可行 conflict interaction；
- [ ] 每个 task 的有效事件定义一致。

### 31.3 Stage 2A 验收

在 `meta-test-template` 上，至少 2/3 个 PEARL seed 满足：

1. `PEARL@5` 比 `PEARL@0` 提升至少 10 个百分点；
2. `PEARL@5` Adaptation AUC 比 pooled SAC 高至少 10% 相对提升；
3. `PEARL@5` 在等环境步预算下优于 Scratch SAC；
4. `invalid_rate@5 <= 25%`。

### 31.4 Stage 2B 验收

在完全 held-out 的 `meta-test-logical` 上，至少 2/3 个 PEARL seed 满足：

1. `PEARL@5 > PEARL@0`；
2. `PEARL@10 > topology-conditioned pooled SAC`；
3. `PEARL@10 > Scratch SAC@相同环境步预算`；
4. Adaptation AUC 相对 pooled SAC 提升至少 10%；
5. `invalid_rate@10 <= 25%`；
6. 提升不来自非目标碰撞、出界或错误碰撞归因。

只有通过 Stage 2B，才可以使用：

```text
cross-logical-scenario few-shot adaptation
```

作为论文主张。

---

## 32. 推荐目录结构

```text
pearl_learning/
├── __init__.py
├── configs/
│   ├── merge_family_pearl.yaml
│   ├── topology_templates.yaml
│   └── baselines.yaml
├── src/
│   ├── __init__.py
│   ├── task_spec.py
│   ├── taskbook.py
│   ├── casebook.py
│   ├── task_env.py
│   ├── conflict_frame.py
│   ├── observation.py
│   ├── reward.py
│   ├── metrics.py
│   ├── replay.py
│   ├── context_encoder.py
│   ├── networks.py
│   ├── pearl_agent.py
│   ├── pearl_trainer.py
│   ├── collector.py
│   ├── evaluator.py
│   ├── checkpoint.py
│   ├── adapters/
│   │   ├── base.py
│   │   ├── on_ramp.py
│   │   ├── lane_drop.py
│   │   ├── bottleneck.py
│   │   └── y_merge.py
│   └── utils.py
├── scripts/
│   ├── audit_topologies.py
│   ├── build_taskbook.py
│   ├── sanity.py
│   ├── audit_observation_contract.py
│   ├── audit_task_heterogeneity.py
│   ├── train_task_sac.py
│   ├── train_pooled_sac.py
│   ├── train_oracle_sac.py
│   ├── train_pearl.py
│   ├── evaluate_fewshot.py
│   ├── audit_no_gradient_adaptation.py
│   ├── replay.py
│   └── report.py
└── tests/
    ├── test_taskbook.py
    ├── test_casebook.py
    ├── test_adapters.py
    ├── test_conflict_frame.py
    ├── test_observation_contract.py
    ├── test_reward_consistency.py
    ├── test_pairwise_collision.py
    ├── test_context_encoder.py
    ├── test_product_of_gaussians.py
    ├── test_replay_buffers.py
    ├── test_actor_critic_shapes.py
    ├── test_meta_update.py
    ├── test_support_query_isolation.py
    ├── test_no_gradient_meta_test.py
    └── test_stage1_regression.py
```

---

## 33. 配置骨架

创建：

```text
pearl_learning/configs/merge_family_pearl.yaml
```

建议：

```yaml
project:
  output_root: results/pearl_learning
  stage1_checkpoint: results/sac_scenario_mining/merge_sac_seed2/best_model.zip

environment:
  scenario_family: merge_like
  horizon: 180
  observation_schema: logical_merge_obs_v1
  observation_dim: 56
  action_schema: steering_throttle_v1
  action_dim: 2
  use_render: false

sut:
  controller: idm_baseline
  fixed_parameters: true
  enable_lane_change: false

task_family:
  schema: logical_merge_task_v1
  seen_logical_types:
    - on_ramp_merge
    - lane_drop_merge
    - bottleneck_merge
  held_out_logical_types:
    - y_merge

  splits:
    meta_train:
      num_tasks: 12
    meta_validation:
      num_tasks: 3
    meta_test_template:
      num_tasks: 3
    meta_test_logical:
      num_tasks: 4

cases:
  per_task:
    train_pool: 32
    validation_support: 10
    validation_query: 20
    test_support: 10
    test_query: 20

reward:
  ttc_cap: 5.0
  ttc_dense_threshold: 3.0
  ttc_weight: 1.0
  proximity_weight: 0.1
  target_collision_bonus: 200.0
  non_target_collision_penalty: 3.0
  out_of_road_penalty: 5.0
  wrong_route_penalty: 5.0
  action_l2_weight: 0.01
  action_smoothness_weight: 0.02

pearl:
  latent_dim: 5
  use_information_bottleneck: true
  include_next_observation: true
  context_reward_scale: 200.0
  context_batch_size: 128
  context_sample_size_eval: 256
  kl_beta: 0.1
  posterior_eval_mode: mean

networks:
  context_hidden_sizes: [200, 200]
  actor_hidden_sizes: [256, 256]
  critic_hidden_sizes: [256, 256]

sac:
  actor_lr: 0.0003
  critic_lr: 0.0003
  context_lr: 0.0003
  alpha_lr: 0.0003
  gamma: 0.99
  tau: 0.005
  batch_size: 256
  ent_coef: auto

meta_training:
  meta_batch_size: 6
  bootstrap_episodes_per_task: 5
  prior_episodes_per_sampled_task: 1
  posterior_episodes_per_sampled_task: 1
  gradient_updates_per_iteration: 100
  total_environment_steps: 1500000
  validation_interval_steps: 100000

evaluation:
  shots: [0, 1, 2, 5, 10]
  deterministic_query: true
  query_cases_per_task: 20
  top_k_scenarios: 10
  paired_bootstrap_samples: 10000

experiment:
  training_seeds: [0, 1, 2]
  device: auto
```

---

## 34. 推荐实现顺序

### M0：冻结 Stage 1

完成：

- Git tag；
- Stage 1 回归；
- pairwise collision；
- strict validity；
- 新结果目录。

### M1：拓扑审计

完成：

- 候选地图枚举；
- top-down 图；
- lane graph；
- conflict point；
- 角色路线；
- 可回放性。

### M2：统一环境

完成：

- LogicalScenarioAdapter；
- 4 个 merge-like adapters；
- conflict-centric frame；
- logical_merge_obs_v1；
- fixed SUT；
- 统一 reward。

### M3：Taskbook / Casebook

完成：

- meta-train；
- meta-validation；
- meta-test-template；
- meta-test-logical；
- support/query；
- hash。

### M4：任务异质性

完成：

- per-task SAC；
- cross-task matrix；
- Stage 1 SAC 跨场景；
- pooled SAC；
- Oracle SAC；
- Go/No-Go。

没有通过 M4，不进入正式 PEARL。

### M5：PEARL 数学模块

先在 toy task 上测试：

- prior；
- posterior；
- Gaussian product；
- KL；
- actor/critic shape；
- replay buffer；
- one-step update；
- checkpoint。

### M6：MetaDrive PEARL smoke

使用小任务集：

```text
4 train tasks
2 validation tasks
20k–50k environment steps
```

检查：

- 无 NaN；
- posterior 正常更新；
- buffer 不混 task；
- query 不进 context；
- 元测试无梯度。

### M7：正式 Meta-training

使用冻结 taskbook 和 3 个 seed。

### M8：Few-shot 基线评估

完成：

- pooled SAC；
- scratch SAC；
- fine-tune；
- oracle；
- PEARL；
- Stage 2A；
- Stage 2B。

### M9：审计与归档

完成：

- 参数 hash；
- support/query leakage；
- top-K replay；
- manifest；
- plots；
- README。

---

## 35. 必须支持的命令

### 拓扑审计

```bash
python -m pearl_learning.scripts.audit_topologies \
  --config pearl_learning/configs/merge_family_pearl.yaml
```

### 构建 taskbook

```bash
python -m pearl_learning.scripts.build_taskbook \
  --config pearl_learning/configs/merge_family_pearl.yaml
```

### 环境 sanity

```bash
python -m pearl_learning.scripts.sanity \
  --config pearl_learning/configs/merge_family_pearl.yaml
```

### 观测合同审计

```bash
python -m pearl_learning.scripts.audit_observation_contract \
  --config pearl_learning/configs/merge_family_pearl.yaml
```

### 任务异质性

```bash
python -m pearl_learning.scripts.audit_task_heterogeneity \
  --config pearl_learning/configs/merge_family_pearl.yaml
```

### Pooled SAC

```bash
python -m pearl_learning.scripts.train_pooled_sac \
  --config pearl_learning/configs/merge_family_pearl.yaml \
  --seed 0
```

### PEARL smoke

```bash
python -m pearl_learning.scripts.train_pearl \
  --config pearl_learning/configs/merge_family_pearl.yaml \
  --seed 0 \
  --max-env-steps 50000 \
  --run-name pearl_smoke_seed0
```

### 正式 PEARL

```bash
python -m pearl_learning.scripts.train_pearl \
  --config pearl_learning/configs/merge_family_pearl.yaml \
  --seed 0 \
  --run-name pearl_seed0
```

### 模板级 few-shot

```bash
python -m pearl_learning.scripts.evaluate_fewshot \
  --config pearl_learning/configs/merge_family_pearl.yaml \
  --checkpoint results/pearl_learning/pearl_seed0/best_model.pt \
  --split meta_test_template
```

### 逻辑子类型级 few-shot

```bash
python -m pearl_learning.scripts.evaluate_fewshot \
  --config pearl_learning/configs/merge_family_pearl.yaml \
  --checkpoint results/pearl_learning/pearl_seed0/best_model.pt \
  --split meta_test_logical
```

### 报告

```bash
python -m pearl_learning.scripts.report \
  --results-root results/pearl_learning
```

---

## 36. 输出目录

```text
results/pearl_learning/
├── topology_audit/
├── taskbooks/
│   ├── meta_train_tasks.json
│   ├── meta_validation_tasks.json
│   ├── meta_test_template_tasks.json
│   ├── meta_test_logical_tasks.json
│   └── taskbook_hash.json
├── heterogeneity_audit/
│   ├── policy_task_matrix.csv
│   ├── policy_task_matrix.png
│   ├── stage1_cross_task.csv
│   ├── pooled_sac_summary.json
│   └── decision.json
├── pearl_seed0/
│   ├── config_resolved.json
│   ├── versions.json
│   ├── checkpoints/
│   ├── train_progress.csv
│   ├── validation/
│   └── best_model.pt
├── final_eval/
│   ├── meta_test_template/
│   ├── meta_test_logical/
│   ├── pooled_sac/
│   ├── scratch_sac/
│   ├── oracle_sac/
│   ├── comparison.csv
│   └── plots/
└── README.md
```

---

## 37. 必须生成的图表

1. `topology_templates.png`
2. `task_heterogeneity_matrix.png`
3. `stage1_sac_cross_task.png`
4. `valid_critical_rate_vs_shots_template.png`
5. `valid_critical_rate_vs_shots_logical.png`
6. `target_collision_rate_vs_shots.png`
7. `invalid_rate_vs_shots.png`
8. `median_min_ttc_vs_shots.png`
9. `adaptation_auc_comparison.png`
10. `posterior_variance_vs_shots.png`
11. `latent_embeddings_by_logical_task.png`
12. `per_task_gain_heatmap.png`

---

## 38. 失败判据

### 38.1 环境语义不统一

现象：

- 不同 Adapter 的 action 含义不同；
- target collision 定义不同；
- observation 字段含义不同；
- reward 量纲不同。

处理：

- 停止 PEARL；
- 修复统一环境合同。

### 38.2 Pooled SAC 已经饱和

现象：

```text
Topology-conditioned pooled SAC
在 meta-test-logical 上接近 100%
```

处理：

- 检查 task 是否过于相似；
- 检查 task label 泄漏；
- 增加逻辑结构差异；
- 增加有效性约束；
- 不继续堆 Context Encoder。

### 38.3 PEARL posterior collapse

现象：

- posterior 接近 prior；
- variance 不随 support 减少；
- z 对 task 无区分。

处理：

- 检查 context 是否包含 next observation；
- 检查 buffer 是否混 task；
- 检查 Encoder 是否收到 Q gradient；
- 调整 KL beta；
- 检查 task 是否存在可适配差异。

### 38.4 PEARL 只记住显式拓扑

现象：

- 不使用 support 也能得到相同 z；
- z 只按 topology descriptor 聚类；
- K-shot 曲线没有变化。

处理：

- 比较 PEARL@0 与 PEARL@K；
- 做 no-context 消融；
- 检查 Context Encoder 是否真正使用 transition；
- 检查 topology descriptor 是否过度泄漏模板。

### 38.5 Scratch SAC 异常强

检查：

- 是否使用了 query 数据；
- 是否获得更多环境步；
- 是否获得更多梯度更新；
- 是否直接读取 task ID。

### 38.6 完全 held-out logical type 无增益

如果 Stage 2A 成功而 Stage 2B 失败：

- 不直接引入 MoE 掩盖问题；
- 检查 held-out logical type 是否仍属于 merge-like；
- 检查统一 observation 是否能表达该场景；
- 检查 Actor 容量；
- 报告 PEARL 的泛化边界；
- 再决定是否进入 MoE 阶段。

---

## 39. Codex 完成后的回复格式

Codex 必须提供：

1. Stage 1 回归结果；
2. 候选拓扑审计结果；
3. 实际实现的 logical types；
4. Task/Case 定义；
5. 固定 SUT 说明；
6. Adapter 文件与角色规则；
7. conflict-centric observation 字段；
8. reward 一致性说明；
9. pairwise collision 方法；
10. taskbook hash；
11. support/query split；
12. task heterogeneity matrix；
13. pooled SAC 结果；
14. PEARL 网络结构；
15. context 维度；
16. 梯度流说明；
17. replay buffer 组织；
18. meta-training 实际命令；
19. meta-test 无梯度审计；
20. Stage 2A 结果；
21. Stage 2B 结果；
22. top-K replay 结果；
23. 是否满足跨逻辑场景主张；
24. 尚未解决的问题；
25. 是否进入 MoE 阶段。

---

## 40. 进入 MoE 阶段的条件

只有在以下情况下进入 MoE：

1. PEARL 在相关 merge-like task 上具有 few-shot 增益；
2. 单一 PEARL Actor 在加入更大差异场景后出现明显负迁移；
3. merge-like、crossing-like、roundabout-like 形成可识别的任务簇；
4. 参数量匹配的宽网络无法替代专家分工；
5. 统一状态和动作接口已经稳定。

MoE 不用于修复：

- 错误环境；
- 不统一动作；
- 不可比较的 reward；
- task label 泄漏；
- 无效的 PEARL 实现。

---

## 41. 一句话目标

> **固定 SUT、动作空间和危险场景奖励，把 Task 定义为具体的 merge-like 逻辑场景模板；先建立统一的跨场景环境与观测，再用 PEARL 在训练时未见的逻辑场景模板上通过 1–10 个 support episodes 快速提高有效安全关键场景挖掘能力，并严格对比 topology-conditioned pooled SAC 与等预算 Scratch SAC。**
