# MVR Stage1 训练与验收计划

## 目标与当前状态

Stage1 的目标是在未见 SUT、未见道路几何和少量在线证据条件下，学习可迁移的对抗场景挖掘策略。对抗场景必须同时满足：

- 指定 Functional Scenario 的交通语义确实发生；
- 事件发生在目标 SUT 与对抗车之间；
- 交通、路线和动力学约束均未被破坏；
- 训练与评估遵守固定 simulator episode budget。

当前没有通过 Formal Stage1。此前控制接口产生的 pilot checkpoint、GIF 和评估结果均已废弃，不能用于方法性能声明。保留的证据包括：冻结的 `archives/sac_scenario_mining/` 单任务 SAC 正向对照，以及当前二维控制接口的无训练可达性筛查。

## 当前方法

```text
Outer interaction candidate + x0 + Cut-in onset
        -> Functional / Logical Scenario contract
        -> native IDM nominal controller
        -> Inner SAC [delta steering, delta acceleration]
        -> TrafficActionShield
        -> ScenarioSemanticMonitor
        -> valid critical reward and failure analysis
        -> trajectory/outcome token -> PEARL latent -> MoE Outer policy
```

Inner SAC 只输出有界连续车辆修正：

```text
a_exec = Shield(a_nominal + [delta steering, delta acceleration])
```

Native IDM 负责正常行驶，场景契约负责路线、合法机动、冲突区和 Cut-in onset。Cut-in onset 是 Outer 连续参数 `maneuver_onset_progress`，位于合法 merge window 内；它不是 SAC 动作维度。Inner observation 为 11 维物理交互状态：相对位置/航向、双方速度与闭合速度、距离、双方 route progress、相对 conflict ETA 和 challenge phase。

Outer 目前仍保留 `approach_conflict`、`yield_then_press` 与 `gap_close` 三个场景 profile。它们不再是 Inner action，但仍影响 nominal intent；在开始正式训练前，应以“移除 profile”的消融确认其是否必要，避免把人为行为语义误当作学习效果。

## 合理对抗行为契约

### Cut-in

- 对抗车从 source lane 合法进入 target lane；
- intrusion 以 oriented vehicle footprint 与 target-lane corridor 的真实重叠判定；
- critical event 仅在 target-lane interaction phase 内有效。

### Merge

- 对抗车走 branch entry，SUT 保持 mainline；
- 只有双方位于收敛冲突区域时的 target collision/near miss 才有效；
- 对抗车没有切入主车车道时，也可通过收敛几何形成合法交互。

### Roundabout

- 双方遵守各自 entry-to-exit 路线；
- 事件需要发生在共享冲突区域；
- 不能以驶离道路、错误路线或非目标碰撞换取 reward。

`ScenarioSemanticMonitor` 将 collision 置于 near-miss 之上作为 decisive event。near-miss 不终止路线测试；event bonus 只在 `event_just_captured` 时发放一次。

## 训练协议

1. 在 36 个训练 task 上运行 Inner pretrain，使用确定性、平衡的 candidate/x0 抽样。
2. 记录每次 rollout 的 raw SAC action、executed action、事件、reward 和训练信号密度。
3. 运行 PEARL posterior、Inner latent calibration 和 MoE Outer；每一阶段只更新其声明的组件。
4. 在未见 SUT / 道路几何组合上，以固定 budget 做少样本在线适应和独立评估。
5. 对每个 checkpoint 保存配置、taskbook hash、控制契约和 source provenance；控制接口改变后不得恢复不兼容 checkpoint。

训练前必须先完成 no-RL action reachability 筛查。其作用是验证动作与 x0 是否能在合法约束下进入 challenge region，不是性能报告。

```powershell
conda run -n metadrive python -m mvr.scripts.sweep_stage1_actions --config mvr/configs/mvr_stage1.yaml --output results/mvr/diagnostics/s0_action_reachability.json
```

## 当前筛查证据与解释

当前 S0 使用三个 validation task、每 task 四个固定初始条件、五个常量二维 residual，共 60 个配对 simulator episode。完整记录位于 `results/mvr/diagnostics/s0_action_reachability.json`。

- Cut-in：`acceleration_brake` 相对 base 平均降低 TTC 和距离，且存在合法 target collision；二维物理动作接口可产生有效交互。
- Merge：当前固定 x0 下未出现 valid critical event；制动能降低 TTC，但仍未进入临界区。
- Roundabout：当前固定 x0 离交互区过远，未出现 valid critical event。

这说明下一步首先是校准 Merge 与 Roundabout 的 Logical Scenario/x0 分布，而不是重新引入时机状态机或扩大 SAC 预算。

## Formal Stage1 验收

只有以下全部成立才能声明通过：

- 工程：配置、taskbook、checkpoint、评估和可视化可复现；
- 覆盖：所有训练 family、candidate、SUT 和 geometry 轴按预算覆盖；
- 语义与交通：有效事件满足目标交互、合法路线和 shield/traffic 约束；
- 可控性：训练策略在固定 x0 配对实验中优于 base 与随机 residual，并能重复产生合法 critical interaction；
- 迁移：未见 SUT 和未见道路几何下的固定预算结果满足预先定义的联合迁移门槛。

未满足任一项时，只能报告诊断现象，不得将 collision 或 near-miss rate 表述为方法性能。

## 维护规则

- `mvr/` 是唯一活跃实现；`archives/` 仅保留冻结基线和正向对照。
- `results/mvr/` 只保存与当前控制契约兼容且可复现的证据。
- 删除失效 checkpoint、旧 GIF、过期计划和缓存；不保留兼容层。
- 新增 schema、动作接口或事件字段时，必须同步增加公共契约测试和 headless MetaDrive 测试。
