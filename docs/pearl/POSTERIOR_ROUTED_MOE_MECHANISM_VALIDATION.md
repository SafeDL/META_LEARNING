# 后验路由 MoE：机制验证与正式收益

## 验证目标

在路由 MoE 工程实现正确的基础上，先做有边界的机制预实验；只有后验适应后续正式证明 support 有效后，才完成容量匹配、多训练种子、冻结测试协议下的正式机制验证，验证以下因果链：

```text
support 轨迹
  -> posterior 改变
  -> task-level route 改变
  -> 专家行为改变
  -> 未见任务上的严格有效关键场景挖掘改善
```

本阶段必须排除三种替代解释：提升只来自更多参数、router 只做静态拓扑分类、结果只来自单个训练种子或少数 query case。

## 当前代码判断

- [moe.py](../../pearl_learning/src/moe.py) 已实现静态描述符、posterior mean、posterior mean + log-variance 等 router 输入模式，以及公开的 uniform、frozen-prior 和 expert-knockout 路由干预接口。
- [pearl_agent.py](../../pearl_learning/src/pearl_agent.py) 已实现 dense/actor-only MoE 共用的 SAC 训练核心；内部 SAC 路径固定零 latent 并冻结 context encoder，MoE-SAC 只使用静态描述符。
- [evaluator.py](../../pearl_learning/src/evaluator.py) 已实现 query-only frozen-router、frozen-latent、both-frozen、uniform 和 expert-knockout；support 采集保持完整自适应，并可审计匿名 experts 在相同初始状态下的动作差异。
- [run_routed_moe_pilot.py](../../pearl_learning/scripts/run_routed_moe_pilot.py) 已统一运行 SAC、MoE-SAC、PEARL、PEARL-MoE，保存 2×2 效应、资源统计、路由轨迹和机制干预产物。
- router swap、random-frozen routing、容量匹配 dense 网络、成套 router 输入消融和多种子区间估计仍未实现为正式实验，因此机制验证仍未正式完成。
- 现有 transferability、support-selection 和 disentangled-representation 接口是独立研究变量，不能被当作本机制验证的 MoE 结果或默认一并启用。

## 已完成的 pilot

初始 2-expert 2×2 pilot 已证明统一训练、干预和审计链路可以运行，但在 `K=4` 未观察到 PEARL-MoE 对 PEARL 的 VCSR 增益，`full − frozen-router` 也为 0。其完整可追溯产物保留在 [pilot manifest](../../results/pearl_learning/posterior_routed_moe_mechanism/pilot/manifest.json)。

后续的 4-expert Top-2 专门化 pilot 取代它作为当前的诊断依据，结果与边界见本文末尾的“后验可分性与路由专门化 pilot”。两轮结果均不支持“posterior 驱动的有益专家路由”主张，状态维持 `PILOT / formal INCOMPLETE`。

## 前置条件

- 后验适应 manifest 为 `PASS`，且冻结了 K、上下文、taskbook/casebook、主指标和统计判据。
- 路由 MoE 工程 manifest 为 `PASS`，dense 回归、MoE checkpoint、梯度边界和无泄漏测试全部通过。
- 正式配置关闭 disentangled representation、主动 support 选择、迁移拒绝器、uncertainty temperature、routing consistency 和 expert diversity。除非某项被预注册为单独消融，否则不得与主 MoE 同时启用。
- support selection 固定，所有方法共享同一 support/query cases、query 顺序和新任务环境步预算。

## 正式任务规模

当前 10 个 meta-train 任务、2 个 meta-validation 任务和 4 个 meta-test-logical 任务只足够调试，不足以支撑 4 个匿名专家的稳定专门化结论。正式实验的设计目标至少为：

- meta-train：24 个任务，即 3 个训练逻辑类型 × 4 个物理几何 × 2 个相反隐藏规则；
- meta-validation：8–12 个任务；
- meta-test：12–20 个任务；
- 每个物理几何至少有相反隐藏规则的成对任务；
- support 与 query case 独立，所有 split 哈希不相交。

如果资源限制不能达到该规模，结果必须标为 `pilot`，不能标记正式机制验证通过。只要环境仍只覆盖 merge、lane-drop、bottleneck、Y-merge，论文范围保持 `merge-family`。

## 核心 2×2 实验

以“是否使用 PEARL posterior”和“是否使用 MoE actor”为两个因子：

| 方法 | 概率任务后验 | MoE actor | 路由输入 |
| --- | --- | --- | --- |
| SAC | 否 | 否 | 无 |
| MoE-SAC | 否 | 是 | 仅经审计的静态描述符 `h_T` |
| PEARL | 是 | 否 | 无 |
| PEARL-MoE | 是 | 是 | `h_T + mu_K + log_var_K` |

这里的 SAC 与 MoE-SAC 必须使用与 PEARL 分支相同的 PyTorch actor、SAC objective、观测、归一化、训练任务采样和更新预算。现有 [baselines.py](../../pearl_learning/src/baselines.py) 中基于 Stable-Baselines3 的在线基线可以作为外部参照，但不能与自定义 PEARL-MoE 组成因果 2×2 后声称交互效应。

无 posterior 的内部匹配 SAC 路径应使用 0 维 latent 或固定零 latent，并明确关闭 context encoder。MoE-SAC 的专家结构、expert 数、Top-k、共享主干和训练预算与 PEARL-MoE 相同，只去除 posterior 输入与无梯度适应。

对每个任务、K 和主指标 Y 计算：

```text
Delta_meta = Y(PEARL) - Y(SAC)
Delta_moe_without_meta = Y(MoE-SAC) - Y(SAC)
Delta_moe_with_meta = Y(PEARL-MoE) - Y(PEARL)
Delta_interaction = Delta_moe_with_meta - Delta_moe_without_meta
```

主张“posterior 与 MoE 协同”必须由预注册 K/预算区间上的 `Delta_interaction` 及其区间估计支持，不能只比较四个均值的排序。

## 容量与计算量控制

至少加入以下匹配对照：

| 对照 | 控制的问题 |
| --- | --- |
| 原始 dense PEARL | 与当前基线直接比较 |
| total-parameter-matched dense PEARL | MoE 提升是否仅来自总参数更多 |
| active-compute-matched dense PEARL | MoE 提升是否仅来自单次前向计算更多 |
| uniform-routing experts | 多专家平均是否已经足够 |
| random-routing experts | 学习到的任务路由是否必要 |
| learned posterior routing | 完整方法 |

报告并由脚本自动计算：

- 总参数量与可训练参数量；
- 单次前向激活参数量；
- actor 和完整决策前向 FLOPs；
- batch size=1 的推理延迟及硬件/软件环境；
- 训练 wall-clock、峰值显存和总环境步数。

参数匹配网络只能在 meta-validation 上确定一次。不要为了贴合 test 结果反复改变 dense 宽度。

## Router 输入消融

在相同 MoE actor 和训练预算下比较：

| 输入/路由 | 解释 |
| --- | --- |
| `h_T` | 纯静态拓扑/物理路由 |
| `mu_K` | 纯 posterior mean |
| `h_T + mu_K` | 静态结构与任务推断互补 |
| `h_T + mu_K + log_var_K` | posterior variance 是否提供额外信息 |
| uniform | 不学习任务选择 |
| random frozen | 保留稀疏计算但移除语义路由 |

`log_var_K` 仍只作为输入特征。只有在独立 calibration 证明其可解释性后，另开实验研究 uncertainty-controlled temperature；不能把 Product-of-Gaussians 的自然收缩直接解释为置信度改善。

## 路由机制干预

### Frozen router / latent

对每个测试任务保存 K=0 的 prior latent 与 route `w_0`，执行四种无训练参数更新的评估：

| 干预 | actor latent | route |
| --- | --- | --- |
| full | 当前 `z_K` | 当前 `w_K` |
| frozen-router | 当前 `z_K` | 固定 `w_0` |
| frozen-latent | 固定 prior latent | 当前 `w_K` |
| both-frozen | 固定 prior latent | 固定 `w_0` |

这些干预共享相同 query cases。`full - frozen-router` 是 support 通过专家切换产生作用的直接证据；`full - frozen-latent` 则衡量 experts 内部 latent 条件化的贡献。

### Router swap

在同几何异规则任务对 `(T_A, T_B)` 上交换 `w_K^A` 与 `w_K^B`，但保持各自 posterior latent 和 query case。交换后若任务表现系统性下降，才能支持 route 的任务特异性。交换操作必须记录源/目标 task、K 和 route hash，且不重新训练。

### Expert knockout

逐一屏蔽 expert e，将其权重设为 0 后对剩余激活权重重新归一化，在全部测试任务上重评。报告每个 expert 的逐任务性能下降，而不是只给平均下降。训练前专家保持匿名；“late-conflict expert”等名称只能由 knockout、动作差异和轨迹分析在训练后归纳。

### Routing trajectory

对每个 task 和 seed 输出 `w_0, w_1, w_2, w_4, w_8`，以及：

- `L1(w_K, w_0)` 和相邻 K 的 route distance；
- posterior distance；
- query VCSR/return 的相邻 K 变化；
- expert utilization、entropy、load coefficient of variation；
- route change 与性能 change 的任务级相关性及区间。

trajectory 必须使用后验适应的固定嵌套上下文。相关性只作为机制一致性证据，不能代替 frozen-router 或 swap 干预。

## 专家专门化与防坍缩审计

至少报告：

- 每个训练/验证/测试任务的 route heatmap；
- 每个 expert 的激活率、平均权重、Top-k 命中率；
- router entropy 和 expert load coefficient of variation；
- experts 输出的动作均值差异，或相同状态/latent 下动作分布的对称 KL；
- expert knockout 影响矩阵；
- 按 logical type、物理几何和隐藏规则生成的 posthoc 分组视图。

隐藏规则和 logical type 只允许出现在 `posthoc_only` 审计产物，不能进入训练路由。以下任一现象必须明确报告：单 expert 长期占用绝大部分任务、不同 experts 动作近乎相同、路由跨种子完全不稳定、路由只按几何聚类而忽略隐藏规则。

## 评估指标

### 主指标

- VCSR，按任务、K、`B_K` 和训练种子报告。
- validation freeze 中预注册的预算性能 AUC；AUC 的插值和缺失点规则必须预先固定。

### 次要指标

- target collision rate、critical rate、invalid rate；
- median min TTC、median min distance、mean query return；
- 首次严格有效关键场景的 episode 数和累计 query 环境步数；
- 有效关键初始条件覆盖与多样性；
- router/专家机制指标；
- 参数、计算、延迟、训练时间和显存。

`episodes_to_first_valid_critical` 依赖冻结 query 顺序，只能在顺序一致时成对比较。论文若使用“time to first”，必须同时给出累计环境步数，而不是只给 episode 序号。

## 统计设计

- 正式方法至少 3 个独立训练种子，目标 5 个；所有方法尽量使用配对种子。
- 任务是独立统计单位；seed 和 K 是重复/层次结构。
- 对主效应、交互效应和关键干预使用层次化 bootstrap 或 mixed-effects model，报告效应量与置信区间。
- 对多个 K 和多个消融预注册主比较并控制多重比较；未预注册分析标为 exploratory。
- validation 只负责模型和超参数选择；冻结 manifest 后 test 只运行一次。失败种子、崩溃和 NaN 不得静默删除。

## 正式机制验证通过条件

只有以下证据同时成立才能通过：

1. PEARL-MoE 相对 dense PEARL 的主指标成对效应为正，区间估计满足预注册标准。
2. PEARL-MoE 相对 MoE-SAC 的成对效应为正，说明性能不只是静态 MoE。
3. `Delta_interaction` 为正且区间估计满足预注册标准，支持 posterior 与 MoE 的协同。
4. total-parameter-matched 和 active-compute-matched dense 对照不能解释完整增益。
5. frozen-router、random/uniform route、router swap 中至少形成一致的性能退化证据，且 routing trajectory 与 support 后验变化一致。
6. experts 没有功能性坍缩；专门化在多个种子上可复现，或至少在功能影响层面可对齐。
7. 结论在独立 meta-test、逐任务结果和实际环境步预算下成立，没有用平均值隐藏系统性负迁移。

阈值、主 K、AUC 定义和区间判据必须在 test 前写入 frozen manifest。若只满足“PEARL-MoE 平均值最高”，但交互、干预或容量控制失败，机制验证仍为未通过。

## 目标代码与产物

优先扩展统一实验和汇总入口：

- [baselines.py](../../pearl_learning/src/baselines.py)：增加内部架构匹配的 SAC/MoE-SAC，不替换现有外部基线。
- [evaluator.py](../../pearl_learning/src/evaluator.py)：frozen route/latent、swap、knockout 和 trajectory 评估。
- 路由 MoE 模块：提供明确的干预接口，不在 evaluator 直接修改私有 tensor。
- `tools/profile_pearl_resources.py`：如需新增，集中参数/FLOPs/延迟/显存统计，避免在各实验脚本重复实现。
- `pearl_learning/configs/posterior_routed_moe_pilot.yaml`：当前机制预实验配置；正式配置应在验证通过后单独冻结。

```text
results/pearl_learning/posterior_routed_moe_mechanism/
  manifest.json
  frozen_validation_selection.json
  metrics_by_method_seed_task_k.jsonl
  factorial_effects.json
  capacity_compute_profile.json
  router_input_ablations.json
  routing_interventions.json
  expert_specialization.json
  statistical_summary.json
  compact_results.json
```

## 交付报告格式

```text
机制验证：PASS | FAIL | PILOT | INCOMPLETE
正式任务与训练种子规模：...
2x2 主效应与交互区间：...
容量/计算匹配结果：...
frozen/swap/knockout 结果：...
collapse 与负迁移审计：...
冻结 test 产物和哈希：...
可支持的最强结论：...
是否允许进入约束场景挖掘：YES | NO
```

## 2026-08-10：后验可分性与路由专门化 pilot

为排除原始 `2 experts + Top-2` 实际激活全部专家的混合效应，新增了独立的验证集 pilot：固定 10 个 meta-train 任务、4 个 meta-validation 任务、训练种子 41、每个变体约 10,000 个环境步、`K={0,1,2,4}`，不使用任何 meta-test 数据。所有变体均使用 4 experts、Top-2 sparse routing，并在四种 router 输入与两档负载均衡权重之间比较：`static`、`posterior_mean`、`static_posterior_mean`、`static_posterior_mean_logvar` × `{0, 0.001}`。

结果保存在 [specialization pilot summary](../../results/pearl_learning/posterior_routed_moe_mechanism/specialization_pilot/summary.json)，逐变体的 full/frozen-router 评估也已保留。8 个变体均通过 support 一致性与评估参数不变性检查。

- 同几何、异规则任务在 `K=4` 的 posterior mean L2 距离为 `0.00667–0.02221`；留一几何对的事后规则识别准确率在 `0–1` 之间波动。由于只有两个几何对和一个训练种子，不能证明后验稳定地区分异规则任务。
- `static` 路由在 support 后完全不变，符合其不接收后验输入的设计。其余 posterior 输入的平均 `L1(w_K,w_0)` 分别仅为 `3.7e-5–1.7e-4`（不含方差）和 `0.00105–0.00327`（含方差）；同几何异规则任务在 `K=4` 的 route L1 最大仅 `1.39e-4`。相比 route L1 的理论上限 2，这些均不构成实质的任务路由分离。
- 仅 `static_posterior_mean_logvar + 0.001` 在 `K=4` 出现 `full - frozen-router VCSR = 0.0625`；其余 7 个变体均为 0。该单种子、少 query case 的孤立点不能作为因果路由增益，也没有与稳定的异规则 route 分离同时出现。

因此，本轮将状态维持为 **PILOT / formal INCOMPLETE**：4-expert sparse routing 排除了“两个专家被强制平均”的直接结构性解释，但未产生可重复的后验驱动专门化证据。下一步应先加强并诊断 context/posterior 对异规则任务的可分性（增加有信息量的 support 轨迹与任务级 posterior 审计），而不是直接扩大 MoE 正式性能测试。
