# 约束式安全关键场景挖掘

## 实现目标

在机制验证已证明 posterior-routed MoE 的独立价值后，将安全关键场景挖掘从单一奖励优化扩展为显式约束问题，量化关键性、有效性、行为合理性和多样性的 Pareto 权衡。

本阶段不以“碰撞更多”为成功标准，也不默认当前事件指标等价于 RSS 责任判定。只有完成形式化定义、代码验证和责任审计后，才能使用 `RSS-constrained`；此前统一称为 `validity/plausibility-constrained scenario mining`。

## 当前代码判断

当前工程已有以下基础：

- [reward.py](../../pearl_learning/src/reward.py) 用 TTC、距离、目标碰撞奖励、非目标碰撞/出界/错误路线惩罚，以及 action L2 和 action smoothness 构造标量奖励。
- [metrics.py](../../pearl_learning/src/metrics.py) 记录 target/non-target collision、双方出界、wrong route、lane marking violation、min TTC 和 min distance，并定义严格有效关键场景。
- [task_env.py](../../pearl_learning/src/task_env.py) 记录接触方式、目标接触规则、冲突区进入角色和轨迹进度等事件信息。
- 当前 SUT 是固定参数的 IDMPolicy；[task_env.py](../../pearl_learning/src/task_env.py) 的 episode record 也明确写入该控制器标识。

当前工程尚未实现：

- PET、纵/横向加速度与 jerk 的完整 episode 指标；
- RSS safe distance、proper response、危险时刻和责任归因；
- plausibility violation 的冻结阈值和概率约束；
- cost critic、Lagrange multiplier 或其他 CMDP 优化路径；
- 独立的第二类 SUT 稳健性验证。

因此，现有 invalid penalty 或 action smoothness 不能被改名为“RSS 约束”。

## 前置定义：先冻结 ODD 与责任问题

在写训练代码前，新增一份人可读且机器可校验的约束规范，至少定义：

- 道路范围、车道规则、速度范围、车辆动力学范围和仿真时间步；
- adversary 与 SUT 的角色、可控量和感知假设；
- 目标冲突对与非目标交通参与者；
- dangerous situation 的开始/结束时刻；
- safe distance 与 proper response 使用的变量、单位和边界条件；
- 什么条件下接触被判为 adversary 责任、SUT 责任、共同责任或无法判定；
- 缺失观测、数值异常和边界相等时的处理规则。

必须先解决一个目标冲突：若要求主动制造碰撞的 adversary 全程满足保守安全距离和 proper response，目标碰撞可能在定义上不可实现。不得自行掩盖该冲突。可选择并冻结一种研究问题，例如：

1. adversary 行为保持在动力学与交通合理性边界内，寻找暴露 SUT 响应缺陷的危险交互；或
2. 允许 adversary 触发危险，但明确限制非目标损害和不合理动作，并单独报告责任类别。

两种定义不能在同一结果表中混用。若责任定义尚未获得项目负责人确认或基于轨迹回放验证，约束场景挖掘只能做到 plausibility constraints。

## 约束问题

保持机制验证冻结的 PEARL-MoE 架构、任务后验和路由，定义轨迹收益与成本：

```text
maximize   E[R_criticality(tau) + beta * R_diversity(tau)]
subject to E[C_invalid(tau)] <= epsilon_invalid
           E[C_plausibility(tau)] <= epsilon_plausibility
           E[C_responsibility(tau)] <= epsilon_responsibility  # 仅在定义验证后
```

训练使用显式 Lagrangian 或等价的约束强化学习方法：

```text
L_actor_constrained = L_actor
                    + lambda_invalid * (E[C_invalid] - epsilon_invalid)
                    + lambda_plausibility * (E[C_plausibility] - epsilon_plausibility)
                    + lambda_responsibility * (E[C_responsibility] - epsilon_responsibility)
lambda_j <- projection_nonnegative(lambda_j + lr_j * violation_j)
```

要求：

- reward 与每类 constraint cost 分开记录和学习；不能只把所有项目加进 reward 后称为约束优化。
- cost 的方向、单位、episode 聚合和阈值在配置中显式定义。
- v1 优先使用一个聚合 validity/plausibility cost 或少量独立 cost critics，避免一次加入过多不稳定乘子。
- cost critics 保持 dense，避免同时改变 MoE 机制。
- lambda 的更新频率、初值、上限和投影规则进入 checkpoint 与 manifest。

## 指标与事件实现

### Criticality

保留现有 min TTC、target collision、critical rate 和 VCSR，并新增：

- PET：基于冻结 conflict-zone 定义和双方通过时刻计算；无有效先后通过时标为不可计算，不填充为 0。
- near-miss rate：阈值在 validation 前冻结，并同时报告连续 TTC/PET，防止阈值选择性结论。
- 接触时相对速度和接触位置，用于区分轻微接触与高风险接触。

### Validity

保留并分项报告：

- non-target collision；
- adversary/SUT out of road；
- wrong route；
- lane marking violation；
- 仿真异常、无效重置和无法判定接触。

VCSR 的现有定义继续作为主指标，不能因约束训练而悄悄修改。需要新定义时使用新 metric schema，并同时重算所有基线。

### Plausibility

在每个 step 记录真实车辆运动量，而不是只使用策略 action：

- longitudinal/lateral acceleration；
- longitudinal/lateral jerk；
- yaw rate、必要时 yaw acceleration；
- 速度、车道横向偏移和逆向/不合理路线事件；
- 控制饱和持续时间和剧烈方向反转。

阈值来源、单位、仿真采样周期和滤波方法必须写入约束规范。现有 `action_smoothness_weight` 衡量动作变化，不能替代物理 jerk。

### Responsibility

只有在前置定义完成后才实现：

- 每步 safe-distance 状态；
- dangerous situation 首次时刻；
- 双方在响应时间窗内的 proper-response 状态；
- 接触/near-miss 的责任类别与证据时间线；
- `responsibility_valid` 及 `undetermined`，后者不得自动计为有效。

责任判定函数应是纯函数：输入冻结轨迹与参数，输出类别、布尔分项和解释性时间戳，便于用手工构造轨迹单元测试。

### Diversity

保留现有初始速度与出生位置覆盖指标，并增加至少一种轨迹级多样性：

- 冲突区相对状态轨迹的归一化距离；或
- 冻结特征上的风险模式覆盖。

不得仅用随机种子数或唯一 case ID 数代替行为多样性。多样性只能使用训练或生成轨迹，不得反向参与 test 模型选择。

## 实验矩阵

在相同机制验证 checkpoint 选择、taskbook/casebook、K、support、query 和环境步预算下比较：

| 方法 | 用途 |
| --- | --- |
| criticality-only | 展示忽略有效性时的上限与副作用 |
| current reward | 当前 TTC/collision/invalid penalty 基线 |
| soft validity/plausibility penalty | 区分简单 reward shaping 与约束优化 |
| constrained PEARL-MoE | 显式 cost critic + Lagrange |
| constrained PEARL-MoE + diversity | 检验约束下的模式覆盖 |
| RSS-constrained PEARL-MoE | 仅在责任定义验证后加入 |

每组约束阈值或拉格朗日目标形成一个 Pareto 点。超参数在 meta-validation 选择并冻结，test 只评一次。

至少绘制：

- VCSR 对 invalid rate；
- VCSR 对 plausibility violation rate；
- VCSR 对 responsibility-invalid rate（若实现）；
- diversity 对 invalid/plausibility cost；
- `B_K` 对上述 Pareto 指标。

不能只展示总 reward，因为不同方法的 reward 量纲和 penalty 会改变。

## 固定 SUT 主实验与跨 SUT 检查

主实验继续使用当前固定 IDM SUT，以保持与后验适应和机制验证的因果可比性。随后增加至少一个决策机制不同的冻结 SUT 作为独立 robustness split，例如另一类规则控制器或冻结学习策略。

要求：

- SUT 版本、配置、模型哈希和随机性完全记录。
- SUT 类型不是 task label，不进入 posterior 或 router。
- 不在 test SUT 上微调或选择约束阈值。
- 分别报告 in-SUT 与 cross-SUT，不把后者混入训练任务平均值。

若只改变 IDM 参数，应表述为 parameter robustness，而不是跨控制器泛化。若没有第二类控制器，最终结论限定为固定 IDM SUT。

## 目标代码位置

- [task_env.py](../../pearl_learning/src/task_env.py)：逐步运动状态、冲突时刻和责任判定所需原始量。
- [metrics.py](../../pearl_learning/src/metrics.py)：新 metric schema、PET、运动学、责任和多样性汇总。
- [reward.py](../../pearl_learning/src/reward.py)：保留 reward 分项；不要在此隐藏 constraint cost。
- `pearl_learning/src/constraints.py`：待新增的 cost 定义、阈值校验和纯责任判定函数。
- [pearl_agent.py](../../pearl_learning/src/pearl_agent.py)：显式 cost critics、actor constrained loss 和 lambda 更新。
- [checkpoint.py](../../pearl_learning/src/checkpoint.py)：cost critics、optimizers、lambda、constraint schema 和 ODD hash。
- [evaluator.py](../../pearl_learning/src/evaluator.py)：分项 cost、Pareto 和 SUT split 输出。
- `pearl_learning/configs/constrained_scenario_mining.yaml`：唯一正式约束配置。

遵守 [style.md](../style.md)：共享计算只保留一个实现，非法状态显式报错；不创建带版本号或“最终版”后缀的重复脚本、旧格式 fallback 或只转发参数的 wrapper。

## 必须通过的自动测试

### 指标正确性

- 用手工构造轨迹验证 TTC/PET、加速度、jerk、阈值边界和不可计算状态。
- episode 汇总可从 step records 精确重算；单位与仿真时间步一致。
- VCSR 和 invalid 定义须在当前实现生成的新记录上做独立回放一致性验证；退役格式的旧记录不作为兼容性基准。
- `undetermined` 责任状态不会被计为 valid。

### 约束优化

- cost critic target、terminal/truncated mask 与 reward critic 语义一致。
- cost 超预算时 lambda 非减，低于预算时按规则回落且始终非负。
- reward、cost 和 lambda 的梯度只更新预期模块，无 NaN 或未记录裁剪。
- lambda、cost critics 和 optimizer 完成 checkpoint round trip。
- 将所有 lambda 固定为 0 时，机制验证使用的 PEARL-MoE 行为回归测试通过。

### 责任判定

- 预先构造双方分别违反、共同违反、均未违反和无法判定的轨迹。
- dangerous time、response window 和责任类别满足冻结规范。
- 调换车辆角色不会在未定义情况下静默复用同一参数。
- 每个结果可输出最小充分证据时间线，便于人工回放核对。

### 数据与复现

- query 不进入 posterior、router、lambda 更新或阈值选择。
- ODD、constraint、metric 和 SUT schema/hash 写入每个 checkpoint 与结果 manifest。
- 同配置同 seed 的指标、route 与约束状态可复现。

## 约束场景挖掘通过条件

1. 约束定义、单位、阈值和 ODD 已冻结，指标与优化单元测试全部通过。
2. 相对 current reward 和 soft penalty，显式约束方法在独立 test 上降低预注册 invalid/plausibility cost，且 VCSR 与多样性没有不可接受的坍缩。
3. 报告完整 Pareto front 和区间估计，没有只选取一个有利权重点。
4. 约束满足率按任务和训练种子稳定，不依赖少数容易任务。
5. 若使用 RSS/责任表述，责任判定已通过构造轨迹和抽样回放的人机审计，并单独报告 `undetermined`。
6. 固定 IDM 主结果和至少一种 cross-SUT/parameter robustness 结果分开呈现；若未完成则明确限制结论。

约束场景挖掘通过也只支持仿真 ODD 内的场景挖掘证据，不构成真实道路安全保证、SUT 认证或完整 RSS 合规证明。

## 交付物

```text
results/pearl_learning/constrained_scenario_mining/
  manifest.json
  odd_and_responsibility_spec.json
  constraint_schema.json
  metric_validation_report.json
  metrics_by_method_seed_task_k.jsonl
  pareto_points.json
  responsibility_audit.jsonl
  sut_robustness.json
  statistical_summary.json
  compact_results.json
```

## 交付报告格式

```text
约束场景挖掘：PASS | FAIL | PLAUSIBILITY_ONLY | INCOMPLETE
冻结 ODD/责任定义：...
指标与约束测试：...
Pareto 与区间结果：...
VCSR/多样性代价：...
责任无法判定比例：...
固定 SUT 与 robustness 结果：...
最终允许使用的论文表述：...
```
