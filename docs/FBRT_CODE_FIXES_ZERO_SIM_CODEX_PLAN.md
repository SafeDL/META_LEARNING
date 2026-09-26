# FBRT-Memory：代码修复与零新增仿真回放任务书

> **交给 Codex 的执行目标：修正已定位的数据流、特征索引和奖励错误，在已有实测结果库上完成一次修正后的回放并交付结果。不要扩充实验、训练策略、修改场景来追求正向结果。**

- 项目：`SafeDL/META_LEARNING`
- 本任务书依据的代码提交：`2aa313ecc94779d8443ddc850b5ff3beabb546c2`。
- 日期：2026-09-26。
- 工作对象：`method_chains/failure_memory_regression/` 的 `memory_v2` 链路。
- 新增物理仿真预算：**0 次**，包括 smoke、audit、回放补录、缓存缺失补跑。
- 新增驾驶策略训练：**0 次**。
- 执行原则：**修正实现，不保证收益；已有结果重新计算，不重新生成。**

本文件是修改说明，不代表修改已经完成。文中的新增函数、测试文件、CLI 和输出路径均为 Codex 应实现的目标接口；不能直接当作仓库已有能力。源码事实与本任务书提出的修复决策在下文分开说明。

---

## 0. 范围与终止条件：本轮到底做什么

### 必须完成

1. 修复旧回归任务中历史失败被 `completed=True` 过滤的问题。
2. 修复新增失效 RBF 特征后，特征位置与历史先验系数错位的问题。
3. 修复跨 agent 任务中 UCB 错用 `regression`、实际不能获得失败奖励的问题。
4. 保留已观测的碰撞对象等信息；分别报告“无历史失败”“无新增回归”“目标失败全部可被基线发现”等情况。
5. 运行少量**纯数据／纯模型单元测试**，然后只在现有结果库上完成一次完整修正回放。
6. 写出修改内容、修正后的比较结果及其适用范围，保留负结果。

### 本轮不做

- 不增加场景、场景参数维度、被测算法、修改版本或物理随机种子。
- 不重新运行 `smoke`、`compact-bank`、`audit_v4` 或任何 episode 执行器。
- 不升级 highway-env、Stable-Baselines3，不重新下载或加载 PPO 权重。
- 不更改碰撞定义、场景范围、控制频率、策略动作空间或既有物理记录。
- 不新增 GP、Transformer、元强化学习、信息增益规划器或新的评分组合。
- 不调参到获胜，不自动寻找更有利的 agent 顺序。
- 不要求统计显著、所有指标同时提高、每个修改影响多个场景，才允许交付。

### 完成条件

**三个核心错误已修正、相应纯计算测试通过、现有银行可回放的任务已经完成、结果如实输出，即完成本轮。**

若修正后仍无增益，直接交付 `no_gain` 或 `mixed` 结果，不继续扩实验。缓存缺失时，只把受影响任务标为 `MISSING_CACHE`，完成其他任务和代码交付，不自动补仿真。

本文件覆盖旧任务书中与本轮冲突的“新增仿真额度”“扩展 SUT／场景”“必须产生正向结果后再交付”等要求；不改变历史报告对当时实验的事实记录。

---

## 1. 已定位问题与证据

| 编号 | 已定位的源码事实 | 影响 | 修改位置 |
|---|---|---|---|
| P0-A | `cache_stage()` 构建 `parent_history` 时要求 `completed is True` | 碰撞终止的参考记录未传入失效模式构建，历史输入变为通过样本 | `experiment_v2.py` |
| P0-B | 特征顺序是“基础坐标＋失效中心＋覆盖中心”；新增中心加到 `centers`，但 `extend_prior()` 只在向量末尾补值 | 原覆盖特征的系数被赋给新失效特征，后续系数整体错位 | `pattern_memory.py`、`bayes_model.py`、`selector_v2.py` |
| P0-C | UCB 在两种任务中都以 `regression` 更新；跨 agent 候选没有父版本通过标记 | 跨 agent 中发生碰撞也无法给 UCB 正奖励 | `selector_v2.py` |
| P1-A | 任务没有明确区分“历史无失败”和“目标池无回归”与方法无效 | 总体状态容易掩盖不同任务实际检验的内容 | 回放入口、报告 |
| P1-B | `observed` 未完整保留碰撞对象／事件信息；模式语义主要由模板名映射 | “切出后静止车”中的移动前车碰撞可能被错误叙述为静止目标失效 | `selector_v2.py`、报告 |

**边界说明：**P0-A/B/C 是可以从代码确定的实现问题；它们分别导致多少个发现数量的损失，尚未被隔离计算。本轮一起修复，不宣称某个错误单独解释了全部负结果。

### 需要保留的已知任务事实

以下为原 `memory_v2` 报告中的事实，不是修复后预期值：

- MOBIL 参考版在 80 个候选上零碰撞；关闭后车保护版有 12 个碰撞。
- PPO 参考版有 19 个碰撞，观测延迟版有 18 个；61 个参考通过候选上没有新增回归。
- 原跨 agent 顺序为 MOBIL 参考版 → PPO 参考版；前序 MOBIL 没有可传递的已观测失败。
- 原 MOBIL 回归中 NoMemory 在 B=20 已找到全部 12 个失败。
- 原 S02 两个 PPO 构建的 31 次碰撞均指向移动前车，而非静止目标。

这些事实不能通过修改报告状态、调换任务顺序、导入目标未查询标签来“修正”。它们决定当前任务能够支持的解释范围。[S5][S6]

---

## 2. 保护输入数据，建立独立的修复输出

### 2.1 冻结输入

只读使用：

```text
results/method_chains/failure_memory_regression/standard_aligned/core/
    reference_archive.csv
    candidate_pool.csv
    target_response_bank.csv

results/method_chains/failure_memory_regression/memory_v2/
    archive_v2.jsonl
    summary_by_task.csv
    report.md
    acceptance.json
    compact_bank/scenario_cases.jsonl
    compact_bank/episodes.jsonl
    sessions/                  # 只用于查阅旧任务配置、随机种子或已有日志
    validation_audit_v4/        # 只读诊断，不混入主银行
```

`compact_bank/episodes.jsonl` 是原 `_current_bank()` 使用的实测银行路径。[S1]

开始时记录实际存在输入的 SHA-256；结束时确认这些输入未被改写。缺失文件列表写入报告，不把不存在的文件当作空历史悄悄继续。

旧 `archive_v2.jsonl` 可能包含既往执行后写回的会话记录。因此，它不是可以无条件全部交给所有任务的训练集：具体历史视图必须按第 3 节和第 7 节构建。

### 2.2 新输出目录

```text
results/method_chains/failure_memory_regression/repair_20260926/
```

回放分支的新快照也写入此目录，不能写回原 `memory_v2/archive_v2.jsonl`。

若目录已有修复输出，读取其 manifest 判断是否为同一代码与配置的续跑；不能静默混合不同修改版本的结果，也不需要删除旧结果。

### 2.3 Codex 工作方式

在当前工作分支上做最小 patch。先记录 `git rev-parse HEAD`、`git status --short`；不要 `git reset --hard`，不要覆盖用户未提交修改。若当前代码较基准已经修复某项，保留修复并增加对应测试，不倒退到旧实现。

只读文档和源码审阅不增加物理预算；允许使用 NumPy/SciPy 重新拟合已有数据模型。

---

## 3. P0-A：历史建模集合与回归候选集合必须分开

### 3.1 当前错误

`experiment_v2.py::cache_stage()`：

```python
parent_history = [
    row for row in root_records
    if row["build_id"] == "idm_ref"
    and row.get("simulator_seed") == seed
    and row.get("completed") is True
]
```

最后一项仅适合“参考通过候选”，不适合“历史建模记录”。碰撞造成提前终止，不表示该次失败记录不可用。[S1]

### 3.2 目标逻辑

定义两个独立谓词，供各调用点统一使用：

```python
from collections.abc import Mapping
from typing import Any


def is_usable_outcome(row: Mapping[str, Any]) -> bool:
    """输入必须已由加载器把布尔字段规范化；不在此处猜测字符串。"""
    if row.get("inconclusive", False):
        return False
    if row.get("ego_collision") is True:
        return True               # 碰撞终止也是有效失败观测
    return (row.get("ego_collision") is False
            and row.get("completed") is True)


def is_parent_pass(row: Mapping[str, Any]) -> bool:
    return (not row.get("inconclusive", False)
            and row.get("ego_collision") is False
            and row.get("completed") is True)
```

输入 CSV 字符串 `"True"/"False"` 先复用 `archive_v2.py` 的规范化逻辑；不能用 `bool("False")`。

修正历史构建：

```python
parent_history = [
    row for row in root_records
    if row.get("build_id") == "idm_ref"
    and row.get("simulator_seed") == seed
    and row.get("visibility") == "historical"
    and is_usable_outcome(row)
]
```

随后仍按当前 `_contextual_history()` 保留与候选相同的物理上下文。缺失 visibility 的旧参考输入由加载器明确补为 `historical`；不能将未知来源或目标评估记录一概改为历史可见。

候选仍保持：

```python
candidates = [row for row in original_candidates if is_parent_pass(parent_by_id[row["scenario_id"]])]
```

不要为了多利用失败信息，将参考失败点加入原回归候选池。**失败点用于历史建模；参考通过点用于新回归检测。**

### 3.3 历史来源不能混乱

- 旧 IDM 回放：当前任务只使用对应种子、对应参考 build 的完整有效历史。
- 目标 build 的完整 `target_response_bank.csv` 仍是 evaluator-only，只能由 oracle 逐次揭示。
- 参考历史无需限制为候选 ID 的交集，否则参考失败会再次被间接删除。
- 前序会话的记录只有在协议允许、且确为该方法已查询时，才能用于后续 agent。
- `completed=False`、`ego_collision=False` 的不完整执行不是通过；本轮不新增推断标签。

### 3.4 记录真正到达模型的历史

每个任务在过滤前、上下文过滤后各记录：

```text
history_record_count
history_pass_count
history_failure_count
history_build_ids
history_context_ids
initial_pattern_count
excluded_record_counts_by_reason
```

写入新 `task_inputs.jsonl`。统计对象必须是实际传给 `build_pattern_cards()` 和 `source_fits()` 的列表，不是全局档案规模。

已有旧参考档案总数为 480、候选通过数为 320；读取时自行核对失败和不确定记录的实际计数，不在代码中硬编码“必须等于某个失败数”来迫使数据通过。

### 3.5 最小测试

构造四条小记录：有效通过、碰撞终止、背景事故导致的不完整执行、目标 evaluator-only。检查：

- 前两条进入历史；仅第一条进入回归通过候选。
- 第三条不进入训练标签；第四条不因本修复变为历史。
- 同一模板中提供一条有效失败和一条有效通过时，模式构建可产生真实记录 ID 支撑的卡片。
- 历史无失败是允许状态；不能为了让计数非零而制造卡片。

---

## 4. P0-B：按稳定特征 ID 对齐先验，禁止仅按长度补齐

### 4.1 错位示例

原特征：

```text
[bias, x1, x2, coverage_1, coverage_2, coverage_3, coverage_4]
```

加入一个新失败中心后的特征：

```text
[bias, x1, x2, failure_new, coverage_1, coverage_2, coverage_3, coverage_4]
```

原先验均值若为 `[b, a1, a2, c1, c2, c3, c4]`，则新均值应为：

```text
[b, a1, a2, 0, c1, c2, c3, c4]
```

当前 `extend_prior()` 末尾补零会得到错误的：

```text
[b, a1, a2, c1, c2, c3, c4, 0]
```

原因见 `RBFDictionary.features()`、`new_failure_card()` 和 `extend_prior()`。[S2][S3]

### 4.2 采用唯一修复路线：完整 feature schema＋ID 重映射

不要同时实现多套候选修复。保留当前 RBF 计算公式和排列方式，增加完整有序特征 ID：

```text
bias
coord:<第一个有效物理参数名>
coord:<第二个有效物理参数名>
...                                      # 已支持的其他输入，仅保持兼容
failure:<center_id>
...
coverage:<center_id>
...
```

要求：

- `ordered_feature_ids()` 与 `features()` 逐项对应，包含 bias 和所有坐标，不只是 RBF。
- 目前 `feature_ids` 仅列 RBF；不要误把它直接当作完整 schema。可保留旧属性，新增完整属性以免破坏调用者。
- 坐标名顺序必须与 `active_values()` 一致。
- 同一任务中，既有特征的 ID、中心位置、带宽、归一化定义不应改变。发生变化时不能复用该 ID 的系数。
- 特征 schema 同时保存 template、context、参数化版本；不允许只因维度相同就跨上下文复用。
- 新失效特征先验默认 `N(0,4)`，沿用原默认，不调参。

### 4.3 可直接采用的重映射辅助函数

以下函数用于**对角高斯源先验**。它不是把目标后验再作为下一轮先验：

```python
from collections.abc import Sequence
import numpy as np


def align_diagonal_prior(
    old_ids: Sequence[str],
    old_mean: np.ndarray,
    old_variance: np.ndarray,
    new_ids: Sequence[str],
    *,
    default_mean: float = 0.0,
    default_variance: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """按特征身份迁移系数；新特征独立使用默认先验。"""
    old_ids, new_ids = tuple(old_ids), tuple(new_ids)
    if len(set(old_ids)) != len(old_ids) or len(set(new_ids)) != len(new_ids):
        raise ValueError("feature IDs must be unique")
    if not set(old_ids).issubset(new_ids):
        raise ValueError("existing source-prior features cannot disappear in a session")

    mean = np.asarray(old_mean, dtype=float)
    variance = np.asarray(old_variance, dtype=float)
    if mean.shape != (len(old_ids),) or variance.shape != mean.shape:
        raise ValueError("prior arrays do not match the old feature IDs")
    if (not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(variance))
            or np.any(variance <= 0)):
        raise ValueError("source prior must be finite with positive variances")
    if (not np.isfinite(default_mean)
            or not np.isfinite(default_variance)
            or default_variance <= 0):
        raise ValueError("invalid default prior")

    lookup = {feature_id: i for i, feature_id in enumerate(old_ids)}
    new_mean = np.full(len(new_ids), default_mean, dtype=float)
    new_variance = np.full(len(new_ids), default_variance, dtype=float)
    for j, feature_id in enumerate(new_ids):
        if feature_id in lookup:
            i = lookup[feature_id]
            new_mean[j] = mean[i]
            new_variance[j] = variance[i]
    return new_mean, new_variance
```

### 4.4 调用链必须一起改

1. 构建初始字典。
2. 基于初始字典拟合 source，保存 `source_prior_ids / mean / variance`。
3. 这些 source prior 在单个目标会话内保持不变。
4. 每轮调用 `target_posterior()` 前，从上述**原始 source prior** 对齐到当前 `ordered_feature_ids()`。
5. 用重新计算的当前特征矩阵，以及全部已查询有效目标观测，各一次，进行后验拟合。
6. 新增失败中心后，下一轮重新映射；不能只在第一次扩维时修补。
7. 所有 source model 缓存都保存 schema。没有 schema 的旧模型缓存从已有历史重新拟合，不按向量长度猜测。

建议参数形式：

```python
target_posterior(
    dictionary,
    observations,
    source_prior_ids,
    source_prior_mean,
    source_prior_variance,
)
```

禁止下面的双重计数：

```text
把包含 D_1:t 的 posterior 当 prior
→ 再把 D_1:t 全部作为 likelihood 计算一次
```

本轮继续采用“固定源先验＋全部已观测目标数据重拟合”，不引入另一套在线贝叶斯实现。

### 4.5 空历史也要正确初始化

`build_source_prior()` 当前在 `not fits` 时返回长度 3 的数组。改为由当前完整 schema 决定维度，例如给该函数传入 `feature_ids`，直接返回相同长度的零均值／方差 4。

不要先构造一个假定只有两个坐标的先验，再依靠末尾补齐碰运气。七维支持只做接口兼容，不新增七维银行回放。

### 4.6 最小测试

- 非零且互不相同的覆盖系数，在新增失败中心后仍绑定原覆盖特征。
- 连续新增两个失败中心后，再次检查旧系数和方差。
- 仅改变特征排列时，同时排列系数，固定参数下的线性预测 `phi @ theta` 不变。
- 新特征均值为零时，新增前后固定均值参数产生的 logit 不变；**不要求积分后的概率完全不变**，因为新特征的非零先验方差本来可能改变预测不确定性。
- 重复 ID、丢失旧 ID、错误维度被明确拒绝。
- 空历史可建立与字典相同维度的先验。
- 同一批 observations 重拟合两次不增加样本计数，不把旧目标后验重复计入源先验。

---

## 5. P0-C：按任务类型定义 UCB 奖励

### 5.1 当前错误

当前选择器无论 `mode` 是什么都使用：

```python
ucb_reward[scene["template_id"]] += int(regression)
```

跨 agent 候选通常没有 `parent_pass`；因此碰撞不等于回归，UCB 收到零奖励。[S4]

### 5.2 目标定义

```python
from collections.abc import Mapping
from typing import Any


def task_reward(
    mode: str,
    outcome: Mapping[str, Any],
    parent_pass: bool | None = None,
) -> int:
    if mode not in {"regression", "cross_agent"}:
        raise ValueError(f"unsupported task mode: {mode}")
    if mode == "regression" and not isinstance(parent_pass, bool):
        raise ValueError("regression tasks require an explicit parent-pass label")

    valid_collision = (not outcome.get("inconclusive", False)
                       and outcome.get("ego_collision") is True)
    if mode == "cross_agent":
        return int(valid_collision)
    return int(parent_pass and valid_collision)
```

将本次回报存为独立字段：

```python
reward = task_reward(mode, outcome, parent_pass if mode == "regression" else None)
ucb_count[scene["template_id"]] += 1
ucb_reward[scene["template_id"]] += reward
```

输出同时包含：

```text
mode
valid_collision
regression                   # cross_agent 中使用 null／not_applicable
selection_reward
ucb_count_after
ucb_reward_after
```

- regression：新回归才是奖励。
- cross_agent：该 agent 的有效碰撞就是奖励，不要求存在父版本。
- 不确定结果消耗一次已查询预算、不给正奖励；不把它当安全标签训练。
- 每次查询只能更新一次 UCB 计数与奖励。
- `_summaries()` 与选择器采用相同任务语义；不能只改报告而不改学习奖励。

### 5.3 最小测试

用内存中的两个模板、若干固定标签构造 fake oracle，不运行仿真：

- cross_agent 无父标签但真实碰撞，奖励为 1。
- regression 父版本通过且目标碰撞，奖励为 1。
- regression 父版本未通过，奖励为 0；缺失父标签应被发现。
- inconclusive 不产生正奖励。
- 在跨 agent 的实际选择循环中，发现碰撞后 `ucb_rewards` 确实增加，而不是只测试辅助函数。

---

## 6. P1：保留失败信息，但不更改原始排名的主标签

### 6.1 修复在线观测记录的信息丢失

`selector_v2.py` 构建 `observed` 时，额外保留原 outcome 已有的：

```text
collision_partner_role
collision_time_s
public_signature
event_times
execution_contract_version
trajectory_path
```

字段映射要兼容旧 `collision_partner` 与新 `collision_partner_role`。缺失则保留 `null`，不要凭场景名称推断碰撞对象。

模式卡可以记录已观测的 `observed_partner_roles` 或 witness 摘要，继续保留失败—通过边，但本轮**不更换 RBF 的分组规则、似然或主优化目标**。

### 6.2 S02 只做准确解释

保留原银行的 `any_ego_collision` 主标签，不重命名为“撞静止目标”。报告按碰撞对象提供一个分解：

```text
lead / static / rear / unknown
```

旧 S02 的 `first_exit_s=0` 问题保留标记为不可靠字段，不使用 v4 轨迹替换 v2 的事件值或排名。若原行无对象标签，不必为了补标签重跑 episode。

### 6.3 不在本轮悄悄完成另一套方法

本轮修复后，“failure-based”至少应有一条可以追溯的信息链：

```text
真实历史失败记录
→ 实际进入某任务历史
→ 形成卡片／失效中心
→ 产生正确绑定的特征和先验
→ 与已查询目标结果共同决定下一次选择
```

不要求每个任务都必须有历史失败，也不以“有卡片 ID 出现在日志”代替模型真正使用该特征。单元测试需覆盖一个有历史失败的最小例子即可；没有历史失败的任务按实际状态报告。

---

## 7. 回放协议：只用已有结果，不重新组建更有利任务

### 7.1 唯一主回放配置

保持原任务、候选与顺序：

| 分组 | 任务 | 处理方式 |
|---|---|---|
| 旧 IDM 回归 | 3 个既有种子 × 3 个既有修改 | 修正参考历史过滤，候选／目标标签不变 |
| MOBIL 回归 | `mobil_ref_v2 → mobil_rear_guard_off_v2` | 保留零历史失败的事实，不从目标借用失败 |
| PPO 回归 | `ppo_ref_v2 → ppo_obs_age020_v2` | 保留零新增回归的事实 |
| 跨 agent | 原顺序 `mobil_ref_v2 → ppo_ref_v2` | 各方法独立历史分支，仅传递本分支已查询结果 |

方法保持原五个：

```text
Random
HistoryRank-UCB-v2
FailureDistance-v2
FBRT-NoMemory
FBRT-Memory
```

总逻辑预算保持 `B=20`，检查点保持 `@1/@5/@10/@20`。沿用原 Random 的 10 次重放及其 seed 规则；其他方法沿用原任务的 seed 规则，不增加搜索种子、不选最好的一次。

不得混入 v3/v4 初始状态审计样本，不反转 agent 顺序、不新增留一任务。本轮不是重新设计更有利的迁移基准。

### 7.2 对结果变化的解释边界

修复后完整回放只有一套配置；不要求逐项开关 bug 做大量消融。与旧保存结果比较时注明：**多个确定的代码问题共同修复，差值不能全部归因于其中单一修复。**

现有不同方法可能使用方法相关 seed；本轮为保持与旧结果的可追溯比较，不另改 seed 策略。不得用一次随机路径的局部优势声称稳定优势。

### 7.3 任务元数据要分“选择器可见”和“离线评估专用”

**选择器可见，在任务开始前计算：**

```text
source_history_count
source_history_failure_count
initial_pattern_count
history_kind = empty | pass_only | failure_and_pass | failure_only
source_build_ids
context_id
```

**离线 evaluator 专用，不传给选择器：**

```text
target_failure_pool_count
has_target_regression
oracle_failure_count_at_B = min(B, target_failure_pool_count)
```

禁止用完整目标池“有没有回归”“哪个模板失败多”决定在线选例、停止时间或 Memory／NoMemory 切换。

即使离线知道 PPO 池无回归，也保持原回放预算以输出对应路径；报告将其效率指标记为不适用，而不是把它伪装成算法失败。

### 7.4 历史快照和重复执行

- 从原始参考文件和被冻结的初始历史重建视图，避免原 archive 中先前运行写入的目标结果污染本轮起点。
- 旧目标记录始终 evaluator-only；不是因为文件叫 archive 就可以用作源数据。
- 跨 agent 保留原上下文过滤与 `_cross_agent_seed_records()` 的隔离意图。
- 同一方法的第一会话真实查询结果可写入其第二会话历史；不同方法、不同 repeat 不能共享后验或查询结果。
- 每次重新运行修复回放，从同一初始快照开始；不能继承上一次修复重放追加的目标观测。
- 原上下文精确过滤导致无历史时，报告 `empty`，不要取消物理上下文隔离以制造迁移。

---

## 8. 纯缓存入口：避免旧入口自动补仿真

### 8.1 新增目标 CLI

建议新增：

```text
method_chains/failure_memory_regression/repair_replay.py
```

目标命令：

```powershell
conda run -n metadrive python -m method_chains.failure_memory_regression.repair_replay --offline-only --output results/method_chains/failure_memory_regression/repair_20260926
```

这是需要实现的目标入口，当前任务书不声称它已经存在。

可另提供 `--check-inputs`，只读取输入并打印缺失文件与历史计数，不运行选择循环。不要增加几十个流程阶段。

### 8.2 必须采用的工程约束

- 纯回放模块只依赖数据加载、模式构建、模型、选择器和报告。
- 不调用 `import_stage()`、`compact_bank_stage()`、`smoke_stage()`、`audit_v4` 或 `_ensure_physical_episode()`。
- 避免导入带物理执行副作用或启动资源检查的顶层模块。必要时把纯数据任务构造提取为小函数，不复制整个执行器。
- 不以当前 registry／PPO 权重是否在本机存在来决定旧物理银行能否被回放；完整银行已经包含实际结果。
- 数据缺失立即返回 `MISSING_CACHE`，不能退化为生成缺失 episode。
- 使用独立源路径与输出路径，避免为重定向 ROOT 而意外重定向输入银行。
- 对原来可能创建环境或执行 episode 的接口，在纯回放测试中设置一个会抛出异常的替身，证明回放路径不会触碰它们。

### 8.3 缓存计账

新 ledger 至少包含：

```json
{
  "additional_physical_episodes": 0,
  "policy_training_runs": 0,
  "input_bank_fingerprints_unchanged": true,
  "completed_replay_tasks": 0,
  "missing_cache_tasks": [],
  "logical_query_count": 0,
  "model_fit_count": 0
}
```

上述零初始值由实际回放更新。`model_fit_count` 统计真实拟合调用次数，不再拿 summary 行数充当拟合次数。

---

## 9. 报告：分清“修复完成”与“方法是否有效”

### 9.1 每项任务保留的指标

```text
candidate_count
actual_queries
failure_pool_count                  # evaluator 专用统计
failure_count_at_1/5/10/20
first_failure_rank                  # 未发现为 null，显示为 >B
failure_recall_at_1/5/10/20          # 分母为零时为 null
initial_history_failure_count
initial_pattern_count
history_kind
```

未检测到的任务不丢弃。若展示首次发现的汇总，用检测率配合 `min(first_rank, B+1)` 等明确的截尾约定，不能只对成功任务取平均却不说明漏检。

### 9.2 必须区分的解释状态

| 状态 | 含义 | 不应写成 |
|---|---|---|
| `HAS_FAILURE_MEMORY` | 当前任务确有可用历史失败／模式 | 全局有卡片就自动满足 |
| `PASS_ONLY_HISTORY` | 有历史，但只有通过记录 | 已验证失效模式迁移 |
| `NO_COMPATIBLE_HISTORY` | 上下文过滤后无可用历史 | 方法失效／需要制造源数据 |
| `NO_TARGET_FAILURE_IN_POOL` | 当前完整目标池没有要发现的失败 | 算法漏检全部失败 |
| `BASELINE_AT_ORACLE_CEILING` | 相同 B 下某基线已找到全部可发现失败 | 继续要求更多发现数 |
| `MISSING_CACHE` | 对应文件／记录缺失 | 自动补仿真或推断标签 |

这些是解释标记，不是挑选有利任务的开关。全部任务仍在报告中保留；有失败任务的效率汇总与零失败任务的说明分开。

### 9.3 最终状态格式

废止新报告中单一的 `IMPLEMENTED_WITH_GAIN` 总结。输出两个互相独立字段：

```text
engineering_status = fixed_and_replayed | fixed_with_missing_cache | blocked_by_code_error
empirical_effect = gain | mixed | no_gain | not_applicable
```

`empirical_effect` 同时列出任务级差值，不能因为一项任务提高就把全部研究标为 gain。修复完成但 Memory 仍不如 NoMemory，是合法完整交付。

报告最后只回答：

1. 历史失败是否真正到达了方法？
2. 特征及先验是否按身份正确对应？
3. 每种任务的奖励是否正确？
4. 在现有任务中，修复后的 Memory 相对 NoMemory 和现有基线表现怎样？
5. 哪些任务没有历史失败／没有目标回归，因此不支持相应迁移结论？

不追加“请再运行更多算法或场景才能给出判断”的新执行清单。

---

## 10. 最小改动清单与执行顺序

### 修改文件

| 文件 | 改动 |
|---|---|
| `experiment_v2.py` | 修正旧历史过滤；把可复用的纯数据任务构造与物理执行分开；不覆盖原结果 |
| `pattern_memory.py` | 增加完整有序 feature schema；保持新增中心 ID 稳定 |
| `bayes_model.py` | 以 ID 对齐先验；空历史按实际 schema 初始化；防止目标观测重复计入 |
| `selector_v2.py` | 接通源先验 schema；按 mode 更新 UCB；保留已观测失败元信息 |
| `report_v2.py` 或小型修复报告函数 | 增加任务适用状态和旧／新对照，不更换主指标 |
| `repair_replay.py`（新增） | 只读现有银行、分支隔离、零仿真回放入口 |
| `tests/test_repair_offline.py`（新增） | 纯计算测试；不实例化仿真环境 |

### 执行顺序

**A. 读取并固定输入。**记录代码 HEAD、输入哈希，读出实际历史计数。只检查与本修复相关的数据，不再全面审计整个项目。

**B. 一次完成 P0-A/B/C。**增加最小测试；先用小型假记录验证路径，再开始银行回放。

**C. 完成 P1 与纯回放入口。**保留字段、解释零失败任务；关闭所有物理补跑路径。

**D. 一次完整缓存回放。**原五种方法、原任务、原预算。允许修复程序崩溃后续跑，不允许因为结果不正向而换参数再跑。

**E. 写报告并交付。**不论 gain/no_gain，都结束本轮。

### 测试目标命令

```powershell
conda run -n metadrive python -m pytest method_chains/failure_memory_regression/tests/test_repair_offline.py -q -p no:cacheprovider
```

测试文件集中覆盖以下九类性质即可，不要求建立新的大型验收框架：

1. 历史失败保留，候选仍限定父版本通过。
2. evaluator-only 与其他方法分支结果不可见。
3. 动态特征插入后非零系数／方差不串位。
4. 空历史和连续多次新增特征可处理。
5. 重复后验计算不重复累计目标数据。
6. 跨 agent UCB 碰撞奖励为正，回归奖励要求父版本通过。
7. 无目标失败时 recall 为 null，无历史失败时不伪造卡片。
8. 新观测中的碰撞对象等已有信息被保存，未知值不猜测。
9. 完整 fake-bank 回放不触发任何物理执行接口，输入文件不改写。

这些是修复正确性的单元测试，不是新增自动驾驶实验。

---

## 11. 最终交付物

新输出目录至少保留：

```text
repair_20260926/
    repair_report.md           # 修改事实、结果表、解释边界
    manifest.json              # 代码版本、输入哈希、配置、计账
    task_inputs.jsonl          # 实际进入方法的历史／卡片计数
    summary_by_task.csv        # 修复后的统一结果
    comparison_before_after.csv
    sessions/                 # 查询、更新、必要的分支快照
```

报告中附测试结果和 `git diff --stat`。不必生成大量 GIF、PPT、额外审计目录或“下一阶段”计划。

`comparison_before_after.csv` 直接读取旧保存结果与新结果比较；原结果缺失时记 `before_missing`，不要为补齐旧表重新运行原错误代码或模拟器。

代码交付与报告中应明确：

- 本轮新增物理执行为 0。
- 哪些问题已修正、哪些任务由于现有资料范围仍无法检验失败迁移。
- 是否仍存在 Memory 负收益；不承诺修 bug 后必胜。
- 旧场景及模型的适用性限制没有被伪装成已解决。

---

## 12. 可直接贴给 Codex 的总指令

> 在 SafeDL/META_LEARNING 当前工作区上，依据本任务书做最小代码修复，不重构为新研究框架。优先修复三项确定错误：`cache_stage()` 误删历史失败、动态 RBF 特征与源先验系数错位、cross_agent 的 UCB 错用 regression 奖励。历史建模必须保留有效失败和通过，但回归候选仍是父版本通过集合；所有源先验按稳定 feature ID 对齐，目标 observations 每次只计一次；UCB 根据任务模式使用正确收益。补充失败对象信息保存与任务适用状态，不改变现有碰撞主标签。只读原 legacy/core 和 memory_v2 物理银行，新增一个 offline-only 入口，输出到 repair_20260926，原数据与原报告不覆盖。本轮禁止新增仿真、策略训练、权重下载、场景扩展、算法扩展和调参搜索。完成纯计算测试后，按原任务／候选／预算／随机重复规则仅做一次修复后回放，交付代码差异和结果。数据缺失标记 MISSING_CACHE；没有历史失败、没有目标回归分别说明。修复后无增益也正常交付，不继续追加实验。

---

## 参考源码与报告（固定到已审阅提交）

本任务书的修复依据是下面的实际源码和已有报告，不要求 Codex 重新做文献调研。前期关于低秩元学习、规范认证、多阶段性能门槛的旧设计不属于本次修复范围。

- **[S1] 实验入口、历史过滤、任务视图、银行路径**  
  https://github.com/SafeDL/META_LEARNING/blob/2aa313ecc94779d8443ddc850b5ff3beabb546c2/method_chains/failure_memory_regression/experiment_v2.py
- **[S2] 失效模式、RBF 特征顺序、新中心追加**  
  https://github.com/SafeDL/META_LEARNING/blob/2aa313ecc94779d8443ddc850b5ff3beabb546c2/method_chains/failure_memory_regression/pattern_memory.py
- **[S3] 源先验、extend_prior、目标后验**  
  https://github.com/SafeDL/META_LEARNING/blob/2aa313ecc94779d8443ddc850b5ff3beabb546c2/method_chains/failure_memory_regression/bayes_model.py
- **[S4] 选择器、回归标签、UCB 奖励、observed 记录**  
  https://github.com/SafeDL/META_LEARNING/blob/2aa313ecc94779d8443ddc850b5ff3beabb546c2/method_chains/failure_memory_regression/selector_v2.py
- **[S5] 原 memory_v2 结果报告**  
  https://github.com/SafeDL/META_LEARNING/blob/2aa313ecc94779d8443ddc850b5ff3beabb546c2/results/method_chains/failure_memory_regression/memory_v2/report.md
- **[S6] 原验收状态及零失败／零回归诊断**  
  https://github.com/SafeDL/META_LEARNING/blob/2aa313ecc94779d8443ddc850b5ff3beabb546c2/results/method_chains/failure_memory_regression/memory_v2/acceptance.json
- **[S7] 已有场景与初始状态审计；不在本轮重跑**  
  https://github.com/SafeDL/META_LEARNING/blob/2aa313ecc94779d8443ddc850b5ff3beabb546c2/results/method_chains/failure_memory_regression/memory_v2/validation_audit_v4/README.md
- **[S8] 旧数据解析与 visibility 定义**  
  https://github.com/SafeDL/META_LEARNING/blob/2aa313ecc94779d8443ddc850b5ff3beabb546c2/method_chains/failure_memory_regression/archive_v2.py

**交付底线不是“必须赢”，而是“历史失败确实被使用、数学对象正确对应、任务收益定义正确、已有结果得到可信解释”。本轮到此为止。**
