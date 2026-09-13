# PR-BRVT：版本回归语义修订、代码修改目标与最小验证协议

**日期：2026-09-13**  
**核对仓库：SafeDL/META_LEARNING**  
**核对提交：`1d6e7d244bbae43ec23e728394ef0f4f010d6884`**  
**文档性质：待实施的代码修改说明，不是已实现补丁或新增实验结果。**

> **本轮唯一目标：保留现有低秩先验、贝叶斯更新和 Highway-env 执行器，让“历史版本用于学习、上一版本用于判定回退、新版本用于有限预算测试”真正进入代码。**
>
> 不再引入新的方法名称，不增加神经网络，不恢复固定 K 诊断阶段，不扫描 acquisition 权重。先把回归测试的对象和评价定义做对，再运行一次冻结的版本序列验证。

---

## 1. 先把研究问题固定下来

### 1.1 一句话定义

给定同一控制器产品的历史版本测试档案，针对新版本，在最多 20 次新版本测试内，优先发现**上一版本在相同场景下没有触发关键安全事件，而新版本触发了关键安全事件**的场景。

这里的“关键安全事件”继续采用当前仓库的 `collision or near_miss`，不在本轮修改其判据。

通俗地说：

> 历史版本教我们怎样预测；上一版本告诉我们哪些地方原来没问题；新版本的实际测试告诉我们哪些地方现在坏了。

### 1.2 两类关心的问题，统一为一个主目标

在历史结果完整的已知场景池内，回退场景再分成两个互斥子类：

| 子类 | 过去版本的记录 | 上一版本 | 新版本 | 本轮如何处理 |
|---|---|---|---|---|
| 档案内首次出现的失效 | 所有已归档历史版本均无关键事件 | 无关键事件 | 有关键事件 | 计入回退总数，标记 `new_in_archive` |
| 重新出现的失效 | 更早版本至少出现过关键事件 | 无关键事件 | 有关键事件 | 计入回退总数，标记 `reintroduced` |
| 持续存在的旧失效 | 不限 | 有关键事件 | 有关键事件 | 不是本轮主目标中的 safe→critical 回退 |
| 已修复的失效 | 不限 | 有关键事件 | 无关键事件 | 记录为修复，不计入回退 |
| 稳定无关键事件 | 不限 | 无关键事件 | 无关键事件 | 不计入回退 |

**重要边界：**

- “档案内首次出现”不等于已经证明一个全新的软件根因；本轮统计的是失效场景，不是去重后的软件 bug。
- 未测试过的旧版本/场景不能被当成“曾经安全”。缺失标签是 `unknown`，不是 `False`。
- 当前库内 `near_miss→collision` 属于严重程度恶化，但不是二元 `safe→critical` 转换。本轮保留原始结果供分析，不额外扩大主目标。
- 场景库以外、全新功能/运行设计域中的问题发现，不由当前离散锚点模型覆盖。本文不能宣称解决所有类型的新问题。
- “无关键事件”只是当前仿真判据下的结果，不是现实安全认证。

### 1.3 不再把两个 reference 混成一个加权分数

历史 population 和直接前一版本承担不同职责，而不是再加两个权重：

```text
多个历史版本 ──→ 低秩函数先验 ──→ 新版本风险预测
上一版本结果 ──→ 原先无关键事件的候选集合
新版本观测   ──→ 更新后验 + 确认真实回退
```

旧的 `p_target × historical_rarity` 会降低“早期坏过、上一版修好、新版又坏”的优先级。这与本轮回归目标不一致。

**本轮将历史罕见度从主 acquisition 中移除，但保留历史数据对函数先验的作用。**历史首次出现与重新出现只作为回退场景的分类，不互相降权。

---

## 2. 当前代码已经有什么，缺什么

以下事实来自核对提交中的源码和已提交结果；后文标明的新文件、参数提案和验收规则均为本次修改建议。

| 当前文件 | 已有能力 | 当前缺口 |
|---|---|---|
| `mvr/highway/diva/low_rank_prior.py` | source mean、中心化 SVD、latent covariance | source 的时间顺序由调用者负责；没有版本谱系 |
| `mvr/highway/diva/posterior.py` | 使用已揭示结果做解析后验更新 | 不负责区分历史版本与未来版本 |
| `mvr/highway/diva/regression_probability.py` | 计算超过 0.75 的后验预测概率 | 不是经过验证的现实事故概率 |
| `mvr/highway/diva/regression_reference.py` | 历史失效比例及 Beta 平滑 | 没有直接前一版本的已知安全掩码 |
| `mvr/highway/diva/regression_mining.py` | 顺序选择、揭示、更新 | 仍按历史 rarity 加权，评价也是 population-relative |
| `mvr/highway/experiments/run_loso_regression_mining.py` | 留一 SUT 比较 | `np.delete(target_row)` 会在版本序列中使用未来版本 |
| `mvr/highway/sut/idm_profiles.py` | 3 IDM + 3 FVDM 的固定参数控制器 | A–F 是异构系统，不是连续发布版本 |
| `mvr/highway/data/response_bank.py` | 同锚点、同 seed 的缓存构建与读取 | builder 写死 `PROFILE_NAMES/get_profile`；没有版本元数据 |
| `mvr/highway/scripts/run_pr_brvt_mvp.py` | 读取修复后的 E6 bank 做离线回放 | 输入固定为 A–F 的旧 bank |

现有 PR-BRVT 的已提交汇总为：RCS@20=5.6548、TSF@20=5.3333、Raw Score@20=15.25，Gate BRVT-1 为 `continue`。这是**开发集上的群体相对脆弱性发现结果**，不是本次版本回归实验的成绩。[S1–S7]

另一个需要收紧的解释：当前 Population-Novelty baseline 使用 `mean_vulnerability × rarity`，而主方法使用 `posterior_critical_probability × rarity`。二者不只相差“有无 target feedback”，还相差概率变换。因此旧结果不能单独证明所有增益都来自贝叶斯更新。下面的 `Frozen-History` 对照只关闭更新，消除这一混淆。[S2]

**保留所有旧源码、旧 JSON 和旧 Gate；不覆盖，不把旧 `continue` 自动转移到新任务。**

---

## 3. 唯一的主目标与 acquisition

### 3.1 符号

- `j`：同一控制器谱系中的版本序号。
- `t`：当前新版本已经执行的测试次数。
- `x`：固定候选池中的一个场景；包括两个连续参数及 interaction mode。
- `F_j(x)`：版本 j 在 x 上是否发生关键事件，即 `collision or near_miss`。
- `y_j(x)`：当前连续 vulnerability，用于后验更新，不替代正式事件标签。
- `H_j`：仅包含该谱系中 `version_order < j` 的已归档历史版本。
- `D_t`：新版本已经执行的 t 个测试结果。

### 3.2 真实回退标签

在上一版本和新版本的结果均有效、场景及 seed 对齐时：

\[
G_j(x)=\mathbf{1}[F_{j-1}(x)=0\ \land\ F_j(x)=1].
\]

这一定义只用真实测试结果，不用预测概率，也不用 population rarity。

设上一版本有效且已知的无关键事件掩码为：

\[
E_j(x)=\mathbf{1}[\text{previous outcome known and valid}\ \land\ F_{j-1}(x)=0].
\]

### 3.3 保留现有先验与后验

仍使用：

\[
y_j(x)\approx\mu_{H_j}(x)+b_{H_j}(x)^\top z_j,
\qquad z_j\mid D_t\sim\mathcal N(m_t,\Sigma_t).
\]

继续调用 `LowRankPrior.fit()`、`adapt_posterior()` 和 `critical_probability()`。

冻结：`prior_rank=2`、`critical_threshold=0.75`、`observation_noise=0.03`。

不把先验偷偷改为 `previous_response + basis @ latent`：那会改变统计模型，需要另行论证。本轮只修正版本语义、候选集和评价目标。

### 3.4 新的选择规则

\[
p_t(x)=P[y_j(x)\ge0.75\mid H_j,D_t],
\]

\[
x_{t+1}=\arg\max_{x\in\mathcal E_j\setminus Q_t}p_t(x),
\]

其中 `E_j` 是上一版已确认无关键事件的候选集合，`Q_t` 是已测试集合。

等价理解：**只在上一版没出问题的场景中，优先选择新版最可能出问题的场景。**

实现时，非候选位置必须设为 `-np.inf`，不能简单乘 0；否则在所有候选概率接近 0 时，平局可能选到被排除的场景。

本轮不增加：历史罕见度权重、shared-risk bonus、uncertainty bonus、固定 K、额外 acquisition 超参数。

### 3.5 这为什么同时覆盖两类回退

两个场景都满足上一版本无关键事件，其中一个早期从未失败，另一个早期失败过。只要新版本风险预测相同，本轮不因为第二个场景“早期经常失败”而降低其测试优先级。

两类场景的区别由真实发现之后的档案分类表达，而不是靠两套不同 acquisition 争抢预算。

### 3.6 不要给这个公式赋予超出其能力的含义

- 这是单步贪心选择，不是有限预算全局最优策略，也不等于最大化未来信息价值。
- 后验概率包含当前模型及其固定噪声假设；没有概率校准证明时，不写“90% 的真实回退概率”。
- 新版均值可能偏离旧版均值，低秩模型也可能漏掉全新的局部失效机制；版本关系本身不保证可迁移。
- 当某处所有历史版本响应都相同，中心化 basis 在该处可能几乎为零。这种从未在历史差异中出现的漏洞，不保证能由几次其他场景观测推断出来。
- 本轮是小预算顺序自适应测试，不再等同于“严格 K=4 support 后冻结”的 few-shot 协议。

---

## 4. 文件级修改清单

### 4.1 修改现有文件，不破坏旧入口

| 文件 | 明确修改要求 | 不允许做的事 |
|---|---|---|
| `mvr/highway/config.py` | 新增独立 `VersionRegressionConfig`；保留旧两个配置类 | 不改变旧默认值来让旧 Gate 变成新 Gate |
| `mvr/highway/data/response_bank.py` | 给 `build_response_bank()` 增加可选 `profiles`；默认行为不变 | 不将 A–F 全局替换为版本控制器 |
| `docs/PR_DVM_HighwayEnv_MVP_Target_20260913.md` | 增加“历史 population 方案”说明，链接本轮新协议 | 不删除旧负结果或改写其评价目标 |

`profiles` 建议增加在原参数之后：

```python
# 建议接口；尚未在仓库实施。
def build_response_bank(
    anchors: np.ndarray,
    seed: int,
    modes: np.ndarray | None = None,
    profiles: tuple[SUTProfile, ...] | None = None,
) -> ResponseBank:
    ...
```

`profiles is None` 时沿用旧 registry；传入时按实际 profile.name 建立行名，直接将 profile 交给 `run_cutin_episode()`，不要再用旧 `get_profile(name)` 查询不存在的版本名。验证名字唯一、controller family 合法、所有矩阵尺寸正确。

### 4.2 新增文件及职责

```text
mvr/highway/
  configs/
    version_regression_mvp.yaml       # 新协议冻结参数
    version_lineages_v1.json          # 版本谱系、具体参数、变更意图
  sut/
    version_lineage.py                # 读取/校验 manifest，构造 SUTProfile
  data/
    version_bank.py                   # ResponseBank + 版本元数据、哈希和有效性
  diva/
    version_reference.py              # 上一版安全掩码、历史首次/重新出现分类
    version_mining.py                 # 选择—揭示—更新；只访问单条新结果
  experiments/
    run_version_regression.py         # chronological holdout、四方法、公平评分
  scripts/
    run_version_regression_mvp.py     # freeze / build / evaluate 三个子命令
  tests/
    test_version_regression.py        # 语义、泄漏、预算、旧功能兼容

docs/
  PR_BRVT_Version_Regression_Target.md # 本文放入仓库的建议位置
```

这是同一个最小工作流，不是八个新研究模块。不要额外建立 Teacher、版本识别网络或另一套 surrogate。

### 4.3 保持不变的核心代码

```text
mvr/highway/envs/cutin_env.py
mvr/highway/sut/idm_profiles.py
mvr/highway/diva/low_rank_prior.py
mvr/highway/diva/posterior.py
mvr/highway/diva/regression_probability.py
mvr/highway/diva/outcomes.py
mvr/highway/experiments/run_loso_regression_mining.py
mvr/highway/scripts/run_pr_brvt_mvp.py
```

环境的动作修复、20 Hz 仿真、场景范围、两种 mode、正式事件与 vulnerability 的定义继续保留。旧 LOSO 入口继续解释为跨系统开发实验，不要在原函数中静默改为时间划分。[S3–S9]

---

## 5. 数据结构与信息隔离

### 5.1 每个版本必须有元数据

`version_lineages_v1.json` 中每个版本记录：

```json
{
  "lineage_id": "idm_dev",
  "version_id": "idm_dev_v4",
  "version_order": 4,
  "predecessor_id": "idm_dev_v3",
  "controller": "IDM",
  "profile": {},
  "change_note": "相对 v3 减少反应等待；只是改进意图，尚非效果结论",
  "origin": "controlled_parameter_evolution"
}
```

`profile` 实际必须填入第 8 节给出的完整有效参数，不允许以空字典运行。运行产物另记录 git SHA、manifest SHA256、场景/seed 哈希、执行器版本和依赖版本。

必须验证：同一 lineage 的顺序唯一、predecessor 真的是直接前一版本、无环、无跨家族伪时间链。排序依据只能是版本号/明确的开发顺序，不能依据失效率。

### 5.2 `VersionedBank` 不直接复制旧 ResponseBank

建议使用组合：

```python
@dataclass(frozen=True)
class VersionedBank:
    responses: ResponseBank
    versions: tuple[VersionMeta, ...]
    observed_mask: np.ndarray  # [version, anchor]
    valid_mask: np.ndarray     # [version, anchor]
    episode_seeds: np.ndarray  # [anchor]
    metadata: dict
```

旧 NPZ 结构继续使用；额外版本信息与 provenance 存入 JSON/独立数组，`allow_pickle=False`。

`valid_mask` 在这里指一次结果按冻结协议有效完成/以合法碰撞终止，不等于现实可避免性证明。碰撞 episode 的 `completed=False` 是正常的，不能因此丢弃所有碰撞。

本轮正式 bank 要求完整、有效、有限的 vulnerability 与正确事件标记；构建异常必须报错或明确记录无效，不得填零后继续当作安全。无穷大 TTC 可以表示未出现有限 TTC，不与 NaN 混淆。[S8–S9]

### 5.3 选择器只看历史和已揭示反馈

新增一个 `TargetReplay.reveal(anchor_index)` 接口，内部持有 target 行；主方法只能拿到单条：

```text
anchor_index
vulnerability
collision
near_miss
valid
```

选择器不接收完整 target 数组，也不接收该版本的全池回退数量、未来版本、profile 参数或完整轨迹。完整 target 行仅供离线 evaluator 在所有选择完成后评分。

`TargetReplay` 自行检查预算和重复查询。日志中保存每次 query 的顺序及观测，形成可重放记录。已有 Python 私有字段不构成安全沙箱，仍需通过下面的干预测试检查调用链。

### 5.4 明确分离两个“学习对象”

- SUT 是被测试的控制器，它在测试期间必须固定。
- 更新的是测试器的后验，不是新版本 SUT 的策略、参数或控制器权重。

“无神经网络训练”不等于完全没有学习：每个历史切片仍然做 SVD/协方差估计；target 阶段做解析统计更新。

---

## 6. 主循环参考逻辑

```python
# 接口级伪代码：约束实现顺序，不是声称已有的可执行入口。
for lineage in manifest.lineages:
    for target_order in (4, 5, 6):
        history = bank.history(lineage, before=target_order)
        previous = history.immediate_predecessor

        prior = LowRankPrior.fit(history.vulnerability, rank=2)
        reference = build_version_reference(history)
        posterior = adapt_posterior(prior, empty_indices, empty_values)
        replay = evaluator.open_target(lineage, target_order, budget=20)

        selected, observed = [], []
        for _ in range(min(20, reference.eligible_count)):
            probabilities = critical_probability(
                prior, posterior, threshold=0.75, observation_noise=0.03
            )
            scores = np.full(probabilities.shape, -np.inf)
            allowed = reference.previous_safe.copy()
            allowed[selected] = False
            scores[allowed] = probabilities[allowed]
            chosen = deterministic_argmax(scores)

            outcome = replay.reveal(chosen)  # 到此才访问新版本值
            assert outcome.valid
            selected.append(chosen)
            observed.append(outcome.vulnerability)
            record_actual_regression(reference, chosen, outcome)
            posterior = adapt_posterior(
                prior, selected, observed, observation_noise=0.03
            )

        evaluator.score_completed_trace(replay.trace)
```

实现要求：

- 平局采用固定 anchor index 次序，全部方法一致。
- `eligible_count=0` 时输出 `no_eligible_previous_safe_cases`，实际 target 调用为 0，不伪造 20 次。
- 候选不足 20 时全部方法使用相同 `B_eff=min(20, eligible_count)`。
- 概率低不自动终止测试。本轮没有经过验证的“没有回退”置信证明或自适应停止阈值。
- 预算耗尽但没发现回退，输出 `no_regression_observed_within_budget`，不能输出“新版安全”或“没有回退”。
- 不允许读取真实 target 全池答案后选择停止、过滤版本或改变 `B_eff`。

---

## 7. 由 LOSO 改为时间顺序验证

LOSO 在当前项目是 **Leave-One-SUT-Out，留一被测系统验证**；它仍用于旧 A–F 数据，不作为新版本实验的主划分。

新协议只做：

| 一次测试任务 | 可用于先验的已归档版本 | 直接前一版本 | 未见 target |
|---|---|---|---|
| 第 1 次 | v1、v2、v3 | v3 | v4 |
| 第 2 次 | v1、v2、v3、v4 | v4 | v5 |
| 第 3 次 | v1、v2、v3、v4、v5 | v5 | v6 |

两条 lineage 分别独立执行，不互相混用历史。不允许 `all_versions - target`，必须是 `version_order < target_order`。

**完整历史档案假设必须写清楚：**到了 v5 测试时，v4 的完整 128 场景档案被假定已经由历史测试流程建立。不能把前一次只查询了 20 条的新版本结果，暗中变成免费的 128 条。下面的 bank 建设预算显式支付完整档案成本。

这是“每次新版本小预算测试、同时已有完整历史档案”的模拟，不是“整个产品生命周期每版永远只允许 20 次测试”。后一个问题需要不完整历史矩阵方法，不在本轮实现。

补充解释：第一折只有 3 个 source 版本，中心化矩阵的秩至多为 2，因此 rank-2 EVR=100% 可能只是代数必然，不能再作为迁移成功 Gate。版本太相似或 latent covariance 几乎退化时如实记录，不靠提高秩制造增益。[S6]

---

## 8. 版本 SUT 的最小构造提案

### 8.1 不把 A–F 改名为 v1–v6

新建两条受控参数演化谱系：`idm_dev_v1…v6`、`fvdm_dev_v1…v6`。复用现有控制器类，不从零训练驾驶策略。

**以下具体数值是本文件提出的初始构造，不是已运行、已证明改进、或保证能制造回退的配置。**目标是给实现者一套可以冻结的输入，避免反复根据主方法成绩选 SUT。[S10]

### 8.2 IDM 谱系：保持车辆能力不变，改变软件控制参数

固定参数：

```text
controller=IDM
max_brake=6.0
comfort_acceleration=3.0
target_speed=27.0
```

| 版本 | time_wanted | desired_gap | reaction_delay | 相对上一版的开发意图 |
|---|---:|---:|---:|---|
| v1 | 1.8 | 5.0 | 0.40 | 初始控制配置 |
| v2 | 1.8 | 5.0 | 0.25 | 缩短首次检测前车后的等待 |
| v3 | 1.6 | 5.0 | 0.25 | 调整跟车效率与保守性 |
| v4 | 1.6 | 5.0 | 0.10 | 再次缩短等待 |
| v5 | 1.3 | 4.5 | 0.10 | 更积极的间距配置 |
| v6 | 1.4 | 4.5 | 0.05 | 反应与间距的联合修订 |

### 8.3 FVDM 谱系

固定参数：

```text
controller=FVDM
max_brake=6.0
comfort_acceleration=3.0
target_speed=27.0
fvdm_transition_gap=5.0
```

| 版本 | desired_gap | reaction_delay | fvdm_sensitivity | fvdm_velocity_gain | 开发意图 |
|---|---:|---:|---:|---:|---|
| v1 | 6.0 | 0.40 | 0.45 | 0.45 | 初始控制配置 |
| v2 | 6.0 | 0.25 | 0.45 | 0.45 | 缩短等待 |
| v3 | 6.0 | 0.25 | 0.45 | 0.60 | 增强相对速度反馈 |
| v4 | 6.0 | 0.25 | 0.60 | 0.60 | 调整跟驰响应 |
| v5 | 5.0 | 0.25 | 0.60 | 0.60 | 更积极的间距配置 |
| v6 | 5.0 | 0.10 | 0.60 | 0.75 | 反应与相对速度反馈修订 |

未列字段沿用现有 `SUTProfile` 默认值；加载后必须把完整解析参数写入冻结 manifest，不能只保存覆盖项。

### 8.4 如何解释“连续改进”

这两条首先是**连续开发/修改序列**，不是保证单调更安全的序列。

在与主评价不重叠的固定 32 个校准场景上，记录每版已有的 failure rate 与 mean vulnerability，检查演化趋势。不得对版本按测试结果重新排序。

本轮不增加自动优化器或循环搜索 SUT 参数。首次冻结后若校准没有显示预期改善：保留该结果，把声明改为“受控版本演化”，不要声称“持续改进”，更不要挑选能让 PR-BRVT 获胜的版本组合。

要明确区分：

- **连续开发**：有确定的版本顺序和变更记录。
- **总体改善**：独立校准/评价中 aggregate 指标实际改善。
- **逐场景单调改善**：所有旧安全场景仍安全；如果成立，就可能根本没有回退。

在同一固定场景集合上，有下面的计数恒等式：

\[
N_{\text{critical,new}}-N_{\text{critical,prev}}
=N_{\text{regressions}}-N_{\text{repairs}}.
\]

因此总体事故变少可以同时伴随少量回退；但**不能要求人为必须出现回退**。没有回退的版本同样保留。

### 8.5 几个当前控制器实现的边界

- 当前 `reaction_delay` 是首次检测前车后的特定等待机制，不是完整感知延迟或持续观测滞后模型。
- 当前 IDM 的 `max_brake` 同时进入加速度截断和 desired-gap 公式；增加它不能简单解释为“所有场景更安全”。本提案把它固定，减少车辆能力变化与软件变化混淆。
- 参数化 IDM/FVDM 只能支持受控控制器版本实验，不能宣称真实神经网络持续学习或完整 ADS 产品发布验证。
- 不用 target 版本 ID、场景编号或回退标签在 SUT 内硬编码“此处故意撞车”。故障注入若将来确有需要，必须单独标注，不能冒充自然版本退化。

---

## 9. 冻结配置与仿真成本

### 9.1 建议配置

```yaml
schema: pr_brvt_version_regression_v1
base_commit: 1d6e7d244bbae43ec23e728394ef0f4f010d6884

lineages: [idm_dev, fvdm_dev]
versions_per_lineage: 6
target_version_orders: [4, 5, 6]
minimum_history_versions: 3

scenario:
  evaluation_anchors: 128
  calibration_anchors: 32
  evaluation_seed: 2026091303
  calibration_seed: 2026091301
  modes: [fast_intrusion, cutin_braking]
  generator: generate_dual_mode_anchor_bank
  episode_seed_rule: anchor_seed_plus_anchor_index

model:
  prior_rank: 2
  critical_threshold: 0.75
  observation_noise: 0.03
  fit_sources: same_lineage_strictly_earlier_versions

protocol:
  total_target_budget: 20
  selection_scope: previous_known_valid_noncritical
  update_after_each_reveal: true
  direct_target_identity_input: false
  missing_previous_is_safe: false
  target_outcome_access: reveal_only
  early_stop_by_predicted_probability: false

comparison:
  methods: [Random-Eligible, Previous-Boundary, Frozen-History, Sequential-History]
  random_repeats: 20
  tie_break: ascending_anchor_index

reporting:
  primary: regression_count_at_budget
  descriptive: [new_in_archive_count, reintroduced_count, regression_recall, raw_critical_score]
  keep_legacy_rcs_tsf: false
  overwrite_existing_run: false
```

两个 seed 只是预先固定的随机数种子，不表示日期约束。沿用当前 Sobol 范围：`initial_gap=[5,40]`、`relative_speed=[-8,2]`，沿用当前 mode 生成方式；两个集合不得包含同一个完整场景及 seed。[S7]

### 9.2 成本不能再写成“全部零仿真”

| 工作 | 新增 Highway-env 调用 |
|---|---:|
| 纯逻辑单元测试、已有数组兼容检查 | 0 |
| 两条谱系 × 6 版 × 32 校准场景 | 384 |
| 两条谱系 × 6 版 × 128 主评价场景 | 1536 |
| 四方法及随机重复在同一新 bank 上回放 | 0 |
| **一次完整新 bank 建设** | **1920** |

不包含已有环境单元测试自身执行的少量 episode，运行器应另行记账。GPU/驾驶策略梯度训练为 0；统计先验仍在 CPU 上拟合。

同一 `(version, anchor, seed, executor_hash, profile_hash)` 仅执行一次。所有 baseline 重用该结果，不为每个方法重复跑仿真。

`B=20` 指每个目标测试任务的可见新版本调用上限；完整历史档案和离线上限标签建设是额外成本，不能隐藏在“20 次”之内。

旧 A–F bank 可用于兼容检查，但不能不付新仿真成本就变成这 12 个版本的数据。

---

## 10. 只保留四个公平对照

所有方法使用**完全相同的上一版已确认无关键事件候选集**、相同 `B_eff`、同一 outcome oracle。不要让主方法过滤旧失败，而 baseline 仍在全场景池浪费预算。

| 方法 | 在相同 eligible 集合内如何选择 | 要检验什么 |
|---|---|---|
| Random-Eligible | 无放回均匀随机 | 没有优先排序时的表现 |
| Previous-Boundary | 按上一版真实 vulnerability 从高到低 | 只测试上一版离危险最近的场景是否已经足够 |
| Frozen-History | 初始化历史先验后，按同一个 `critical_probability()` 排序；始终不更新 | 多版本历史先验能做多少；精确隔离 target feedback |
| Sequential-History（主方法） | 同一初始先验和概率公式；每条新观测后更新 | 在线反馈是否提高真实回退发现 |

关键约束：

- Frozen-History 与主方法初始排序必须一致；两者唯一的方法差异是是否根据已揭示新版本结果更新后验。
- Previous-Boundary 不需要“预测上一版是否安全”：上一版记录已经可见。它在 eligible 集合内按 `previous_vulnerability` 排序即可。
- 不把 Gaussian-tail 变换本身的收益归因于 adaptation。
- 三个确定性方法各执行一次完整回放；Random 重复 20 次。随机重复不是 20 个独立 SUT。
- 旧 Shared Prior / Population-Novelty 的结果留在旧实验中，不作为新任务唯一强基线。

---

## 11. 指标只保留一个主计数，两类解释

### 11.1 主指标：`RegressionCount@B`

\[
\mathrm{RegressionCount}@B=\sum_{t=1}^{B_{\mathrm{eff}}}G_j(x_t).
\]

中文：**最多 B 次测试中确认了多少个“上一版无关键事件、新版有关键事件”的场景。**

每个 critical 场景记 1，不因碰撞/近失严重程度使用不同主权重。这样与当前 `P(critical)` acquisition 一致。碰撞、近失分别记录在明细中。

### 11.2 两类解释性计数

```text
new_in_archive_count
reintroduced_count
```

在完整历史与有效成对结果下必须满足：

```text
regression_count == new_in_archive_count + reintroduced_count
```

历史中从未出现只表示“在已归档版本上从未观察到”，不是“证明历史永远安全”。无需再使用 `pi_history <= 0.2` 定义本轮主回退标签。

### 11.3 版本越来越强时怎样读结果

离线 evaluator 在回放结束后计算整个 eligible bank 的真实回退数量 `available_regressions`。这是标注辅助，不准给选择器使用。

当该数大于 0 时，附带：

\[
\mathrm{RegressionRecall}@B
=\frac{\mathrm{RegressionCount}@B}{\mathrm{available\_regressions}}.
\]

它只是“已经存在的回退找到了多少比例”。新版只有两个回退、测到两个，应解释为该有限候选池内找全，而不是嫌事故数太低。

`available_regressions=0` 时 recall 写 `null/NA`，不写 0 或 1；该版本仍保留在表中，但不参与有回退版本的平均 recall。

Raw Critical Score 可作辅助记录；RCS/TSF 保留在旧实验目录，不成为新版本协议的通过条件。尤其不再把历史稀有度加权分数叫作“真实新增 bug 数”。

### 11.4 没发现不等于不存在

主方法只根据实际事件确认回退，不把 `p_t(x)` 大于某个值当作已发现回退。零发现版本的正确结论是：**预算内未观察到回退**。

如果离线全池标签也为零，只能说该冻结候选池和 seed 上没有观察到回退，不能外推到连续场景空间或现实道路。

---

## 12. 单一验证流程与验收规则

### 12.1 工程正确性是前提，不是论文效果

下列条件全部满足才生成科学结果汇总：

- 时间切片、predecessor、事件语义和哈希检查通过；
- 不读取未来/未查询 target 结果；
- 所有方法预算、eligible mask 和配对 seed 一致；
- 旧入口仍可运行；
- 不覆盖旧结果，不在评价后重选参数。

任何一项失败输出 `invalid_run`，先修工程，不解释算法优劣。

### 12.2 本轮只回答两个方法问题

1. 主方法相对于 **Frozen-History**，是否增加真实回退发现？
2. 主方法相对于 **Previous-Boundary**，是否表明多版本历史学习比最近版本的简单排序更有用？

不再要求 RCS 提升、Raw 保留 90%、每个新版本都必须有新漏洞，或固定 4/6 版本必须严格获胜。

### 12.3 可执行的开发判定

为避免程序只有“成功/失败”并诱导换数据，汇总器固定输出以下状态之一：

| 状态 | 判定条件 | 含义与后续动作 |
|---|---|---|
| `invalid_run` | 工程协议不成立 | 修复实现，不能讨论效果 |
| `insufficient_regression_evidence` | 两条谱系没有各自至少一个含回退任务；或在本预算下全部候选已被穷举 | 工程可用，但本次数据不足以支持跨谱系方法优势；不自动造故障或扩大 bank |
| `no_adaptation_gain` | 有效含回退任务上主方法平均 recall 不严格高于 Frozen-History | 当前顺序更新未显示净价值；保留负结果，不调权重 |
| `no_advantage_over_previous` | 超过 Frozen-History，但未超过 Previous-Boundary | 还不能主张多版本方法优于简单最近版本参考 |
| `provisional_support` | 超过上述两个对照，且两条谱系各自平均 recall 相对两对照均不为负 | 受控版本场景下得到初步正证据；不是统计显著性或真实产品认证 |
| `lineage_limited_support` | 总体超过两对照，但至少一条谱系平均增益为负 | 报告适用范围不一致，不包装成普适方法 |

比较时先在每个 target transition 内得到结果，再对有回退的 transition 取平均；随机方法先在同一 transition 内平均。使用固定数值容差 `1e-12` 判断浮点等值。

主计数、每谱系表现及全部零回退任务同时公开。`provisional_support` 只是预先声明的开发决策，不等于显著性检验。“两个谱系各一个含回退任务”的条件仅防止用单一案例代表全部，并不是足够的论文样本量。

这些任务共享历史版本、使用同一锚点池，不相互独立；曲线阴影如显示标准差，必须标为跨任务离散程度，不能称 95% CI。不得把 scenario 数、时间步数或随机重复当作独立版本样本。

### 12.4 不允许自动转入的事项

任一结果都不自动触发扩大模型、增加场景模式、搜索更多版本参数、替换评分阈值或迁移 MetaDrive。完成这一组冻结回放后即输出报告并停止本轮。

---

## 13. 必须通过的单元测试

集中写入 `test_version_regression.py`，大部分使用小数组，不运行仿真。

| 测试 | 必须断言的行为 |
|---|---|
| 版本链 | predecessor 顺序正确；循环、重复版本号、跨家族假 lineage 被拒绝 |
| 时间隔离 | 修改 v6 完整结果不能改变 v4 的先验、reference 或任一步选择 |
| 未查询结果隔离 | 只修改当前 target 尚未揭示的结果，下一步选择不变 |
| 同初始状态 | Frozen-History 与主方法的第一个选择完全一致 |
| 新问题标签 | 历史全无关键事件、新版 critical → `new_in_archive=True` |
| 重新出现标签 | 早期 critical、上一版无关键事件、新版 critical → `reintroduced=True` |
| 不再受旧 rarity 压制 | 两候选上一版均无关键事件且 p 相同，不因早期某候选失败多而降分 |
| 持续旧故障 | 上一版 critical、新版 critical → 不计回退、也不在 eligible 内 |
| 未知不是安全 | predecessor 未执行/无效不能进入 eligible |
| 预算/去重 | 所有 reveal 唯一且不超过 B_eff；重复请求报错 |
| 掩码零概率 | eligible 概率全为 0 时仍不选择被掩码排除的 index |
| 候选不足 | eligible 只有 3 个时真实调用 3 次，输出实际预算 |
| 零回退 | 新旧结果一致时回退数为 0、recall 为 null，不误报安全保证 |
| 分类守恒 | 回退总数等于首次出现与重新出现之和 |
| 概率语义一致 | 生成 bank 中 `(vulnerability >= 0.75) == (collision or near_miss)` |
| 旧 builder 兼容 | 不传 profiles 时仍然构建/读取旧 registry 语义，不改变旧方法参数 |

建议的最小标签样例（1 表示 critical，0 表示 noncritical）：

| anchor | v1 | v2 | v3（上一版） | v4（新版） | 期望分类 |
|---|---:|---:|---:|---:|---|
| x0 | 0 | 0 | 0 | 1 | 首次出现 |
| x1 | 1 | 0 | 0 | 1 | 重新出现 |
| x2 | 1 | 1 | 1 | 1 | 持续旧故障，不计回退 |
| x3 | 0 | 0 | 0 | 0 | 稳定无关键事件 |
| x4 | 1 | 1 | 1 | 0 | 修复 |
| x5 | 0 | 1 | 0 | 1 | 重新出现 |

全池评分必须得到 `regression_count=3`、`new_in_archive_count=1`、`reintroduced_count=2`。

这个 toy 检查标签与信息权限，不是合成一个容易获胜的研究 benchmark。

---

## 14. 产物、命令与完成定义

### 14.1 输出目录

```text
results/diva_highway/version_regression_v1/
  manifest.frozen.json
  protocol.frozen.yaml
  provenance.json
  calibration/
    response_bank.npz
    version_quality.csv
  evaluation/
    response_bank.npz
    version_metadata.json
    episode_manifest.json
  replay/
    per_transition.csv
    query_traces.jsonl
    summary.json
    regression_count_curve.png
    outcome_breakdown.csv
  report.md
```

只要求一张回退发现曲线与一张结果表，不再默认生成多套 AUC/NDCG/latent 图。

### 14.2 必须保存的结果字段

```text
lineage_id, previous_version, target_version, history_versions
method, repeat, budget_limit, budget_used, eligible_count
regression_count, new_in_archive_count, reintroduced_count
available_regressions, regression_recall, raw_critical_score
stop_reason, protocol_hash, bank_hash, code_commit
```

`query_traces.jsonl` 每条保存选择前的概率、实际所选场景 ID、揭示后的真实事件、累计真实回退数。不要把完整 target 答案写进选择器可见状态。

`provenance.json` 至少区分：新增仿真、缓存命中、历史档案构建、离线完整标注、各方法逻辑 reveal 次数、模型拟合/推断耗时。

### 14.3 建议命令

**以下新命令是实现完成后的接口目标，当前核对提交中尚不存在这些子命令。**

```powershell
# 1. 实现后先跑纯逻辑测试；不建设新 bank。
conda run -n metadrive python -m pytest mvr/highway/tests/test_version_regression.py -q

# 2. 全部旧测试；其已有仿真调用另行记账。
conda run -n metadrive python -m pytest mvr/highway/tests -q

# 3. 冻结具体版本、协议、源码与依赖哈希。此命令不仿真。
conda run -n metadrive python -m mvr.highway.scripts.run_version_regression_mvp freeze --config mvr/highway/configs/version_regression_mvp.yaml --manifest mvr/highway/configs/version_lineages_v1.json --output results/diva_highway/version_regression_v1

# 4. 一次性构建两个固定集合；不运行方法选择或调参。
conda run -n metadrive python -m mvr.highway.scripts.run_version_regression_mvp build --run-dir results/diva_highway/version_regression_v1

# 5. 对冻结 bank 执行时间划分与四方法回放；不新增仿真。
conda run -n metadrive python -m mvr.highway.scripts.run_version_regression_mvp evaluate --run-dir results/diva_highway/version_regression_v1
```

若已有 run-dir 的内容与当前输入哈希不同，应报错，不覆盖。完全相同哈希下可以断点续建，缓存 key 必须包括版本参数和执行器信息。

### 14.4 完成定义

本轮完成意味着：

1. 新版本 reference/chronological/reveal 协议实现并通过测试；
2. A–F 旧数据未改名成版本谱系，旧结果与旧入口保留；
3. 两条参数演化谱系、两个场景集合及方法参数在读取新结果前冻结；
4. 同一 bank 的全部版本转换按原计划公开，包括没有回退的转换；
5. 输出一个结果表、一张曲线、成本账本和一个明确但不过度解释的状态；
6. 运行完成后不自动开始新一轮调参与加规模。

工程完成与论文假设成立是两件事。全测试通过但无适配增益，仍然是完整且可解释的一轮结果。

---

## 15. 给实现者的任务摘要

> 在当前 `1d6e7d2` 上新增一个独立的 PR-BRVT 版本参考模式。保持低秩 prior、解析 posterior、critical_probability 和 Highway-env 环境不变。新模式只用同一谱系的严格过去版本拟合先验；以直接前一版本已确认 noncritical 的场景构造统一候选集；按 posterior critical probability 顺序选择，得到一条真实新版本结果后再更新。
>
> 主评分使用真实 `previous noncritical AND current critical` 场景计数，并分解为档案内首次出现和重新出现。历史 rarity 不再抑制曾经修好后再次发生的问题；旧 RCS/TSF 保留在旧实验中。
>
> 为全部方法使用同一个 eligible mask。新增 Previous-Boundary 和与主方法仅差后验更新的 Frozen-History，避免把概率变换的收益误当作反馈收益。不得使用未来版本或未揭示 target 结果。
>
> 两条六版本参数序列为受控构造，不预先宣称单调提升、不注入必撞标签、不依据方法表现挑选版本。一次新 bank 最多 1920 个正式构建 episode，方法比较全缓存回放。零回退、无净收益或跨谱系不一致都如实报告，不自动换指标或扩大实验。

---

## 16. 来源与依据边界

本文件前述“当前实现/已有结果”由以下固定提交源码支持；文件级修订、候选版本参数、额外配置、费用预算和验收状态是**本次提出的设计**，不是仓库原有事实，也不是已验证方法贡献。

- [S1：当前 PR-BRVT 已提交汇总](https://github.com/SafeDL/META_LEARNING/blob/1d6e7d244bbae43ec23e728394ef0f4f010d6884/results/diva_highway/pr_brvt_mvp/summary.json)
- [S2：当前 LOSO 回放、基线与 Gate](https://github.com/SafeDL/META_LEARNING/blob/1d6e7d244bbae43ec23e728394ef0f4f010d6884/mvr/highway/experiments/run_loso_regression_mining.py)
- [S3：当前顺序挖掘及历史 rarity acquisition](https://github.com/SafeDL/META_LEARNING/blob/1d6e7d244bbae43ec23e728394ef0f4f010d6884/mvr/highway/diva/regression_mining.py)
- [S4：当前 critical probability](https://github.com/SafeDL/META_LEARNING/blob/1d6e7d244bbae43ec23e728394ef0f4f010d6884/mvr/highway/diva/regression_probability.py)
- [S5：解析 posterior](https://github.com/SafeDL/META_LEARNING/blob/1d6e7d244bbae43ec23e728394ef0f4f010d6884/mvr/highway/diva/posterior.py)
- [S6：中心化 SVD 先验](https://github.com/SafeDL/META_LEARNING/blob/1d6e7d244bbae43ec23e728394ef0f4f010d6884/mvr/highway/diva/low_rank_prior.py)
- [S7：锚点范围与双模式生成器](https://github.com/SafeDL/META_LEARNING/blob/1d6e7d244bbae43ec23e728394ef0f4f010d6884/mvr/highway/data/generate_anchor_bank.py)
- [S8：ResponseBank 与固定 registry builder](https://github.com/SafeDL/META_LEARNING/blob/1d6e7d244bbae43ec23e728394ef0f4f010d6884/mvr/highway/data/response_bank.py)
- [S9：Cut-in 执行器及实际事件/vulnerability 定义](https://github.com/SafeDL/META_LEARNING/blob/1d6e7d244bbae43ec23e728394ef0f4f010d6884/mvr/highway/envs/cutin_env.py)
- [S10：现有 SUTProfile、IDM/FVDM 控制器与参数含义](https://github.com/SafeDL/META_LEARNING/blob/1d6e7d244bbae43ec23e728394ef0f4f010d6884/mvr/highway/sut/idm_profiles.py)
- [S11：当前配置类](https://github.com/SafeDL/META_LEARNING/blob/1d6e7d244bbae43ec23e728394ef0f4f010d6884/mvr/highway/config.py)
- [S12：当前研究目标文档](https://github.com/SafeDL/META_LEARNING/blob/1d6e7d244bbae43ec23e728394ef0f4f010d6884/docs/PR_DVM_HighwayEnv_MVP_Target_20260913.md)

本协议依据最近讨论将研究收敛到版本级安全回退；不额外声称新颖性已完成查新，也不将当前群体相对结果等同于软件版本回归证据。
