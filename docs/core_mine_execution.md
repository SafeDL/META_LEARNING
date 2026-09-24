# Codex 执行目标：CoRe-Mine 创新凝练与快速实验

> **任务性质：实现小规模实验，用结果选择最终研究问题和最简有效方法。**
> 不再写一轮长篇调研，不重训 SAC/PPO/PEARL，不重构整个仓库。先复用已有响应库完成方法筛选；有明确收益后，再做少量物理确认。

- 项目：`SafeDL/META_LEARNING`
- 设计日期：2026-09-22
- 已核对的代码基点：`0e57aadf2a2dde10caefa37c6d76ab82adacfe64`
- 建议新模块：`method_chains/core_mine/`
- 建议结果目录：`results/method_chains/core_mine/`
- 本文是**待执行方案**，其中新方法的收益尚未验证。上述新目录及后文 CLI 均由本任务创建。

## 0. 给 Codex 的直接指令

读取本文件和相关现有实现，然后直接进入编码、缓存重放实验、有限迭代和结果汇总。不要把“需要进一步调研”“应先证明理论成立”作为默认终点。

本轮必须交付：**一个凝练后的研究问题、一种有实验支持的最简方法或明确的失败结论、同预算对比表、关键消融图、可重跑命令。**允许在开发集上修改方法两轮；保留每轮结果，不要求第一次设计必然成功。

实施顺序：

```text
已有结果复算与数据适配
    → 改变测试价值目标的最小实验
    → 加入组合—残差后验的最小实验
    → 同目标强基线与消融
    → 选择最简有效版本
    → 缓存验证 / 小规模物理确认
    → 凝练最终论文问题与贡献
```

优先跑完缓存阶段。默认不新增大型依赖、不调用外部付费服务、不下载大模型。保留所有已有正式方法和结果；不得覆盖它们，不自动推送远端。

## 1. 从已有尝试凝练出的研究主线

### 1.1 不再以方法拼接为创新对象

| 已有尝试或现象 | 向上一层抽象 | 本轮实验要验证的变化 |
|---|---|---|
| 功能条件化路由在模块组合基准上有效，但普通控制器基准基本持平 | 历史知识的可迁移单元是局部功能响应，而不一定是整个 SUT | 保留功能级组合，而不是退回单一全局相似度 |
| Posterior Search 的总临界事件发现已接近预算上限，不同方法在功能、区域及碰撞/近失效间分配不同 | “命中更多危险点”不等于“发现更多值得保留的脆弱性区域” | 将搜索目标改成**严重度与非冗余覆盖的边际收益** |
| 当前事件概率由历史事件标签加权得到 | 历史知识可以提供初始结构，却可能限制对目标局部新行为的表达 | 用**组合历史均值＋空间相关残差**更新目标风险 |
| 早期诊断、复杂前瞻和重型 RL 不一定转化为最终收益 | 诊断是否有价值，要由它增加了多少实际发现来决定 | 不固定要求极小 K 成功；先采用逐次反馈、解析更新、一步决策 |

**研究问题：**

> 在有限目标测试预算下，如何利用历史系统可组合的功能响应与少量目标反馈，发现严重且非冗余的脆弱性区域，而不是反复命中相似的已知危险场景？

**候选技术方案：CoRe-Mine（Compositional Residual Vulnerability Mining，组合—残差脆弱性挖掘）。**

```text
历史功能响应 → 组合先验
                     + 目标反馈 → 局部残差后验
                                        ↓
                       严重度感知的边际覆盖收益
                                        ↓
                            下一个真实目标测试
```

本轮只争取两项有辨识度的增量：

1. **方法增量：**把封闭的历史响应选择改成可组合、可局部修正的目标响应模型。
2. **问题与决策增量：**把重复失效点的数量最大化改成已验证失效档案的边际测试价值最大化。

“历史全部安全而目标失效”保留为重要分析维度，**不把它设成整条研究路线必须满足的唯一任务条件**。已有版本回归、共同盲区方向没有足够正例时，不重新陷入等待正例的循环。

### 1.2 先区分三种可能的有效结论

- **只有决策目标有效：**保留简单后验＋边际收益搜索，研究主线是预算约束的非冗余脆弱性发现。
- **残差模型也有效：**保留完整 CoRe-Mine，主张组合迁移与局部差异学习共同改善测试价值。
- **只在历史未覆盖区域有效：**将最终应用定位收窄到历史知识不足时的差异化失效发现，同时保留完整基准结果。

根据开发实验选择一种，冻结后验证。不得在验证结果出来后切换主指标或只保留有利目标。

## 2. 复用资产与最小改动边界

优先读取并复用：

```text
method_chains/function_posterior_search/search.py
method_chains/function_posterior_search/benchmark.py
method_chains/function_posterior_search/confirmation.py
method_chains/function_conditioned_routing/routing.py
method_chains/function_conditioned_routing/experiment.py
method_chains/detour_fusion/fusion.py
highway_env_benchmark/data/response_bank.py
highway_env_benchmark/envs/cutin_env.py
sut_algorithms/highway_env/idm_profiles.py
replications/README.md
results/method_chains/function_posterior_search/confirmation/
```

现有基点中，`ResponseBank` 已保存场景坐标、模式、可选 timing/intensity、连续响应、碰撞、近失效、TTC 和距离；`PosteriorSearch` 已有逐次 `observe/propose` 接口。[S1–S2]

不要重新构建全量响应库。读取当前 HEAD；若与基点不同，记录 SHA 和涉及本任务的接口差异，但不要强制回退用户工作区。

### 2.1 一次性口径处理，不开展大型审计

- 缓存实验沿用缓存记录的事件定义，对所有方法完全一致。
- 首轮主实验使用五类双车功能场景；三车 `passing_cutin` 暂作独立附加结果。这样不必在第一轮为背景车辆碰撞归属重建整个数据集。
- 物理确认必须分别记录 `ego_collision`、`background_collision`、`near_miss`，收益只计目标车辆相关事件。
- 为每个功能设置有效输入维度掩码：未改变执行行为的 timing/intensity 不参与距离、GP 或覆盖度。先按 `_mode_schedule()` 和执行代码确认，不能仅因 NPZ 有四列就使用全部四维。
- 缺少字段时给出明确的可用子集；不凭空恢复轨迹或碰撞类型。

## 3. 数学工具一：严重度感知的边际覆盖收益

### 3.1 事件反馈和真实测试收益

令碰撞指标为 `C(x)`，临界事件指标为 `E(x)=collision OR near_miss`，确保 `C≤E`。定义：

\[
r(x)=\tfrac12E(x)+\tfrac12C(x)\in\{0,0.5,1\}.
\]

`r=1` 为碰撞，`r=0.5` 为近失效，其他为零。只给实际执行并揭示的结果记分。重复执行相同场景不得重复增加档案覆盖。

### 3.2 档案价值，而不是单点最高风险

对同一功能内的有效场景参数按固定物理范围归一化，定义非负覆盖相似度：

\[
\kappa(x,x')=\mathbf1[m(x)=m(x')]
\exp\left[-\frac{\|z(x)-z(x')\|^2}{2\ell_c^2}\right].
\]

默认 `ell_c=0.20`；超出核心范围的坐标保留真实归一化值，不截断到边界。不同功能之间不产生覆盖。

给候选池每个点权重 `w_i=1/(M·N_m)`，使每种功能的总权重相同。对已执行集合 `Q_t`：

\[
c_i(t)=\max_{a\in Q_t}r(a)\kappa(x_i,a),\qquad
F(Q_t)=\sum_iw_i c_i(t).
\]

同一区域重复碰撞的收益递减；新区域的碰撞有较高收益；同一区域从近失效升级为碰撞仍可增加价值。`F` 衡量已验证危险案例的参数空间覆盖，不是未测试点已被证实失效。

采用统一总目标：

\[
J_B(Q)=F(Q)+\frac{\lambda}{B}\sum_{x\in Q}r(x).
\]

默认 `lambda=0.10`，保留事件严重度总量，避免只追求远距离覆盖。开发阶段只允许尝试 `{0,0.10,0.30}`。为复用同一条查询轨迹，公式中的归一化预算固定为 `B_ref=50`；B=10/20/30/50 均评价该固定规则的相应前缀，不在中途更换分母。

### 3.3 下一次测试的可计算边际收益

模型输出 `p_C(x)` 和 `p_E(x)`，满足 `0≤p_C≤p_E≤1`。定义：

\[
\Delta_v(x)=\sum_iw_i[\max\{c_i(t),v\kappa(x_i,x)\}-c_i(t)].
\]

选择分数：

\[
a_t(x)=p_C(x)\Delta_1(x)
+[p_E(x)-p_C(x)]\Delta_{0.5}(x)
+\frac{\lambda}{B}\frac{p_E(x)+p_C(x)}2.
\]

这是一条直接可实现的、关于本次测试真实结果的期望边际收益规则，不需要训练策略网络或多步树搜索。

初始化沿用现有前 10 次功能覆盖机制，但覆盖约束内的排序使用各方法自己的分数。所有测试计入总预算。相同功能覆盖规则应用于相应匹配对照，单独保留 legacy 原始方法结果。

**理论表述仅保留一句：**固定真实结果时，上述最大相似度型集合效用具有收益递减结构；存在相关后验更新时，不自动援引自适应贪心的近似最优保证。[M1]

## 4. 数学工具二：组合—残差贝叶斯后验

### 4.1 统一连续响应，不让新方法独占额外反馈

为便于快速实现两个有序事件概率，使用同一套公开转换：

\[
b(x)=\begin{cases}\exp[-\max(\mathrm{TTC}(x),0)/3],&\mathrm{TTC}<\infty,\\0,&\text{otherwise},\end{cases}
\]

\[
y(x)=0.25b(x)+0.5E(x)+0.5C(x).
\]

安全、近失效、碰撞分别落在 `[0,0.25]`、`[0.5,0.75]`、`[1,1.25]`。令阈值 `tau_E=0.375`、`tau_C=0.875`。缺失或 NaN TTC 统一令 `b=0` 并记录。

这是本轮提出的建模响应，不替换正式事件标签。所有新增 GP 基线和匹配的 Posterior Search 都获得相同 `C/E/TTC/y`；额外保留原始响应版本，排除收益仅来自换反馈编码。

### 4.2 历史组合分支与局部残差

对每个功能独立维护去重后的历史响应假设 `h=1,…,J_m`：

\[
f_*(x)\mid H_m=h=y_h(x)+\delta_h(x),\qquad
\delta_h\sim\mathcal{GP}(0,k_m).
\]

首版只在已有有限候选池上搜索，`y_h(x)` 直接查历史表，无需训练连续历史模型。残差核用有效物理维度上的 Matérn-5/2；默认长度尺度 `0.30`、幅度 `0.25`、观测标准差 `0.05`。这些是起始超参数，不是已证实最优值。[M2]

仅用已揭示目标点 `I_t` 更新。对于一个历史分支：

\[
\mu_h(x)=y_h(x)+K_{xI}(K_{II}+\sigma^2I)^{-1}[y_I-y_h(I)],
\]

\[
v_h(x)=k(x,x)-K_{xI}(K_{II}+\sigma^2I)^{-1}K_{Ix}.
\]

用 Cholesky 和 jitter，不显式求逆。同功能、同核超参数的分支共享矩阵分解。测试预算最多 50，无需稀疏 GP 或 GPU。

### 4.3 用预测证据选择历史解释

每次 `observe` 前保留预测分布，再更新：

\[
\log q_{h,t}\leftarrow\log q_{h,t-1}
+\log\mathcal N(y_t;\mu_{h,t-1}(x_t),v_{h,t-1}(x_t)+\sigma^2).
\]

通过 `logsumexp` 归一化。不要用拟合完新数据后的残差代替上述事前预测证据，也不要重复累计同一个观测的似然。

增加一个轻量 `null` 分支：常数先验均值固定为 `0.50`，使用同类物理 GP、不查历史响应曲线；初始总权重 `0.10`，其他去重分支平分 `0.90`。这是“少信历史”的候选解释，而不是分布外检测保证。`null` 无收益时可以通过消融删除。

### 4.4 事件概率与完整方法

令 `s_h²=v_h+sigma²`，计算：

\[
p_E(x)=\sum_hq_h\Phi\left(\frac{\mu_h(x)-\tau_E}{s_h(x)}\right),\quad
p_C(x)=\sum_hq_h\Phi\left(\frac{\mu_h(x)-\tau_C}{s_h(x)}\right).
\]

数值上保证方差正、概率有界且 `p_C≤p_E`。这些是模型概率，校准性能由实验衡量。

**完整 CoRe-Mine = 功能历史组合＋局部残差＋预测证据更新＋边际覆盖选择。**

关键能力变化：所有历史在某点都安全时，目标在相邻点的新反馈仍可通过残差改变该点风险，不再只能给历史标签重新加权。

## 5. 三组快速实验，先回答最小问题

### E0：复算与机会扫描——只做一次

读取现有确认响应库及 summary，重新计算：事件数、碰撞数、严重度总和、功能分布、下文独立覆盖指标，以及历史全安全候选中的目标失效数。

输出 `inventory.json`、`baseline_recheck.csv`、`opportunity.md`。`opportunity.md` 不超过一页，回答：总命中是否饱和？覆盖是否还有空间？局部新失效是否存在？

**不设“找不到历史独有失效就停止”的门槛。**它们稀少时照常进入覆盖发现实验。此阶段不训练模型、不补齐全部仿真。E0 最多投入约 45 分钟；输出已完成检查后进入核心实验，不因非关键字段缺失扩展为全面审计。

### E1：只改变决策目标，先检验问题是否有价值

使用完全相同的 Posterior Search 概率，比较：

| 方法 | 唯一变化 |
|---|---|
| FPS-Risk | 原始临界事件概率贪心 |
| FPS-Severity | 按 `0.5·p_E+0.5·p_C` 贪心 |
| FPS-Balanced | 严重度贪心＋按功能均衡查询的简单对照 |
| FPS-Marginal | 本文边际覆盖收益，不加残差 GP |

事件概率由已去重源假设上的 `E/C` 权重分别求得。`FPS-Balanced` 每步优先当前查询次数最少的功能，再按严重度排序。

问题：**同样的预测器，更合适的测试目标是否已经能得到更好的失效档案？**如果是，保留这个简单版本作为后续不可缺少的强基线。

### E2：相同测试目标下，检验方法增量

核心采用 `3×2` 因子实验：

| 后验模型 | 严重度贪心 | 相同边际覆盖选择 |
|---|---|---|
| 闭合的功能历史假设 FPS | FPS-Severity | FPS-Marginal |
| 固定 source 平均响应＋残差 GP | MeanResidual-Risk | MeanResidual-Marginal |
| 组合历史＋残差＋null | CoRe-Risk | **CoRe-Marginal** |

另加 `TargetOnlyGP-Marginal`：同样的物理核和可观察目标反馈，先验均值固定为 `0.50`，使用公共固定超参数，不用 source 数据拟合模型或选择超参数。

从已有实现保留 AdaTE、Function-Conditioned Mining、DETOUR 原始结果作为参考；需要重新运行新候选子集时，复用现有接口，不重写整套复现。FST、ScenarioFuzz 若已有兼容的有限池入口，可加为附加对照，不把重新适配它们设为本轮前置条件。

**关键判据：不能只证明 CoRe 优于原始 risk-only 方法；必须比较 FPS-Marginal、MeanResidual-Marginal 和 TargetOnlyGP-Marginal。**

### E3：只对胜出版本做消融与验证

最多三项新增消融：

1. `NoComposition`：全局历史权重，其他不变。
2. `NoNull`：删除 null 分支，其他不变。
3. `NoResidual`：使用 E2 中 FPS-Marginal，不另写重复实现。

结果不支持的模块直接删去。复杂度更低且效果相当的版本优先。

## 6. 数据划分、预算与计算上限

### 6.1 缓存阶段

- 优先使用现有 Function Posterior Search 五组确认 bank；以文件名排序，前两组用于本轮开发，后三组用于本轮缓存验证。原已有实验可作为设计信息，但不得把这些缓存重新称为完全未知的外部测试。
- 每组保留全部 18 个目标，分别汇报 `exact/interpolated/unseen` 与异质性分组；不按方法表现删除目标。
- 首轮主池去掉 `passing_cutin`，保留全部五类双车场景。分组元信息不得作为选择器输入；功能标签及物理参数可用。
- 各方法生成一条长度 50 的轨迹，统一报告 `B={10,20,30,50}`；**主检查点固定 B=20**。
- 初始化最多 10 次，全部计费；逐次更新允许贯穿 B，不再强行规定 `K=1/2/4` 必须成功。
- 确定性方法固定 tie-break；Random 重复 10 次，先聚合后比较。
- 默认零新增物理仿真、零神经网络训练。CPU 线程数至多 8，不阻塞用户机器。

### 6.2 调参不铺大网格

第一轮只跑默认值。第二轮最多测试 12 个预先列出的配置，总共不超过两轮方法修改。

只允许修改：残差长度尺度 `{0.15,0.30,0.60}`、幅度 `{0.15,0.25,0.50}`、`lambda={0,0.10,0.30}` 中少量有理由的组合；不是跑笛卡尔积。核超参数优先由 source 留一任务选择，方法结构与最终版本由开发 bank 选择。

覆盖指标的分区、主预算、事件标签和成功标准不随开发结果修改。验证只跑冻结配置。实验超时先减少配置和附加基线，不删掉最强的匹配对照。

## 7. 评价指标：不能只优化自己定义的分数

### 7.1 主指标：独立分区的严重度覆盖 CVS@20

使用与选择器 RBF 覆盖核不同的固定分区：每个功能按归一化 `gap × relative_speed` 做 `4×4` 网格。分区边界由 source/candidate 的物理范围在运行目标前固定；边界外设 overflow 格，不裁掉场景。记格子为 `c`：

\[
\mathrm{CVS}@B=\sum_c\max_{x\in Q_B\cap c}r(x),
\]

空格贡献零。首次发现一个碰撞格贡献 1，首次近失效格贡献 0.5，同格重复事件不累加，近失效升级为碰撞再增加 0.5。

这叫**危险场景区域覆盖**，不称为独立软件 bug 数。它不使用完整目标响应来构造格子，也不使用模型预测来计分。

### 7.2 配套指标

固定报告：

- `CollisionCount@B`、`CriticalCount@B`、`SeveritySum@B`。
- `CVS@B`、碰撞格数量、`F(Q_B)`、功能覆盖分布。
- `NewHistoricalFailureCount@B`：所有可比较历史系统安全、目标真实失效的已查询点数；无正例照常报告 0。
- 在完整缓存上可报告 Recall、同预算理想事件数及差距；稀疏物理确认没有完整标签时不报告全池 Recall。
- 决策耗时、模型更新时间、逻辑目标查询数、新增物理 episode 数。

对最终版本补充 `3×3/5×5` 网格敏感性，以及加入实际有效 timing/intensity 的预固定粗分区。若收益只出现在 `4×4`，如实写明；不挑表现最好的划分作为主结果。

### 7.3 预设的工程选型目标

开发阶段，以下作为继续投入的目标而不是统计定理：相对最强匹配基线，`CVS@20` 平均提升至少 10% 且绝对提升至少 1 个碰撞格等价值，同时 `SeveritySum@20` 相对下降不超过 10%。

小于该幅度但方向稳定时也允许进入缓存验证；最终标记为“小幅收益”，不将有限样本显著性作为所有开发步骤的硬阻断。若基线均值为零，只报告绝对差。

统计输出配对差、各场景 seed 的差异、按目标聚合差异及 bootstrap 区间。重采样应保留同一 seed/目标下的方法配对，并说明共享控制器带来的相关性；不要把所有 episode 当独立样本。

## 8. 结果驱动的有限迭代规则

| 观察到的结果 | 下一步修改 | 凝练后的主线 |
|---|---|---|
| FPS-Marginal 已明显提升，CoRe 没有额外收益 | 删掉无效残差/拒绝模块，完成简单方案验证 | 后验引导的预算约束非冗余失效发现 |
| CoRe 在 interpolated/unseen 上提升，在 exact 上持平 | 保留组合—残差方法；做 NoResidual 和 NoComposition | 功能组合迁移与局部差异联合建模 |
| CoRe 只改进历史未覆盖失效 | 保留完整总表，另做这一预定义子任务的验证 | 历史盲区中的目标特异失效发现 |
| 覆盖增加但严重度总和损失过大 | 在开发集增加 lambda；与 FPS-Severity 比较 | 覆盖—严重度的预算权衡，不声称全面提升 |
| null 分支占用过多预算或无实际贡献 | 删除 null；保留可扩展残差 | 不把拒绝迁移硬留作创新模块 |
| MeanResidual 或 TargetOnlyGP 一样好 | 采用更简单模型，贡献集中在目标/选择规则 | 不为保留“元学习”名称增加无效结构 |
| 两轮后所有匹配对照都无稳定差异 | 停止新增仿真，交付完整结果与一种明确下一步假设 | 完成本轮研究筛选，而不是制造优势 |

不得仅因 source 历史独有失效不足，就人为改写事件标签。允许设计固定参数扰动的可执行控制器来研究机制；所有事件仍由仿真产生，并保留所有预先生成的目标，而不是只挑容易证明方法的目标。

## 9. 小规模物理确认：只验证胜出版本

仅当缓存验证出现值得继续的收益时执行。冻结方法及两种最强对照，然后：

1. 选择或创建 4 个未用于本轮开发的可执行目标控制器。优先使用仓库尚未进入本轮 source/target 的现有配置；不足时，在已有控制器合法字段范围内做固定、结果无关的参数组合，并保存 manifest。
2. 使用两组已有场景 bank 的具体场景及相同 simulator seed，复用 source 记录，不为所有新目标执行全池。
3. 每个目标、每组 bank、每种方法最多 50 次逻辑目标查询。三种方法的相同执行可共享缓存，但各方法预算仍独立计费。
4. 实际最多 `4×2×3×50=1,200` 个新目标 episode，再允许至多 24 次代表性回放；硬上限 1,250。只有相关代码、控制器、场景、seed 均相同才可复用物理结果。
5. 三种方法统一使用目标车辆事件判据。背景车碰撞不记作目标失效；仿真异常占用已发起的测试预算并单独记录，不免费重试到成功。

不在看到确认结果后重新生成更有利的控制器。未完成的确认也输出现有进度、预算和原因，不留成无结果的后台任务。

## 10. 建议代码结构与接口

以下为需要新增的结构，不是对当前仓库现状的声明：

```text
method_chains/core_mine/
  __init__.py
  config.py               # 预算、开发/验证划分、少量超参数
  data.py                 # ResponseBank 适配、功能掩码、响应编码
  oracle.py               # 单次 reveal、预算、缓存键
  posterior.py            # 组合—残差 GP、预测证据权重
  acquisition.py          # risk/severity/balanced/marginal
  metrics.py              # CVS、F、事件指标、配对统计
  experiment.py           # audit/develop/validate/confirm/all
  report.py               # 表格、图和研究结论
  tests/
```

统一接口：

```python
predict() -> {"p_event": array, "p_collision": array, "mean": array, "variance": array}
propose(allowed_indices=None) -> int
observe(index: int, outcome: RevealedOutcome) -> None
oracle.reveal(index: int) -> RevealedOutcome
```

选择器不能持有完整目标向量。`oracle` 可以在缓存实验中持有标签，但只能揭示请求的点；评价器可在轨迹结束后读取完整标签计算 Recall。完整目标的覆盖可达性与历史未覆盖失效清单不得传给选择器。

### 必须通过的最少单元测试

- 预算不可超支；同场景不可重复免费计分；碰撞、近失效编码和阈值对应正确。
- `0≤p_collision≤p_event≤1`，GP 数值稳定，源假设去重正确。
- 后验权重使用更新前预测，单个观测只累计一次。
- 覆盖值对真实新增失效单调，重复相同事件不增益，同格近失效升级碰撞有增益。
- 当前动作不因未查询目标标签被替换而变化。
- Matérn 和覆盖核不使用无效参数维度；新旧场景索引可追溯。

CLI 需要由本任务实现，目标命令如下；使用仓库已有环境，不新建大型环境：

```bash
python -m method_chains.core_mine.experiment --stage audit
python -m method_chains.core_mine.experiment --stage develop --max-trials 12
python -m method_chains.core_mine.experiment --stage validate
python -m method_chains.core_mine.experiment --stage confirm --max-new-episodes 1250
python -m pytest method_chains/core_mine/tests -q
```

默认 `--stage all` 运行 audit → develop → validate → report；只有缓存结果符合继续投入规则且显式设置 `--allow-new-simulation` 才进入 confirm，防止无意启动大规模仿真。

## 11. 输出工件与验收

```text
results/method_chains/core_mine/
  manifest.json                # SHA、数据哈希、环境、配置、标签语义
  inventory.json
  baseline_recheck.csv
  trials.csv                   # 所有开发尝试，不删除失败配置
  frozen_config.json
  develop/records.csv
  validate/records.csv
  confirm/records.csv          # 未执行时明确标记，不伪造
  trajectories.jsonl           # 每步预测、动作、已揭示响应、计费
  summary.csv
  ablations.csv
  cost_summary.json
  figures/
  report.md
  research_decision.md
```

首轮只要求 4 张有用的图：CVS 随预算变化、严重度总和随预算变化、分组收益、一个目标的查询分布对照。没有轨迹字段时不生成“故障根因图”；没有真实新增仿真时不画虚构回放。

`research_decision.md` 控制在两页，必须直接回答：

1. 最终研究问题一句话是什么？它相对旧的单点风险搜索增加了什么测试价值？
2. 最简有效方法保留了哪些模块？哪项消融支持每一个模块？
3. 对最强匹配基线，B=20/50 的 CVS、碰撞数、严重度总和及成本分别如何？
4. 收益来自新目标、组合迁移、残差修正，还是仅来自简单覆盖控制？
5. 结论属于“完整方法有效”“简化方法有效”“局部条件有效”还是“本轮未发现可靠增益”？

**完成标准不是保证新方法赢，而是完成上述实验选择，并交付可以直接写进下一轮研究讨论的结果。未经实验支持，不把本文的候选创新描述成已证实贡献。**

## 12. 依据与引用

### 项目依据

- [S1] `method_chains/function_posterior_search/search.py`，基点 `0e57aad`：功能内历史假设、目标反馈更新、事件概率贪心。
- [S2] `highway_env_benchmark/data/response_bank.py`，同一基点：缓存字段和逐次揭示所需数据契约。
- [S3] `results/method_chains/function_posterior_search/confirmation/report.md`，同一基点：五组 bank、18 个目标，以及功能/区域收益差异。
- [S4] `method_chains/function_conditioned_routing/README.md`，同一基点：功能组合迁移与普通控制器对齐实验。
- [S5] 用户附件《motivation+20260908(6).pdf》，第 11–12 页：共享响应结构与系统特异性残差。
- [S6] 用户附件《技术方案+20260910(7).pdf》，第 5、7、10 页：局部残差、失效档案去冗余、固定预算实验。本文把这些设计推进为组合—残差模型和可计算边际收益，而不照搬早期强制冻结后验的流程。
- [S7] 用户附件《可行性调研分析+20260912(5).pdf》，第 11–15 页：预测改善须转化为实际发现收益。本轮不沿用其中较长的多周实施时间表。

### 数学工具出处

- [M1] Golovin, D.; Krause, A. *Adaptive Submodularity: Theory and Applications in Active Learning and Stochastic Optimization*. JAIR, 42, 2011. arXiv:1003.3967。用于理解收益递减和自适应选择；本方案没有证明满足其全部近似保证条件。
- [M2] Tighineanu, P. et al. *Transfer Learning with Gaussian Processes for Bayesian Optimization*. AISTATS / PMLR 151, 2022. arXiv:2111.11223。用于 GP 迁移和解析后验的实现依据；组合—残差与档案边际收益的具体实验组合是本文的待验证设计。

---

**执行摘要：先用已有数据检验“相同后验＋不同测试价值”能否改善失效档案，再检验“组合—残差模型”能否在相同价值目标下带来额外收益。两轮内收敛到最简有效版本，再决定是否花最多 1,250 次新仿真确认。**
