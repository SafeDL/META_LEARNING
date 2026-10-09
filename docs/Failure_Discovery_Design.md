# 有限预算失效发现：方法设计与实验计划

更新日期：2026-10-09。本文是唯一现行设计文档，合并研究动机、当前方法、固定实验协议和下一步计划。实现入口见 [研究目录](../research/representative_sut_testing/README.md)，实验过程与旧方案见 [阶段报告](../research/representative_sut_testing/results/stage_report.md)，完整基线见 [结果对比](../research/representative_sut_testing/results/baseline_comparison.md)。

**当前结论：尚未建立显著优势。** 最近完成的仅符号／风险顺序两种新机制均未提高原数值版本的终点召回，顺序版本相对仅符号控制两项均下降。不晋升、不补种子，停止这组观测变体扩展。本文区分已实现机制、已有证据与后续研究问题。

## 1. 设计目标

在固定场景候选池和 **200 次唯一目标查询** 下，利用历史系统记录与目标反馈，同时提高早期碰撞发现效率和最终召回。

核心问题是：**少量反馈能否支持纠正当前选择？如果不能辨别，继续获取信息是否值得，怎样回退才能控制损失？** 目标不是识别驾驶控制器的真实参数，而是识别会改变后续危险场景选择的响应差异。风险预测更准确、历史来源判断更准确或后验更集中，都不自动意味着发现更多碰撞。

附件调研中保留的设计原则为：

- 保留历史系统在相同场景下的响应差异，避免先平均掉有用的危险排序信息。
- 目标反馈既修正预测，也检验偏离参照选择是否有价值。
- 信息获取要计入查询预算、失败探测损失和剩余发现收益。
- 推断依据是给定场景后的系统响应，不把历史采样分布或真实控制参数当作系统身份。

当前维护一个候选和两个必要对照。既有方法、基线及结果冻结保存；不新增驾驶策略训练，不为取得优势修改被测系统或场景。每个有限阶段先总结，再决定下一项修改。

## 2. 固定实验协议

### 2.1 被测系统

| SUT | 固定入口 | 实现与解释范围 |
|---|---|---|
| IDM | `idm_ref` | 解析纵向跟驰，配置区别于历史来源 |
| FVDM | `fvdm_target` | 最大制动 8 m/s²、停车间距 8 m、延迟 0.15 s、期望速度 23 m/s |
| IDM+MOBIL | `mobil_ref_v2` | 纵向跟驰与自主换道，纵向部分与 IDM 关联 |
| VI-TTC | `vi_ttc_ref_audit_v4` | 项目中的有限 TTC 状态规划 |
| MCTS-CV | `mcts_cv_ref_audit_v4` | 恒速预测的根动作 rollout，不是完整树搜索复现 |
| 冻结 PPO | `ppo_ref_v2` | 已有 checkpoint，观测与动作接口已审计 |

六个条目有关联，不代表六个独立算法家族。比较的是同一 SUT 上不同测试方法的发现效率，不以跨 SUT 碰撞数评价驾驶算法。完整参数见 [实测协议](../research/representative_sut_testing/results/protocol.json)、[SUT 注册表](../sut_algorithms/highway_env/registry.py) 和 [实验配置](../research/representative_sut_testing/config.py)。

### 2.2 场景与执行

| 场景 | 四维参数范围 |
|---|---|
| Cut-in | 初始净间距 8–60 m；前车速度 15–25 m/s；换道时间尺度 1.5–3 s；事件开始 0.5–2 s |
| 前车急刹 | 初始净间距 8–70 m；前车速度 15–25 m/s；减速度 0.5–7 m/s²；事件开始 0.5–2 s |

每池每类 1024 个场景，共 2048 个；坐标第五列仅表示场景类型。执行 12 s，自车初始速度 25 m/s，两车道，场景期望速度 23 m/s。控制器内部参数独立记录，例如原生 IDM 为 27 m/s。

模拟器为 highway-env，运行环境为 `conda activate metadrive`。物理频率 20 Hz，原生和配置式控制器 20 Hz，外部策略 5 Hz 决策。保留自主换道并记录实际交互。

两个共享 Sobol 池的种子对为 `(860031,860032)`、`(860033,860034)`。全部 SUT 共享坐标、初始条件和事件，固定模拟器及控制器随机状态。主比较选择种子为 `(11,23,37,53,71)`。

### 2.3 预算、历史信息与反馈权限

每条曲线的两类场景共用 **200 次唯一查询**，包含 10 次共同初始化及所有诊断查询；报告节点为 10/30/50/100/150/200。每次反馈后重新选择，不额外赠送诊断预算。重复种子产生相同选择序列时，不计为独立证据。

当前候选使用原六个实测 IDM/FVDM 来源：常规 IDM、常规 FVDM、延迟 IDM、延迟 FVDM、制动受限 IDM、预测响应 IDM。历史风险先验、经验响应差异和风险到事件映射均使用原冻结训练划分。历史留出开发中，被留出的目标必须从其响应先验与训练监督中排除。

候选接收已查询的连续风险 R 和同次自车碰撞 C。风险由 TTC、DRAC、距离构成，可能携带接触信息，不是独立于碰撞的廉价 oracle；定义见 [风险计算](../methods/history_guided_testing/risk.py) 和 [执行测量](../methods/history_guided_testing/measurements.py)。冻结参照继续只用原有风险反馈。

独立选择进程只接收场景坐标、允许的历史信息和请求过的反馈。目标身份、真实控制参数及全池标签由评估器保管。历史完整标签可离线生成监督；目标未查询标签不能进入选择、训练或前瞻。

已有 48 个公开 IDM/FVDM 配置曾用于有限开发：两族各 `00..11` 共 24 个训练配置，各 `12..17` 共 12 个验证配置，`18..23` 排除在该轮之外。这些属于两个算法族的参数配置，不能称为 48 个算法。**当前候选不扩充为 30 来源先验，也不使用这批配置的收益学习模型。** 旧划分及结果保存在 [配置验证协议 (archive)](../research/representative_sut_testing/results/retired_experiments.zip)。

### 2.4 文献补充配置

FST 的四历史 IDM、三目标 IDM 加一 FVDM，以及切入时刻间距与相对速度的二维场景，保留为独立补充配置。原文 5/10/20 次预算服务于整体安全估计，不替代当前 200 次失效发现预算。旧四维场景删列不能称为原文复现；实施时单独列明参数出处和仿真适配差异。[FST 原文](https://www.jingxuanyang.com/file_upload/2025-FST-TITS.pdf)。

ATSLG 的 FVDM 代理、ACC+AEB 目标暂不并行展开。控制律和参数须依据原文及引用实现；最小距离小于 1 m 的事故事件与模拟器实际碰撞分列。[ATSLG 原文](https://arxiv.org/pdf/2003.03712)。


## 3. 当前实现：反馈支持的选择纠正与参照回退

流程为：**历史数值／仅事件符号解释 → 真实反馈更新可信程度 → 联合发现损失支持选择 → 独立参照回放控制损失**。清理后的入口为 [evaluate_feedback_correction.py](../research/representative_sut_testing/evaluate_feedback_correction.py)，新执行时写入 `results/feedback_correction/`。已有仅符号结果保存在 `results/safe_risk_order_coupling/event_sign/`；它属于较强开发对照，尚未建立显著优势。Student-t 路线单独保留在 [evaluate_student_risk.py](../research/representative_sut_testing/evaluate_student_risk.py)。

### 3.1 响应模型与历史先验

[censored_margin_response.py](../research/representative_sut_testing/censored_margin_response.py) 维护潜在有符号余量 Z(x)：

- 无碰撞时，使用正值 `y=1−R` 作为危险边界远近的代理观测，采用 Gaussian 数值似然。
- 碰撞时，使用 `Z+noise≤0` 的事件约束，不为未知负余量补一个数值；该次 R 不作为模型的精确余量观测。

Z 是构造的潜在代理量，并非实际车距。实测碰撞风险最低约 0.510、无碰撞风险最高约 0.982，范围重叠；原始 Risk 不存在已验证的统一碰撞阈值。[观测核查](../research/representative_sut_testing/results/reference_feedback_ablation/feedback_semantics_audit.json)。

历史先验使用同类场景的八近邻，带宽固定 0.15，估计六来源事件概率 p_j 与无碰撞代理均值。平均事件概率 p 截在 `[0.01,0.99]`，正报告均值记为 m。令：

`a=−Phi^(-1)(p)`；

`sigma=m/(a+phi(a)/Phi(a))`；

`mu=a×sigma`。

这使初始越界概率与正报告条件均值同时匹配历史。Phi、phi 分别为标准正态分布函数与密度。历史中心化模式 `U_j=(−Phi^(-1)(p_j)−来源均值)/sqrt(5)` 保留来源分歧；核由冻结风险相关性与 `UUᵀ` 相加后重新归一化，再按报告尺度换算。噪声／总方差比例沿用冻结核。

保留的第二个解释是 [event_report_response.py](../research/representative_sut_testing/event_report_response.py) 中的仅符号模型：初始均值为零、报告方差归一为一，相关性及噪声比例仍来自相同历史结构。它只按实际碰撞／未碰撞施加正负约束，不使用目标风险数值确定余量，也不施加安全风险排序。中性数值模型保留为必要对照，已失败的风险顺序约束退出活动实现。

来源的等权汇总只是先验构造，差异继续保留在协方差中；目标反馈会更新两个解释的选择权重。当前尚未实现六个来源分别在线更新可信度，不能将其描述为已完成的六来源辨识。

### 3.2 在线证据与 EP 推断

[adaptive_margin_prior.py](../research/representative_sut_testing/adaptive_margin_prior.py) 从两个组件各 `1/2` 的初始权重出发。每条真实反馈使用查询前预测更新：

`w_(t+1,h) ∝ w_(t,h) × p_h(observation_(t+1)|D_t)`。

默认使用查询前二元事件概率更新可信度，不把连续安全代理密度直接计入权重。两组件接收同一次已付费反馈；历史数值组件使用 R/C，仅符号组件记录 R 但只用 C 更新。权重持续更新，已删除十次初始化后锁定解释和分场景证据共享分支。权重表示预测支持，不是控制器真实身份概率。混合评分保留用于必要数值对照，具体事件评分与对数损失界见第 5.3 节。

[ep_margin_response.py](../research/representative_sut_testing/ep_margin_response.py) 将先前单次矩更新的 ADF 换为联合 EP：每次反馈后保留精确 Gaussian 安全数值站点，反复更新全部碰撞区间站点。采用 GPU float64，阻尼 0.5、相对站点变化阈值 `1e−7`、最多 200 次迭代，未收敛直接报错。只替换推断，先验、似然、选择和保护规则不变。

动机是 ADF 对记录顺序敏感：同样的 200 条已付费记录重排后，最大事件概率差约 0.158，Top-200 集合最多改变 85 个成员。[顺序诊断 (archive)](../research/representative_sut_testing/results/retired_experiments.zip)。EP 是已有近似推断，当前检查不提供一般收敛或真实概率校准保证。[GPML 第 3 章](https://gaussianprocess.org/gpml/chapters/RW3.pdf)。

### 3.3 概率读出与选择支持

对组件 h，Gaussian 工作后验读出为：

`p_h(x)=Phi(−mu_h(x)/sqrt(K_h(x,x)+noise_h(x)))`；

`p_C(x)=sum_h w_h p_h(x)`。

这由均值与报告方差计算越界事件概率；联合选择比较还使用两点协方差，不能仅由两个标准差代替。当前使用正态分布函数，不在风险均值外直接套 sigmoid。

每步先取得独立参照下一场景 j，只在与 j 相同类型的未查询场景中挑选候选 i。仅当 `p_C(i)>p_C(j)+1e−12` 时提出改序；平局由标准化余量与固定种子打破。

[decision_support.py](../research/representative_sut_testing/decision_support.py) 计算局部损失与收益：

`ell=P(C_i=0,C_j=1|D_t)`，即候选未发现、参照本可发现的概率；

`g=P(C_i=1,C_j=0|D_t)`，即候选发现、参照未发现的概率。

各组件的联合 Gaussian 报告协方差包含独立观测噪声，采用确定性二元 Normal 积分，再按当前权重平均。恒等式 `g−ell=p_C(i)−p_C(j)` 用于核查。

共同十次初始化后，每两个查询原定第一个为参照步骤。若实际额度允许，普通候选步骤可进行受保护探测；原定参照步骤只有在候选概率更高且 `ell≤1/sqrt(B)` 时才改用候选，否则观测参照。B=200，阈值实验前锁定，不按目标成绩搜索。[选择执行](../research/representative_sut_testing/discovery_supported_coupling.py)。

两个都很可能碰撞的场景可能近似等效，无须严格辨明谁更危险。已付费前缀诊断中，严格倾向排序只支持跳过 3 个参照时段，二元损失条件支持 487 个。[发现等效诊断 (archive)](../research/representative_sut_testing/results/retired_experiments.zip)。ell 仍是模型内单步量，不是整条延续收益或未知目标上的频率保证。

### 3.4 独立参照与实际损失保护

[frozen_coupling.py](../research/representative_sut_testing/frozen_coupling.py) 复用原 `TestingSession(mode="global_feedback")`。参照只接收其查询顺序中已实际测得的风险；候选额外反馈不提前改变参照。参照后来遇到已查询场景时，读取已有反馈推进回放，不重复计费。

记 A_t 为实际 t 次查询集合，P(k_t) 为已回放参照前缀，`E_t=A_t\P(k_t)`，L_t 为 E_t 中尚未被参照接纳的非碰撞数。因为 `P(k_t)⊆A_t`、`k_t≤t`：

`F_actual(t)=F_reference(k_t)+(t−k_t)−L_t`；

`F_reference(t)≤F_reference(k_t)+(t−k_t)`；

所以 `F_actual(t)−F_reference(t)≥−L_t`。

固定 `alpha=1/sqrt(B)`，维持 `L_t≤1+alpha×F_actual(t)`。新候选查询前，按最坏一次负例预留额度：`L_before+1≤1+alpha×F_before`；不满足则执行参照。真实碰撞增加额度，参照接纳已测负例也释放额度。

于是每个前缀满足：

`F_actual(t)≥(F_reference(t)−1)/(1+alpha)`。

原始累计发现面积差至少为 `−sum_t L_t`。这是相对完整自适应参照的计数界，不依赖工作概率校准。条件为固定池和执行响应、唯一查询、匹配初始化及随机状态；它限制损失，不保证增益。当前放宽额度后，不能宣称终点最多损失一个碰撞；跳过参照步骤后，也不再保证回放前缀至少 105 项。

### 3.5 当前必要对照

| 方法 | 对照作用 |
|---|---|
| 冻结参照 | 原风险反馈链与原查询规则 |
| EP 混合评分 | 最新阶段的直接控制，使用安全数值密度与碰撞概率更新权重 |
| EP 事件评分 | 最新候选，仅用二元事件预测评分更新权重 |
| ADF 发现损失支持 | 较强已评估条目，使用先前 ADF 推断 |
| 既有 Student-t 受保护／无保护方法 | 同六来源、同允许反馈；两者分别在早期面积和终点召回具有优势 |

相对原参照还存在事件反馈权限差异，机制归因依赖同信息对照。原 14 个基线的权限及结果保持冻结，完整表只在结果文件维护。

## 4. 评价与当前证据

### 4.1 固定指标

设全池碰撞数为 N₊、累计发现数为 F_t，共同主指标为：

`Area=sum_(t=1..200) F_t/(200×N₊)`；

`Recall=F200/N₊`。

同时报告六个预算节点、遗漏、预算上界达到率、选择耗时及物理成本。QDScore 和覆盖仅作辅助评价，不参与当前选择或成功条件。逐池上界为 `F_t^max=min(t,N₊)`；终点找全不意味着早期 Area 为 100%，N₊>200 时召回还受预算限制。无碰撞池单列。

汇总先在 SUT 内进行，再对固定条目等权。算法族关联、重复选择序列和更多训练记录不能增加独立样本数。Area、Recall 分别归一化，平均 Recall 更高时平均 F200 仍可能更低。

12 个实测池、24576 次物理执行及 14 个基线的 840 条曲线已核对。完整逐系统对比见 [现有基线结果](../research/representative_sut_testing/results/baseline_comparison.md)。

### 4.2 最新有限筛查

以下是同六 SUT、两池、种子 11、B=200 的开发结果，不与五种子基线均值混作同一组统计证据。最新事件评分阶段新执行 12 条策略曲线、披露 2400 次反馈，复用 24 条冻结参照／原 EP 控制，无新增物理仿真。

| 方法 | Area % | Recall % | 平均 F200 |
|---|---:|---:|---:|
| 冻结参照 | 52.837 | 76.885 | 118.92 |
| 既有 Student-t 受保护方法 | 53.048 | 77.157 | 119.25 |
| 既有 Student-t 无保护方法 | 53.025 | 77.473 | 118.50 |
| ADF 发现损失支持 | 53.052 | 76.915 | 119.08 |
| EP 发现损失支持 | 53.040 | 77.067 | 119.75 |
| EP 事件评分 | 53.051 | 77.067 | 119.75 |
| 仅事件符号报告 | 53.088 | 77.011 | 119.50 |
| 安全风险顺序报告 | 53.075 | 76.945 | 119.42 |

EP 相对 ADF 的 Recall +0.153、Area −0.012 个百分点；相对 Student-t 的 Area −0.008、Recall −0.089。VI-TTC 池 1 比 ADF 多 9 个发现，FVDM 池 1 少 1 个，其余终点数相同。本阶段不晋升、不补种子。[完整结果](../research/representative_sut_testing/results/ep_discovery_supported_prior_coupling/report.md)、[较强方法对比](../research/representative_sut_testing/results/ep_discovery_supported_prior_coupling/stronger_method_comparison.json)。

此前同种子核查重算了 228 条曲线，覆盖 14 个冻结基线、当时已评估机制与额外历史数据参照。该表的前沿仅适用于当时条目；后续加入高斯任务均值和仅符号模型后，应以 [当前重点比较](../research/representative_sut_testing/results/current_method_comparison.md) 判断取舍，目前 Area 和 Recall 没有单一冠军。全部历史坐标的旧 `pooled_reference` 为 Area 53.294%、Recall 77.752%，按额外历史数据及不同初始化单列，不混为同信息消融。[历史同种子比较](../research/representative_sut_testing/results/matched_first_seed_comparison/report.md)。

事件评分相对原 EP 的 Area +0.010、Recall 不变，12 个终点数全部一致，4 条查询轨迹完全相同。它比 Student-t 受保护方法的 Area 高 0.002、Recall 低 0.089；比无保护方法的 Recall 低 0.405 个百分点，仍没有双指标优势。27 项检查、全部前缀事件评分恒等式、实际动作与损失界核对通过；数值代码快照与 401 个冻结基线输入保持一致，不晋升、不补种子。[阶段报告](../research/representative_sut_testing/results/event_evidence_prior_coupling/report.md)、[结果核查](../research/representative_sut_testing/results/event_evidence_prior_coupling/outcome_audit.json)。

25 项相关检查及实际反馈、权重、联合概率、参照回放与逐前缀计数界核对通过。全部 EP 更新在最多 47 次迭代内收敛，平均每条曲线选择进程耗时约 145.6 s。453 次跳过参照步骤的预测局部损失总和约 13.565，仅用已付费标签能确定的实际损失区间为 16–20。未披露参照标签不补读；选择出的样本也不是独立同分布校准集。[推断与损失核查](../research/representative_sut_testing/results/ep_discovery_supported_prior_coupling/inference_calibration_audit.json)。

此前 ADF 的对应预测约 12.531，已付费标签支持实际损失 25–26，显示工作损失概率偏乐观。[校准诊断 (archive)](../research/representative_sut_testing/results/retired_experiments.zip)。两个实际区间来自不同策略轨迹，不能直接归为纯推断校准提升，也不能将局部配对损失等同全程策略遗憾。

### 4.3 已完成尝试的取舍

| 已评估机制 | 对当前设计的影响 |
|---|---|
| 历史纠正收益学习、方向特征 | 留出配置泛化不足，不接入当前选择 |
| 扩展同族历史配置 | 未解决跨算法迁移，恢复原六来源 |
| Student-t 风险尺度 | 反馈可调整风险尺度，保留为较强对照 |
| 有限预算前瞻 | 改变实际选择，未改善双指标；失败实现保存在 [源码归档 (archive)](../research/representative_sut_testing/results/retired_experiments.zip) |
| 固定交替与反馈遮罩 | 参照执行恢复部分发现，直接训练反馈未形成稳定双指标增益 |
| 混合余量观测 | 比同先验二元对照略好，仍低于参照 |
| 中性先验与在线证据 | 减轻部分历史先验偏差，预测可辨别不等于发现优势 |
| 发现损失支持、EP 与事件评分 | 改变部分观测顺序，仍无双指标优势；仅调整评分没有增加终点发现 |
| 独立事件模型与核几何 | 120 个已付费前缀诊断未支持删除风险数值或改用坐标核；没有展开策略实验 |
| 安全风险顺序实际对照 | 数学不变性与推断一致性成立，实际双指标仍未获优势；不保留为优胜主链 |

逐轮模型、参数与完整数值统一在 [阶段报告](../research/representative_sut_testing/results/stage_report.md) 及各阶段报告维护，不在本文重复实验流水。旧数值实现以各阶段 `source_inputs.zip` 为准，不能用当前代码冒充旧方案复现。

## 5. 下一步创新与有限实验计划

### 5.1 需要建立的结论

| 研究问题 | 下一设计应明确的内容 |
|---|---|
| 选择是否可辨别 | 响应解释何时共享可接受选择，何时确实需要区分 |
| 少量反馈是否足够 | 响应分歧、收益间隔、剩余预算与查询机会成本如何决定辨别难度 |
| 不能辨别的原因 | 区分信息不足、观测模型失配、历史迁移偏差与预算内不可辨别 |
| 继续探测是否值得 | 后续发现收益能否覆盖探测损失，以及参照自身会获得的信息 |
| 怎样回退 | 在动态剩余池和自适应参照下控制全程损失，评估保护阻止学习的代价 |

当前优先解决**观测模型与损失概率的可信程度**。EP 已改进近似推断，但收敛后仍有预测损失与实测结果差距；下一机制不能仅通过调整已知池上的支持阈值取得改善。

可保留的分析工具是选择对比 D 的可观测性：Gaussian 工作模型中，数值查询 q 的对比方差减少量为 `Cov(D,Z_q)²/Var(Z_q+noise)`，越界反馈再乘截断矩因子。它衡量反馈与当前选择差异的关系，仍不等于碰撞收益界；若作为理论主线，须明确模型失配范围与主动查询下的有效性。

GP、模型平均、EP 和一般前瞻都是已有工具。研究贡献应由具体可辨别性结论、机会成本分析与独立确认建立。

### 5.2 阶段执行规则

1. **先定义一个修改。** 选择一项具体缺口，说明机制、成立条件、失败情形与回退规则；先核对观测语义、必要代数和信息权限。
2. **独立存放结果。** 在研究目录的 `results/` 下新建按机制命名的子目录，保存协议、源码快照、查询记录与审计；旧方法和结果不覆盖，不在 `docs/` 新增设计文件。
3. **有限筛查。** 固定六 SUT、两池、种子 11、预算 200。候选与两个必要控制形成 36 条核对曲线，可复用相同控制，不重复物理执行。核对初始化、唯一查询、反馈权限与保护界。
4. **阶段决策。** 新候选须在 Area、Recall 两项均值上同时优于冻结参照与直接机制对照，并与较强已评估方法比较、检查逐 SUT 退化。通过后才考虑其余四种子；未通过先总结，不追加次数取得名义胜出。
5. **独立确认。** 方法、实际改进门槛、统计方案及跨轮规则须提前锁定，在未用于修改方法的确认池检验，并考虑场景池与算法族关联。当前反复分析的池均属于开发，不能称盲测。

继续有限阶段和进展总结；文献设置提供出处与可比性，不保证新方法获胜。

### 5.3 本轮锁定验证：用事件评分更新解释可信度

本阶段 12 条新曲线已全部完成，27 项相关检查通过，数值代码与结果冻结。聚合结果见第 4.2 节，未晋升、不扩展种子。

已付费 EP 轨迹中，FVDM、MOBIL、PPO 的六条路径，安全数值反馈对历史／中性解释的累计对数赔率贡献为 +44 至 +55，碰撞反馈贡献为 −10 至 −18。安全数值残差没有普遍极端异常，不能直接归因于离群值。将安全样本也按二元事件评分后，其贡献仍支持历史解释，只是强度下降；同轨迹重放并未证明总体预测改善。[证据通道诊断 (archive)](../research/representative_sut_testing/results/retired_experiments.zip)。

本轮只改变在线权重的评分对象。两个 EP 组件仍接收全部真实 R/C，先验、核、噪声、选择支持阈值与实际额度不变；权重使用查询前事件概率：`w_(t+1,h)∝w_(t,h)×p_(t,h)(C_(t+1))`。连续安全代理密度不直接参与权重更新，但历史风险仍通过组件后验影响下一次事件预测。入口为 [evaluate_event_evidence.py (archive)](../research/representative_sut_testing/results/cleanup/source_inputs.zip)，单独保存到 `results/event_evidence_prior_coupling/`。

该规则是针对事件预测的在线专家聚合，不称作完整 R/C 联合数据的精确贝叶斯后验。令 L_h 为组件在实际共同查询轨迹上的累计二元对数损失，初始权重均为 1/2，则 `L_mix=−log(sum_h exp(−L_h)/2)≤min_h L_h+log(2)`。该恒等式可逐前缀核对，允许组件随已付费反馈更新；比较对象是在同一实际轨迹上的预测，不是各组件独立选例后的发现曲线。[专家预测原始研究](https://www.sciencedirect.com/science/article/pii/S0022000097915567)。

采用模块信息流控制的动机参考 [Semi-Modular Inference](https://proceedings.mlr.press/v108/carmona20a.html)，本实现不等同其完整推断方案。对数损失界不保证校准或碰撞发现提升。本轮实际跳过 434 个参照时段，预测局部损失总和 12.801，已付费标签支持实际损失区间 18–22；评分对齐仍未解决工作概率偏差。两个既有解释在 VI-TTC、MCTS 的四条轨迹上产生完全相同的选例，下一步优先检查它们共同的响应假设与几何结构。

### 5.4 已完成结构检查：独立事件响应与核几何

在相同已付费前缀上，用只接收 C 的 probit 事件 GP 检查中性余量组件的替代结构。首轮采用单位潜变量幅度，原中性组件的平均对数损失为 0.295，既有核／坐标核二元组件为 0.378／0.390，虽改变多数排序，但预测更差。该配置还改变了原信噪比，不能将退化全部归因于删除 R。[首轮诊断 (archive)](../research/representative_sut_testing/results/retired_experiments.zip)。

随后锁定尺度匹配：两个几何组件的 probit 噪声均为 1，潜变量方差为冻结风险核的 `K_frozen(x,x)/noise_frozen`，共享一半场景类偏移、一半归一化相关性；坐标 RBF 长度尺度沿用 0.15。另加与原中性组件完全同先验、同核、同噪声的二元消融。32 项检查通过，两轮共 120 个前缀状态，无新增目标查询。

| 匹配尺度后的组件 | 事件对数损失 | Brier 损失 | 改变纯组件提议数 |
|---|---:|---:|---:|
| 原中性混合 EP | 0.2954 | 0.0927 | — |
| 同先验二元 EP | 0.3051 | 0.0939 | 22/60 |
| 既有相关性二元 GP | 0.3001 | 0.0941 | 36/60 |
| 坐标核二元 GP | 0.3214 | 0.0988 | 52/60 |

没有稳定预测优势，不展开策略实验。风险数值作用随池而变化，不能按 SUT 名称选取有利版本，也不能将排序改变当作 Area／Recall 提升。[匹配诊断 (archive)](../research/representative_sut_testing/results/retired_experiments.zip)。当前证据不支持简单删除 R，下一步应保留其中的信息并放松它到事件余量的数值映射。

未知核超参数及模型失配理论依赖有界 RKHS 范数、噪声和近似误差条件，不直接提供本项目的有限预算发现保证。[未知超参数 BO](https://www.jmlr.org/papers/v20/18-213.html)、[失配 GP Bandit](https://proceedings.neurips.cc/paper/2021/hash/177db6acfe388526a4c7bff88e1feb15-Abstract.html)。当前模型无关的保护仍为第 3.4 节实际计数界。

### 5.5 已完成观测候选：保留风险顺序，放松固定数值尺度

原混合余量将安全反馈 `1−R` 当作精确 Gaussian 报告；二元模型又丢掉其连续信息。下一候选采用部分顺序观测：保持 C 的事件符号，并在同类无碰撞样本间保留 R 的排序，不要求它等于已知尺度下的余量。

设潜在报告为 W：碰撞要求 `W_i≤0`，无碰撞要求 `W_i>0`；若两个无碰撞样本同属功能场景且 `R_i<R_j`，则要求 `W_i>W_j`。不同功能场景不比较，相同 R 不人为排序，碰撞时的 R 不进入安全排序。它没有假定存在一个将所有 Risk 值分开为碰撞／安全的阈值，W 也不是实际车距。

将这些条件组成区域 O(D)，以 `1{W∈O(D)}` 更新 Gaussian 报告先验。使用已有相关性与噪声比例，不新增坐标核或参数扫描；在原混合组件、同先验二元组件与顺序组件之间做前缀诊断。排序条件用相邻不同值组表达，避免重复约束；GP 推断需支持两点线性约束，而不只是逐点符号。

在固定先验和 C 下，任一功能场景内对目标 Risk 作严格递增变换，约束区域不变。这项尺度不变性可以直接推导与测试；它只针对顺序组件，不针对仍使用数值 Risk 的冻结参照或历史组件，也不保证排序关系适用于任意目标。扩展秩似然为处理未知边际尺度提供依据，原文采用 Gibbs，本项目的线性约束 EP 是另行实现的近似。[Hoff 2007](https://arxiv.org/abs/math/0610413)。

该方案已完成单约束精确矩、变换不变性、输入顺序一致性和前缀预测核查，并接入两解释的实际选择器。事件评分、200 次预算、独立参照与真实额度保护保持固定；数值性质成立，但尚未形成发现优势。

现已实现 [rank_event_response.py (archive)](../research/representative_sut_testing/results/cleanup/source_inputs.zip)。37 项相关检查通过；60 个前缀状态的仅符号报告与原 probit 预测最大差 `6.88e−8`，顺序 EP 最多 68 次迭代收敛，保持顺序的 Risk 重参数化产生相同约束。顺序组件对数损失 0.3012、原数值组件 0.2954，尚无总体预测优势，但改变了 26 个提议。[前缀报告 (archive)](../research/representative_sut_testing/results/retired_experiments.zip)。

下一步按实际发现目标做一次有限对照，而不把少量验证点的预测损失代替发现曲线。原数值版本复用既有结果，新增仅符号和安全顺序各 12 条曲线，共 24 条；核心三项机制形成 36 条对照曲线，冻结参照另行复用，全部仍为六 SUT、两池、种子 11、200 次唯一查询。历史混合 EP 不变，只改变第二解释的观测关系。入口 [evaluate_risk_order.py (archive)](../research/representative_sut_testing/results/cleanup/source_inputs.zip)，结果独立存入 `results/safe_risk_order_coupling/`。实现接入的 38 项检查通过；没有核、先验、阈值或 SUT 参数搜索，未通过双指标则不补种子。

该实际对照现已全部完成。仅符号 Area 53.088%、Recall 77.011%；顺序为 53.075%、76.945%，后者相对前者 −0.013、−0.066 个百分点。两种机制相对原数值版本均为 Area 小幅提高、Recall 下降，不晋升、不补种子。[完整报告](../research/representative_sut_testing/results/safe_risk_order_coupling/report.md)。数值／符号／顺序这组观察变体停止扩展；下一步从实际选择损失检查决策机制，不再凭预测指标或新增表示宣布创新。

### 5.6 已完成并退役：让反馈决定证据共享范围

当前可信度权重跨两个功能场景共享，但同一预测解释不一定在切入和跟驰上同样适用。已执行轨迹的影子核查显示，仅符号版本的累计事件对数损失：全局 822.107、分场景 814.564、混合共享范围 814.134。数值版本没有对应收益；这是对仅符号版本开展一个具体对照的依据，不是新的发现结果。[诊断 (archive)](../research/representative_sut_testing/results/retired_experiments.zip)。

保留两个原 GP 预测器及其全部付费 R/C 更新，只调整解释可信度如何汇总。全局汇总使用所有事件证据；分场景汇总只用当前功能场景的事件更新该组权重；自适应版本再用查询前事件预测评分，更新这两种共享范围的权重。选择与联合损失使用当前参照场景的有效组件权重，不使用目标身份。

令 L_G、L_F 为全局／分场景汇总在实际共同查询序列上的累计二元对数损失，自适应范围满足 `L_A=−log((exp(−L_G)+exp(−L_F))/2)≤min(L_G,L_F)+log(2)`。分场景汇总还满足 `L_F≤sum_g min_h L_(g,h)+2log(2)`。这些为在线专家聚合的标准代数性质，允许预测器使用因果反馈更新；它们约束预测损失，不保证发现增益。[专业化预测器原始研究](https://doi.org/10.1145/258533.258616)。

可检验的问题是：少量反馈是否足以决定哪些证据应共享；若不能辨别，就保持范围不确定性，并由原实际计数保护控制发现损失。入口 [evaluate_evidence_scope.py (archive)](../research/representative_sut_testing/results/cleanup/source_inputs.zip)，单独保存到 `results/evidence_scope_coupling/`。只新增分场景／自适应各 12 条曲线；全局仅符号版本与原参照复用，SUT、场景、历史、预算及保护规则不变，不搜索权重或核参数。

该对照已全部完成。分场景 Area 53.005%、自适应共享 53.070%，均低于全局控制 53.088%；三者 Recall 都为 77.011%，12 个终点发现数全部一致。42 项检查与全部实际反馈、权重和前缀恒等式通过，不晋升、不补种子。[共享范围报告 (archive)](../research/representative_sut_testing/results/retired_experiments.zip)。当前重点方法的同种子结果见 [完整比较](../research/representative_sut_testing/results/current_method_comparison.md)；早期 Area 和最终 Recall 没有单一冠军，不能将最近实现称为最佳。

## 6. 理论依据与文件约定

本轮清理后只保留 Student-t 与反馈纠正的评价入口。退役源码、配置和结果链接指向完整归档；归档内保持原相对路径，历史阶段的数值结论不改写为当前实现。代码职责及运行方式统一见 [实现索引](../research/representative_sut_testing/README.md)，清理范围与验证见 [清理记录](../research/representative_sut_testing/results/cleanup/report.md)。本轮没有启动新实验。

| 原始研究 | 可借鉴内容 |
|---|---|
| [GPML](https://gaussianprocess.org/gpml/chapters/RW3.pdf)、[Student-t 过程 2014](https://proceedings.mlr.press/v33/shah14.html) | Gaussian／非 Gaussian 响应推断、EP、共同尺度后验 |
| [COBALT 2026](https://proceedings.mlr.press/v337/karlova26a.html) | 区分数值与区间事件；已知截断条件不能直接当作本项目 Risk 语义 |
| [ENS 2017](https://proceedings.mlr.press/v70/jiang17d.html) | 剩余预算下的非短视发现；一般前瞻结构已有研究 |
| [EC² 2010](https://proceedings.neurips.cc/paper/2010/hash/1e6e0a04d20f50967c64dac2d639a577-Abstract.html)、[决策区域 2014](https://proceedings.mlr.press/v33/javdani14.html) | 支持选择无须唯一识别全部响应假设 |
| [Partial Monitoring 2023](https://www.jmlr.org/papers/v24/22-1248.html)、[BAI 2016](https://proceedings.mlr.press/v49/garivier16a.html) | 可观测性与区分复杂度 |
| [传导线性 Bandit 2019](https://papers.neurips.cc/paper_files/paper/2019/hash/8ba6c657b03fc7c8dd4dff8e45defcd2-Abstract.html)、[R-IDeA 2026](https://proceedings.mlr.press/v300/tang26d.html) | 测量方向、决策差异与误设的误差放大 |
| [保守 Bandit 2017](https://papers.nips.cc/paper_files/paper/2017/hash/bdc4626aa1d1df8e14d80d345b2a442d-Abstract.html)、[SPIBB 2019](https://proceedings.mlr.press/v97/laroche19a.html)、[保守交替探索 2019](https://proceedings.mlr.press/v89/katariya19a.html) | 参照约束与回退；本项目回放计数界单独推导 |
| [Semi-Modular Inference 2020](https://proceedings.mlr.press/v108/carmona20a.html) | 误设下的模块信息流控制 |

上述保证依赖各自条件，不能直接移植为本项目的碰撞发现保证。

**文件约定：** `docs/` 只保留本文与 [style.md](style.md)。本文统一更新现行设计；研究目录的唯一 README 保存职责、入口及结果索引；实验记录、完整对比与旧设计依据保存在结果报告和冻结归档。文件按内容命名，不新增带版本号的平行设计文档。
