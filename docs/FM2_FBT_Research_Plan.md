# FM²-FBT 研究方案 V7.2：证据化失效记忆、变维编码与每场景 50 次测试

> 更新日期：2026-09-28。  
> 修订基础：用户上传的 `FM2_FBT_Research_Plan(1).md`（V7.1）与最新的 `FBRT_Scenario_Parameter_Space_V3.md`。  
> 环境：**highway-env**；研究目标：**有限预算内发现目标算法真实失效**。  
> 预算：**每个功能场景族、每个目标 SUT、每个方法的一次独立会话最多 B=50 次目标执行**。  
> 状态：本文是研究设计与实施规格。未实现、训练或验证本文新增网络；没有新的仿真性能结果。  
> 场景参数轴以本地 `methods/failure_memory_regression/configs/scenario_parameter_space.yaml` 为准；早期远端快照尚未包含这些工作区文件。候选清单已经本地生成，目标 SUT 响应仍未测量。

---

## 0. 本次修改的执行摘要

这次不改变研究问题，也不增加第四个主要算法组件。仍为：

**Historical Failure Mechanism Memory → Few-Shot Target Adaptation → Diverse Failure Expansion（DFE）。**

修订落实以下事项：

1. 将历史知识定义为有证据出处的**失效模式记忆基元**，不是一段自然语言，也不只是一组聚类中心。
2. 每个基元包含适用条件、历史失败见证、同系统通过对照、可观察事件描述与证据支持度。这里的“机制”是观测性行为模式，不是经因果识别证明的软件根因。
3. 参数编码采用“参数语义＋数值”的变长 token 集合；**输入行数可变，token 宽度固定**。不再把“支持变长输入”夸大成“自动理解任何新参数”。
4. 区分 set self-attention 的**置换等变性**与池化／读取结果的**置换不变性**。允许批处理 padding，但必须正确 mask。
5. 第二组件增加明确的**目标证据条件化记忆检索**：目标测试反馈不只是加到最终分类器，还会改变候选对历史基元的检索权重。
6. 网络升级为 128 维层次集合编码、数值分段线性嵌入、关系感知检索；轻量参数共享预测头参考 TabM。不是重新引入 TabPFN，也不因模型较新就预设有效。
7. DFE 适配 V3 的不规则 Sobol 点：采用语义一致的归一化距离与近邻，不使用二维 `grid_index` 或四邻接。
8. 两臂路由是两个选例策略的在线选择，不是两个驾驶算法。默认开发方案采用有折扣的 Beta-Bernoulli Thompson 启发式；其最优性不能由标准平稳 bandit 理论直接推出。
9. 主预算改为 B=50，报告 5/10/20/30/50 前缀。八族全部适用时，一个目标、一种方法的一轮合计最多 400 次逻辑目标查询。
10. 仍然只保留两类论文实验：与完整有限库真值比较，以及与强基线比较。模块单元测试和代码审计不另包装为研究实验。

---

## 1. Motivation 与问题边界

新接入的驾驶算法需要在多类交通条件下测试。历史算法已积累大量执行记录，其中的失败及附近通过样本能够提供冷启动线索；但是不同驾驶算法并不共享完全相同的失效模式。只根据历史失败距离或历史边界排序，可能漏掉目标独有失效；完全忽略历史，又会浪费已经获得的测试证据。

因此，本文研究：**如何将历史失败组织成有条件、有对照、可被新证据否定的记忆；用少量目标执行纠正历史的适用性；在发现真实目标失败后，利用其邻域继续提高失效命中率。**

通俗概括：

> 历史经验提供待验证的线索，目标反馈决定哪些线索值得继续相信，目标失败提供后续局部挖掘的起点。

本文不要求历史与目标存在父子版本关系，不要求模型单调进步；不以精确恢复连续失效边界或自然驾驶事故率作为主目标。历史 FAIL→目标 FAIL 与历史 PASS→目标 FAIL 都是有效目标失效发现。目标通过结果用于学习，但不计入发现奖励。

### 1.1 每个功能场景族是一个独立任务

定义任务：

\[
\mathcal T_{f,*}=(\mathcal X_f,\mathcal H_f,S_*,B=50).
\]

- \(\mathcal X_f\)：预先冻结的有限候选池。
- \(\mathcal H_f\)：允许使用的同族历史记录；不含最终目标 SUT 的封存结果。
- \(S_*\)：目标驾驶算法。
- \(B\)：本会话的全部目标执行预算，包含启动、局部扩展及无法判定的执行。

不同 family 不竞争这 50 次预算。在线 support、目标种子、路由状态均按会话重置。八个功能场景是八个异质任务，不是八个独立随机重复；同一目标跨族共享训练权重可以，但不免费共享本轮已查询的目标答案。

### 1.2 主目标

\[
\mathcal F_{*,f}=\{x\in\mathcal X_f: S_*(x)\text{产生有效失效}\},
\quad
D_f(B)=|Q_{f,B}\cap\mathcal F_{*,f}|.
\]

优化目标是尽早、尽可能多地发现真实目标失效。模型预测不算发现；几何簇不等于独立软件缺陷；一个场景上的无碰撞也不构成整体安全证明。

---

## 2. 调研结果：每项借鉴解决什么，不解决什么

| 工作与来源 | 与本项目直接有关的内容 | 本方案的使用边界 |
|---|---|---|
| IFR / FSB / DSB，[R1] | 根据失败模式的位置、形状与方向，从真实失败出发探索 | 不继承单一凸失效区假设；不免费给种子；不以凸包补目标标签 |
| ScenarioFuzz，ISSTA 2024，[R2] | 历史测试数据辅助种子筛选；事故轨迹聚类描述交互模式 | 说明“历史＋预测器＋局部测试”不是新概念；不搬入 CARLA/GNN 整套环境 |
| Set Transformer，ICML 2019，[R3] | SAB / PMA 处理无序集合，显式建模集合元素关系 | 基础结构；不是新颖性本身，不自动去重，也不自动理解参数语义 |
| Attentive Neural Processes，ICLR 2019，[R4] | 查询位置读取相关的少量输入—输出观测 | 借鉴 support-conditioned prediction；本方案是二值失效排序，不照搬回归理论 |
| 数值特征嵌入，NeurIPS 2022，[R5] | 标量在进入网络前做分段线性或周期嵌入 | 改善连续量表达；历史/开发确定尺度，不能用封存目标拟合预处理 |
| TabR，ICLR 2024，[R6] | 检索样本的标签与 query–neighbor 差值共同参与预测，训练时排除自身检索泄漏 | 借鉴“检索关系＋证据”，不是把所有 SUT 标签混成同一系统真值 |
| TabM，ICLR 2025，[R7] | 参数高效的多预测分支，改善表格学习的性能／成本权衡 | 仅升级融合预测头；不替换失效记忆；分支分歧不是经校准的真实不确定性 |
| Thompson Sampling 及折扣变体，[R8–R9] | 根据观测收益在策略之间探索与利用，折扣降低陈旧收益影响 | 我们的策略会变化、候选会耗尽，因此只是适配启发式，不照搬 regret 保证 |

调研不支持以下声明：历史失效语义首次用于测试、attention 首次用于少样本推断、首次用失败种子扩展、两臂路由首次用于自适应选择。因此，贡献应落在**证据基元怎样组织、目标证据如何纠正检索，以及统一执行预算下是否真的增加发现**。

---

## 3. 与当前实现的关系及必须先修的接口

远端快照中的 `pattern_memory.py` 已有失败见证、通过对照、中心／半径、事故对象／阶段、TTC／clearance、来源等字段。[C1]

但不能直接假设它已经完成 V3 变维适配：

- 多数普通模板的 `PARAMS`／`NEW_BOUNDS` 仍列两个活动轴。
- 缺少字段时 `active_values()` 的旧路径可能返回 `(0.5, 0.5)`。
- `semantic_key(record)[:3]` 用于旧聚类分组，没有按所有碰撞对象／阶段拆成独立机制。
- 原 `RBFDictionary` 对每模板多个物理上下文存在限制。
- `occurrence_by_build` 必须只从已授权历史计算；episode 伪目标也必须从所有汇总字段移除。

新模块必须增加严格的 schema 适配器，不能让 5D 输入在旧路径里悄悄降成 2D，也不能把 `parameterization_version` 名称相似的不同契约混为一谈。

附件声明八族新轴已接线，但本轮远端没有对应 YAML 的可访问内容。正式运行前必须对本地 YAML、manifest、runner 生成统一代码指纹，独立检查每个活动轴确实影响初始化或事件；本文件不代替这项核验。

---

## 4. 历史知识建模：失效模式记忆基元

### 4.1 为什么不直接把事故日志交给 Transformer？

日志包含大量重复、缺失和与候选无关的信息。只保存 cluster center 又会丢失关键对照。因此先在结构层组织证据，再在网络层学习检索。

定义基元：

\[
P_k=(\mathcal C_k,\mathcal W_k^F,\mathcal W_k^P,\Delta_k,\mathcal O_k,\mathcal R_k).
\]

| 字段 | 内容 | 用途 |
|---|---|---|
| \(\mathcal C_k\) 条件 | family、schema、固定上下文、连续参数分布、事件语义 | 说明该证据适用于什么条件 |
| \(\mathcal W_k^F\) 失败见证 | 同一历史 SUT 的真实失败；保留最多四个分散代表及全部来源 ID | 不让一个平均中心抹掉区域结构 |
| \(\mathcal W_k^P\) 通过对照 | 同源、同契约下真实通过的邻近样本；允许为空 | 告诉模型哪些相似条件未失败 |
| \(\Delta_k\) 成对差异 | 每对 failure/pass 参数差值、距离、有效维数和缺失标记 | 描述经验转变方向，不宣称因果 |
| \(\mathcal O_k\) 行为描述 | 对象角色、事件阶段、观测到的相对事件时间、TTC/净间隙统计与 missing mask | 区分参数接近但行为不同的模式 |
| \(\mathcal R_k\) 证据与来源 | 数量、schema/执行契约、source family、允许使用的跨源支持摘要 | 支持审计，避免重复版本淹没其他来源 |

`source_build_id`、执行 ID、checkpoint hash 用于审计与划分，不直接作为能够记住目标弱点的输入。第一版允许使用开发时已知的粗 source family 标签；未知来源用 `UNK_SOURCE`。不得输入带故障含义的目标 build 名称或修改激活标志。

### 4.2 “机制”的准确含义

例如“切入后短时间制动、合流后与 lead 碰撞”是可观察的交互模式。

它不证明系统失败的内部原因是感知延迟、规划错误或制动模块缺陷。除非另外取得受控干预或内部故障证据，不能将这些内部因果标签填入基元，也不能从 attention 权重推断因果归因。

### 4.3 从原始记录到基元

1. 过滤到授权的历史源、相同 family、兼容 schema 和有效执行；排除封存目标数据与不确定标签。
2. 保留旧代码的局部组件思路。在同一 source、固定上下文／契约内，以归一化连续坐标构建 mutual-kNN 图。新 schema 的距离阈值按历史／候选几何标定，不能沿用二维常数并假定高维效果相同。
3. 如果已经观测到不同 collision partner / maneuver phase，先分组或在组件内明确保存多模式，不因几何接近强行称同一机制。未知描述单列 `UNKNOWN`，不推断补齐。
4. 保留分散的 failure witnesses；为各 witness 寻找同源通过对照。距离过远时保留“无局部对照”的状态，不伪装成近邻边界。
5. 保存逐维中心、标准差、成对差异集合／分位数，而不只保留差值平均，避免相反方向抵消。
6. 统计描述字段及有效支持数量；保存全部执行 ID 以便重算。
7. 基元上限开发起始为 64 个。按来源均衡、语义分组和几何分散选择；被截断数量入账。原始历史不删除。

单个失败没有足够证据称为完整区域，可保留 singleton 基元。聚类是存储／检索近似，不是目标未执行区域的真值。

### 4.4 没有历史失败怎么办？

区分三种情况：

- **确有大量通过记录但没有失败**：保存 `PASS_ONLY_COVERAGE` 摘要与少量真实通过代表，不制造 failure token。
- **记录不足／字段缺失**：显式支持不足与 missing mask。
- **完全没有兼容历史**：使用 `NO_HISTORY` token。

所以“没有 failure PatternCard”不等于“没有历史信息”，更不等于目标风险为零。每次历史 attention 都额外允许一个 `NULL_RETRIEVAL` token，模型可以选择不使用任何具体失败基元。

### 4.5 每个基元如何变成 key/value？

- **Key**：主要编码该模式发生时的物理条件与交互结构，用来回答“当前 candidate 与它是否相关”。
- **Value**：编码失败／通过见证、成对差异和行为描述，用来回答“这份相关证据告诉我们什么”。

二者分开，避免仅因为两个样本都贴了同一种碰撞标签就把输入条件视为近邻。候选本身没有事故结果，历史 value 中有真实历史事故结果；两者权限不同。

---

## 5. 变维编码：不是消除维度，而是让参数个数不决定网络大小

### 5.1 三个数量必须分清

- \(d_f\)：该 family 真正参与搜索的活动参数个数；决定几何空间维度。
- \(L_f\)：送入网络的 token 数，可包含活动参数、固定上下文、语义关系；不一定等于 \(d_f\)。
- \(d_{model}=128\)：每个 token 的向量宽度，是固定网络超参数。

**变的是 token 行数，不变的是每行128个特征。** 3个参数不需要3维网络，5个参数不需要5维网络。

### 5.2 实例：同一套编码器输入三行或五行

S06 的三个活动字段：

| 字段 | 示例值 | 附件范围 | 归一化值 |
|---|---:|---:|---:|
| 初始间隙 | 30 m | 15–90 m | 0.200 |
| 前车速度 | 22 m/s | 20–27 m/s | 0.286 |
| 自车初速 | 25 m/s | 20–30 m/s | 0.500 |

S08 的五个活动字段：

| 字段 | 示例值 | 附件范围 | 归一化值 |
|---|---:|---:|---:|
| 初始间隙 | 30 m | 10–60 m | 0.400 |
| 切入车初速 | 20 m/s | 17–23 m/s | 0.500 |
| 换道时间尺度 | 2 s | 1.5–3 s | 0.333 |
| 制动减速度 | 4 m/s² | 1–6 m/s² | 0.600 |
| 实测并线后制动延迟 | 0.4 s | 0–1 s | 0.400 |

这些只是编码算例，不是已执行测试结果。分别得到 `3×128` 与 `5×128` 的参数 token 矩阵；再用同一个集合编码／池化得到 `1×128` 场景表示。

相同的 30 m 在不同研究范围下归一化值不同，因此模型还需要参数语义、family、单位和冻结 schema 信息；不能只给数字0.2或0.4。

### 5.3 参数 token 的明确公式

将输入字段表示为：

\[
(name,actor,group,type,value,unit,bounds,missing).
\]

- 名称先做语义别名映射，而不是靠 JSON 第几列。
- role 区分 ego、lead、rear；单位先统一到 SI。
- type 区分连续、类别、布尔、固定上下文。
- 只有已声明的连续活动参数用于 DFE 几何。

连续量 \(v\) 的规范化：

\[
u=(v-l)/(h-l).
\]

对固定上下文 \(l=h\) 不能照用除法，应使用该语义量预先定义的尺度，并标为 context token。

采用16段固定分段线性编码作为开发起点：

\[
PLE_j(u)=\operatorname{clip}((u-b_j)/(b_{j+1}-b_j),0,1),
\quad b_j=j/16.
\]

输入同时保留未截断的 \(u\) 与越界标志，所以不同历史越界值不会全部塌缩到同一个边界。该 PLE 是借鉴 [R5] 的工程适配，不宣称是其最佳配置。

\[
t_j=W_v[PLE(u_j),u_j,m_j,o_j]
+E_{name}(n_j)+E_{actor}(a_j)+E_{group}(g_j)+E_{type}(\tau_j).
\]

类别变量走独立 embedding 分支，不能把 lane ID、类别编码的数值大小解释成欧氏距离。未知 name 使用显式 UNK 并发出 schema 审计提示；支持变长输入不保证理解未训练过的新物理语义。

### 5.4 为什么可以重排参数顺序？

不加入“第一行／第二行”的位置编码。参数语义由 name/role 等标签携带。对参数矩阵整体换行，集合编码输出的行对应地换行，池化结果保持不变。

批处理可以用 padding 补到当前 batch 最大长度；**padding 不是坏方法，没 mask 或把位置当语义才是问题**。需要同时屏蔽参数、见证、基元和 support 的 padding。

### 5.5 网络维度统一，不代表几何空间可以混算

S06 与 S08 都编码成128维，不意味着可以在两者的嵌入之间计算 DFE 物理方向。在线测试与几何操作始终保持 family-local，并限定相同固定上下文／类别层。

---

## 6. Set self-attention 是什么？Transformer 在这里做什么？

### 6.1 通俗解释

将一张记忆卡视为一位“提供历史证据的人”。self-attention 让每张卡在形成自己的表示时，参考其他卡：是否描述了相似条件、不同失败模式，或同一种交互在多个历史源中的表现。

它不是对卡片进行时间排序，不是在自动写解释，也不等于数据库去重。

数学形式：

\[
SA(X)=\operatorname{softmax}\left(\frac{XW_Q(XW_K)^T}{\sqrt{d_h}}+Mask\right)XW_V.
\]

代码中 Set Transformer 的 `SAB.forward(X)` 实际调用 `MAB(X,X)`；`PMA` 则用学习的固定 query 对集合读取。[R3]

### 6.2 三处不同的 attention

| 层次 | Query / Key / Value 来自哪里 | 意义 |
|---|---|---|
| 参数 self-attention | 同一场景的参数 token | 学习间隙与速度、延迟与制动等组合关系 |
| 历史 memory self-attention | 同族历史基元 | 在多份相互关联的历史证据之间建模关系 |
| candidate-to-memory cross-attention | query是当前候选，key/value是历史基元 | 为每个候选取出不同的相关证据 |

后面还有 candidate-to-target-support cross-attention，读取的则是目标已执行结果。

### 6.3 等变与不变的区别

对置换矩阵 \(P\)，无位置编码的标准集合 self-attention 满足：

\[
SA(PX)=P\,SA(X).
\]

这是**置换等变**：输入怎么换行，输出也怎么换行。

经过不依赖行号的 PMA 或候选读取后：

\[
Pool(PX)=Pool(X),
\]

才是**置换不变**。不是 self-attention 一步就自动把任意长度压成一个向量。

### 6.4 Transformer 的必要性是一项假设，不是结论

它的用途是学习输入条件之间的非线性关系、基元之间的关系，以及候选到历史证据的相关性。不是为了“网络更新颖”而堆层。

若 mean pooling 或原 FBRT 在同一任务上已经更好，则不能因使用 Transformer 就称方法先进。TabM 的研究也提醒：在一些表格基准上，复杂 attention/retrieval 的成本并不必然换来更高效果。[R7]

---

## 7. 网络升级：一套具体的小型关系记忆网络

### 7.1 开发起始配置

| 子模块 | 规格 | 本次作用 |
|---|---|---|
| 语义/数值 tokenizer | PLE 16段，统一128维，显式mask | 替代直接拼原始标量 |
| 参数集合编码 | 2个Pre-LN attention block；4头；FFN隐藏256 | 学习参数间组合关系 |
| 参数池化 | PMA，1个输出token | 可变长度→128维 |
| 见证／对照集合 | failure/pass分别池化，空集用不同null token | 避免把失败和通过混为一个均值 |
| 记忆编码 | 1个SAB，4头，基元上限64 | 学习历史证据关系 |
| 候选—历史读取 | 关系感知cross-attention | 条件匹配及候选差异进入检索 |
| 候选—目标读取 | 1层cross-attention，support最多50 | 读取已查询目标反馈 |
| 融合预测头 | 2层共享MLP；4个参数高效分支，参考TabM | 降低单预测头过拟合风险 |
| 正则 | dropout 0.1、AdamW、开发集早停 | 初值，不是已证明最优 |

这是V7.1的64维网络的明确升级，不要求训练大语言模型，不接入Jev/TabPFN，不重新训练驾驶策略。若训练源很少，保留64维版本作为开发回退；在确认实验前只冻结一个配置。

### 7.2 TabR 与 TabM 只借鉴什么？

TabR 源码不是只查最近邻：检索到的标签编码和 query–neighbor 表示差同时构造 value，并在训练中去掉自匹配。[R6] 本方案借鉴这种**关系＋证据**表示，历史结果只作为历史 value，目标 support 的标签才属于目标系统。

TabM用于固定长度融合向量后的预测头，不直接接收可变列数据，也不是预训练知识来源。分类时各分支概率求平均：

\[
p(x)=\frac1E\sum_{e=1}^{E}\sigma(s_e(x)),\quad E=4.
\]

训练对各分支分别计算损失后求平均。不能把成员预测当成额外目标样本，也不把分支方差称为已校准的碰撞概率置信区间。

这两项为已有技术的适配；论文贡献不归给“首次使用新网络”。

---

## 8. 组件二：用目标证据纠正历史，而不只是再加一个attention

### 8.1 通俗的第二步

第一步得到的是“我根据别的驾驶算法，怀疑这里容易出问题”。第二步的任务是：**用当前算法真正测到的结果检查这个怀疑，修改对相似候选的判断。**

例如旧系统经常在短延迟切入制动时失败，但新系统连续通过了这类测试。应该降低这些历史模式对相似候选的影响，而不是机械地降低所有切入场景的权重。另一个尚未被目标验证的模式，不应一起被否定。

反过来，历史全通过区域出现目标失败后，目标support分支可以直接提高相似候选得分；不需要先在历史里补造一条边界。

### 8.2 保存“历史错在何处”的显式证据

会话开始，在目标support为空时缓存每个候选的历史预测：

\[
p_H^0(x)=Model(x,M,\varnothing).
\]

它只是该冻结网络的历史冷启动输出，不是真实风险。

对已执行的第i个有效目标结果，形成：

\[
s_i=E_T[q_i,E_y(y_i),o_i,p_H^0(x_i),e_i],
\quad e_i=y_i-p_H^0(x_i).
\]

- \(e_i>0\)：历史判断相对低估了这条已观测结果。
- \(e_i<0\)：历史判断相对高估了这条已观测结果。
- \(o_i\)：只包含执行后真正取得的行为摘要与missing mask。

这是单次预测残差，不是统计上已证实的因果变化或模型失配区域。缓存的 \(p_H^0\) 在同一会话不回写，避免用加入该样本后得到的预测再假称“查询前误差”。

### 8.3 目标证据先影响记忆检索

令 \(k_k\) 为基元的条件key，\(m_k\) 为证据value。对每个基元读取目标support：

\[
r_{k,t}=Attn(k_k,\{q_i\},\{s_i\}).
\]

然后计算证据门控：

\[
g_{k,t}=\sigma\big(MLP_g[m_k,r_{k,t},\log(1+n_t)]\big).
\]

无support时规定 \(g_{k,0}=1\)；有support后才启用门控。允许历史误导时更多转向null基元。

候选检索权重：

\[
\alpha_{k,t}(x)=softmax_k\left(
\frac{(W_Qq_x)^T(W_Kk_k)}{\sqrt{d_h}}
+b_{rel}(x,P_k)+\log(g_{k,t}+\epsilon)-\log c_{source(k)}
\right).
\]

- \(b_{rel}\) 来自同语义参数的差值、共同有效字段数、schema/context兼容性；绝不使用候选目标结果。
- \(c_{source(k)}\) 是该源在记忆集中贡献的基元数；减去其对数是来源均衡的工程起点，不代表源独立。
- `NULL_RETRIEVAL` 的门控不受普通模式衰减影响，避免所有历史都不合适时仍被softmax强制选一张。

\[
h_H(x,t)=\sum_k\alpha_{k,t}(x)V(m_k).
\]

这个门控是**学到的融合系数**，不能解释为“历史正确概率”。训练并不保证每看到一次pass它就单调下降，是否真的纠正误导必须由留出测试与诊断日志验证。

### 8.4 目标support也直接影响最终预测

\[
h_T(x,t)=Attn(q_x,\{q_i\},\{s_i\}).
\]

融合向量：

\[
z_{x,t}=[q_x,h_H(x,t),h_T(x,t),h_H(x,t)-h_T(x,t),
\log(1+n_F),\log(1+n_P),a_H,a_T,t/50].
\]

前三个128维向量与差分共512维，再加5个scalar，共517维；代码用schema计算输入维度，不写死旧194维head。

预测头输出 \(p_*(x\mid M,D_t)\)。网络训练后在线冻结权重，每次更新的是support及attention上下文，不需要每个target test后运行反向传播。

### 8.5 第一个组件怎样实际帮助第二个组件选candidate？

1. 组件一把历史基元编码并缓存，所有候选都能检索相关证据。
2. 没有目标support时，得到历史冷启动排序。
3. 有support后，组件二同时重加权历史检索并读取目标直接证据。
4. 产生每个未查询candidate的一个分数。
5. Global分支从本族整个剩余池选最高分；Local分支从真实目标失败邻域选其局部最优。
6. 路由只在这两个已经提出的候选之间决定本次执行哪个。

任何组件都不“生成真实标签”。最终确认只能来自 `TargetOracle.query()`。

---

## 9. 训练：历史SUT上的少样本任务，而不是对50个目标点训练大网络

### 9.1 两层留出

- **外层**：最终目标SUT或强相关checkpoint/配置组整体留出；其记录不得出现在任何family训练、预处理、基元汇总或checkpoint选择中。
- **内层**：从允许的历史SUT中抽一个伪目标，其他源构建同族记忆；伪目标部分结果作support，其余作query监督。

留出新的物理context只检验context泛化，不能替代“未见目标SUT”验证。一次训练中重复抽取同一少数SUT数千次，不等于拥有数千个独立目标。

### 9.2 与 B=50 对齐的support长度

训练长度从 \(0\) 到 \(49\) 采样，在0/1/3/5/10/20/30/49附近均有覆盖。不能仍只训练0–10然后声称已经验证了50步适应。

support不强制包含两类标签。允许空support、全pass、全fail；否则训练时会免费获得实际测试未必拥有的失败种子。

初期随机选support，后期混入冻结历史选例策略或旧模型的离线前缀，缩小主动选择造成的分布差异；这些都只使用训练伪目标银行，不增加目标仿真，也不读取最终目标结果。

### 9.3 损失与校准边界

\[
L=\frac1E\sum_e\left(L_{BCE,e}+\lambda_{rank}L_{rank,e}\right),
\]

\[
L_{rank}=\frac1{|\mathcal P|}\sum_{(x^+,x^-)}softplus(-(s(x^+)-s(x^-))).
\]

采用自然query分布上的BCE，另采failure/pass对计算ranking loss。若使用重采样BCE，应显式采用采样权重修正或承认输出只是ranking score。单类query没有有效ranking对时该项为0。

\(\lambda_{rank}=0.5\) 是开发起点。BCE本身也可以用于排序，增加ranking loss是待检验设计，不声称BCE“不能”服务于D@B。

### 9.4 学会不盲信历史

训练只在训练数据中加入真实可构造的困难episode：同族不同源差异、pass-only memory、缺少通过对照、稀疏support。可做历史基元dropout，但不制造假的目标标签。

训练优化的是少样本目标排序；不能直接给门控填入最终“源和目标相同／不同”的真值作为输入。

---

## 10. 组件三：面向 Sobol 候选池的 DFE

### 10.1 不再依赖二维grid_index

当前 V3 按维数取 1,024／2,048／4,096 个 scrambled Sobol 点。3–5 维中一般不存在规则四邻格点，连续射线也未必穿过下一个候选。[U2]

因此DFE是**对已有有限候选的局部排序**，不是无预算地新增连续场景。只在同family、兼容固定context和类别条件内计算几何。

### 10.2 距离与方向

将连续活动参数归一化后：

\[
d_f(x,z)=\sqrt{\frac1{d_f}\sum_j(\tilde x_j-\tilde z_j)^2}.
\]

按维数归一化改善量纲比较，但不消除高维距离集中问题；尺度在开发确定，不能声称任意维度效果不变。

局部尺度取未使用标签的候选几何第4近邻距离的中位数 \(h_f\)。每个真实target failure seed先提出其8个最近未查询候选，并限制在开发冻结的最大局部半径；无候选就让该seed休眠。基准起点为 \(\sigma_f=h_f\)、半径 \(2h_f\)。这是可验证的初始规则，不是最优公式。

方向只对合法连续参数子空间计算：

\[
v(x;s)=(\tilde x-\tilde s)/\|\tilde x-\tilde s\|_2.
\]

相同点在冻结时去重；不得对零向量算cosine。

### 10.3 方向多样性与局部收益

\[
N(x;s)=\begin{cases}
1,&\mathcal V_s=\varnothing,\\
\min_{v_j\in\mathcal V_s}\frac{1-v(x;s)^Tv_j}{2},&\text{其他情况}.
\end{cases}
\]

\[
S_L(x;s)=p_*(x)\exp(-d_f(x,s)^2/(2\sigma_f^2))
(1+\eta N(x;s))\,b_{pass}(x;s).
\]

\(\eta=0.25\) 是开发起点，避免多样性完全压过失效概率。多seed取最大值并保存origin seed。

目标pass出现在某角锥及距离时，只对该局部方向的更远candidate降权，开发起始乘0.25；**不能永久删除这些候选或宣布该方向全部安全**，因为可能存在另一段失效区域。Global仍可选择这些点。

每个target failure均保留原始证据，最多50个seed，首版无须引入昂贵动态聚类。若为减小重复提案做代表点压缩，不能覆盖原始发现，也不能用簇数代替D@B。

### 10.4 与IFR的区别

借用失败出发与方向多样性；不免费给目标种子；不继承凸性；不精确二分；不拿未执行点增加发现数。[R1]

DFE可能提高局部命中，也可能因多样性和pass探测降低命中；保留 `FM²-NoDFE` 作为同表对照。若无增益，不强行保留。

---

## 11. 两个arms为什么能够自适应？它们又做不到什么？

### 11.1 两个arms就是两种“下一例怎么选”的办法

- G：从整个family剩余池中，选模型评分最高的候选。
- L：围绕已经确认的目标失败，从DFE候选中选一个。

路由不直接控制车，不预测事故根因，也不选择八个family之间的预算。它只学习：**当前会话里，哪种选例办法更容易让一次收费执行变成一个真实failure发现。**

### 11.2 普通 Beta-Bernoulli TS 的直观例子

初始两边都是Beta(1,1)。忽略折扣时：

- G尝试10次，5次发现：Beta(6,6)，均值0.50；
- L尝试5次，4次发现：Beta(5,2)，均值约0.714。

于是L更常被选中，但不是永远被选。TS每次从两个分布各抽一个数，选更大者，未充分测试的分支仍有机会。

这个例子只演示调度原理，不是本项目实验结果。

### 11.3 为什么本项目需要折扣？

候选会耗尽、局部区域可能被测完，网络预测也在随support改变。两臂成功率并不固定。

采用轻量的折扣计数作为默认**开发启发式**：

\[
A_a\leftarrow\gamma A_a,\quad C_a\leftarrow\gamma C_a,\qquad a\in\{G,L\};
\]

\[
\theta_a\sim Beta(1+A_a,1+C_a).
\]

执行所选分支 \(a_t\) 后：

\[
A_{a_t}\leftarrow A_{a_t}+r_t,\quad
C_{a_t}\leftarrow C_{a_t}+1-r_t,
\]

\[
r_t=\mathbf1[\text{该次目标执行确认有效failure}].
\]

\(\gamma=0.95\) 是开发起点；\(\gamma=1\) 回到普通TS。折扣思想有已有研究依据，[R9] 但本文不声称直接复现其中的算法或继承其理论界。

### 11.4 无法判定、相同候选、空局部分支

- **无法判定执行**：模型support不加假PASS/FAIL；本次未产生确认发现，router收益为0。与V7.1“完全不更新router”的规则不同，这里按“每次收费执行的发现产出”对齐奖励。
- **G和L提出同一候选**：只执行一次，只给被选分支记收益；记录proposal overlap，不伪造两次奖励。
- **无target failure或无合法local候选**：L不可用，只选G；不凭空给L成功或失败。
- **无法判定反复出现**：保留原因，先检查执行器，不让router掩盖基础错误。

### 11.5 不存在自动成功保证

只有两个arms不意味着它们一定有效。若G的全局风险排名错误、L没有有价值的邻域，router不能凭空找到failure。它不保证在50步内识别最佳分支，也不能给出平稳独立bandit的最优性结论。

G仍然是风险排序，不等于保证覆盖全空间。只用maximin打破同分也不构成全面探索保证；孤立且历史未提示的失效仍可能漏检。这是适用边界，不通过增加隐藏免费测试来解决。

---

## 12. 完整会话算法与张量接口

```text
run_session(family f, target S*, authorized_history Hf, pool Xf, B=50):
    verify schema / validity policy / checkpoint / source permissions
    M = build_and_encode_historical_primitives(Hf)
    Q = encode_candidates(Xf)                  # candidate-only inputs
    pH0 = model(Q, M, support=empty)            # cached historical prediction
    support = []
    target_failure_seeds = []
    queried = set()
    router_counts = zero

    for t in 1..min(B, number_of_candidates):
        p = evidence_conditioned_model(Q, M, support)
        xG = highest_score_unqueried_candidate(p)
        xL, origin_seed = DFE.propose(p, target_failure_seeds, queried)
        arm = discounted_TS(available={G, L if xL exists})
        x = chosen_candidate(arm)
        prequery = freeze_prediction_and_retrieval_log(x)
        outcome = oracle.query(x)              # exactly one charged execution
        queried.add(x)

        reward = 1 if outcome is valid FAIL else 0
        router.update(chosen_arm=arm, reward=reward)
        if outcome is valid PASS or valid FAIL:
            residual = outcome.label - pH0[x]
            support.append(encode_real_outcome(x, outcome, residual))
        if outcome is valid FAIL:
            target_failure_seeds.add(x)
        DFE.observe_selected_proposal_only(x, outcome, origin_seed)
        save_query_ledger(prequery, outcome, arm)

    return actually_confirmed_target_failures
```

### 12.1 主要张量形状

| 对象 | 形状示例 |
|---|---|
| 一个3D场景的活动参数tokens | `[3, 128]`，另加context tokens |
| 一个5D场景的活动参数tokens | `[5, 128]`，另加context tokens |
| 本族候选的场景表示 | `[N_f, 128]`，`N_f` 为 1,024、2,048 或 4,096 |
| 本族K张历史基元 | `[K, 128]`，K≤64为开发起点 |
| t个已查询有效support | `[t, 128]`，t≤50 |
| 一轮候选到历史权重 | `[N_remaining, K+null]`（每头可另存） |
| 目标最终分数 | `[N_remaining]` |

所有cache都由冻结权重和已授权数据生成；线上dropout关闭，不在候选间偷偷传递标签或让batch顺序改变语义。

---

## 13. B=50 与最新 V3 场景附件如何对齐

附件的已接线开发族是：**S01、S02、S03、S04、S05、S06、S08、S09**，不是S01–S08；其中S07仍是候选。[U2]

| Family | 活动维数 | 开发候选数 | 单会话目标预算 |
|---|---:|---:|---:|
| S01 | 4 | 2,048 | 50 |
| S02 | 4 | 2,048 | 50 |
| S03 | 4 | 2,048 | 50 |
| S04 | 5 | 4,096 | 50 |
| S05 | 4 | 2,048 | 50 |
| S06 | 3 | 1,024 | 50 |
| S08 | 5 | 4,096 | 50 |
| S09 | 4 | 2,048 | 50 |
| 合计 | 不相加为一个空间 | 19,456 | 400 |

这是对本地开发候选计划的数量对齐，不表示 19,456 个候选已经取得目标结果。

每族 50 次查询分别占 3／4／5 维候选池的 4.88%／2.44%／1.22%；D@5/10/20 仍用于观察早期收益。

全库测量每个目标最多 19,456 个执行条目（按每场景一个指定随机实现计算）；在线部署模拟每种方法最多 400 次查询。完整响应库已有时，多方法回放不增加物理执行，但逻辑测试成本仍按每次揭示计费。

若采用每场景R个随机重复，必须明确预算单位是单次rollout还是固定R次组成的scene-test；主协议建议rollout为单位。不能把50个scene-tests×R次物理执行仍称为总共50次测试。

---

## 14. 两类主实验与最小报告

### 14.1 真值比较

对每个family、目标系统的冻结有限池完整测量；封存目标响应，选择器逐条查询。

主指标：D@50；同时报告D@5、D@10、D@20、D@30、Recall@50和HitRate@50。截点来自同一条50次会话前缀，不是分别调参的六次实验。

\[
Recall_f(B)=D_f(B)/|\mathcal F_{*,f}|,
\quad HitRate_f(B)=D_f(B)/B.
\]

实际只执行 \(B_{eff}<B\) 时，用 \(B_{eff}\) 报实际命中率并注明池耗尽。真值failure数为0时Recall=NA，任务保留。含UNKNOWN的银行不能宣称拥有完整确定真值，须报告未知数并按事先规则处理，不能根据目标标签从候选池删除它们。

Oracle发现数上限为 \(\min(B,|\mathcal F_{*,f}|)\)。它只用于解释上限，不作为可部署方法。

### 14.2 同预算基线比较

沿用V7.1的核心比较：Random、FailureDistance、HistoryRank、原FBRT-v3目标失效适配版、IFR-DSB适配、FM²-NoDFE、FM²完整方法。

单family时HistoryRank-UCB的跨模板UCB退化，应如实注明；原FBRT-v3不能被默默修改成新网络。IFR没有免费种子，内部每次执行都计费。

`FM²-NoDFE` 只能隔离整个DFE＋router包，不足以分别证明记忆语义、attention、证据门控均有效。若不增设相应同表消融，就将具体组件效果写成机制解释与候选贡献，而非分别被证明的结论。

### 14.3 主表

| Family | d | Target | N | GT failures | Method | D5 | D10 | D20 | D30 | D50 | Recall50 | HitRate50 |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|

逐family展示累计发现曲线；macro-average不替代逐项结果。八族、相邻场景和重复选择器seed不是同分布独立样本，不能据行数虚增显著性。

训练时间、峰值内存、每查询预测耗时及测试总成本另放简短成本表，不把模型计算称为免费。

---

## 15. 场景准入与未知结果处理

保留最新附件的范围，不因本方案更改数值：S07/S10/S11/S12为待实现候选，S13/S14延后；IA/IB按附件退役，不作为新增数据来源。[U2]

- S05/S09需要真实的横向决策/避让能力，固定车道NL不能因为能执行脚本就自动适用。
- S02的名义TTC是目标布置参数，不是目标rollout的实测TTC，更不是已验证视觉遮挡。
- S04/S09只统计碰撞时，不声称验证完整恢复／再起步功能。功能事件可以做诊断，不在事后偷偷改失败终点。
- 提前发生的有效ego碰撞保留为failure，不能因为随后事件没执行完而排除。
- 非碰撞时事件缺失是否有效，应按测试意图提前定义，区分初始化错误和ego合法提前避险。
- 初始重叠、单位不一致、活动轴没有作用路径等在开发阶段修复；有效但不利的结果保留。

按维数设置的 1,024／2,048／4,096 点是开发计算预算，不证明覆盖连续空间；不使用 11^d 全笛卡尔积，不把 SUT 不兼容的任务强行塞进八族平均。

---

## 16. 代码实施清单

| 文件／模块 | 工作 |
|---|---|
| `fm2_schema.py` | 读取本地已冻结V3 schema；名字/单位/角色/类型规范化；变维mask；拒绝静默降维 |
| `fm2_memory.py` | 从授权历史构建证据基元；同源通过对照；pass-only/unknown；来源均衡 |
| `fm2_encoder.py` | PLE、参数SAB/PMA、见证编码、key/value分支 |
| `fm2_model.py` | 历史读取、support读取、证据条件化门控、TabM式轻量融合头 |
| `fm2_train.py` | 外层target组留出，内层episodic训练，support0–49，BCE＋ranking |
| `fm2_dfe.py` | Sobol池kNN、连续子空间方向、局部降权、seed与提案日志 |
| `fm2_router.py` | 可用臂检查、折扣TS、单次执行收益与重叠候选处理 |
| `fm2_selector.py` | 单family会话B50、模型输出与两个proposal、唯一oracle入口 |
| `fm2_evaluate.py` | 全库封存、前缀D/Recall/HitRate、每family结果与成本 |

保留原 `selector.py`、`bayes_model.py` 和旧实验身份作为基线。`new_failure_card()` 若依赖旧二维dictionary，不能宣称直接复用即可；可以保留其日志结构而给FM²实现新的schema-aware观察登记。

### 16.1 必须执行的工程测试

这些是单元测试，不是新增论文实验：

- 同一场景重排参数，`eval()`输出数值等价。
- 相同数字交换name/actor后应能够表达不同语义。
- 3D/4D/5D不同长度正向传播，mask padding不改变结果。
- 空历史、历史全通过、空support及单类support不产生NaN。
- 冻结目标及伪目标结果不出现在任何历史汇总、归一化统计或记忆筛选中。
- 前query分数和基元权重在揭示答案前保存。
- 标签翻转只影响已公开support后的预测，不回写旧日志。
- Sobol邻域不依赖grid_index；类别维数不进入方向cosine。
- 所有无法判定执行计费；不把它训练成PASS。
- G/L同提案只执行一次；不为未执行arm奖励。
- 单会话最多50，八个独立会话最多400；新会话清空目标状态。
- 本地runner/YAML/model/checkpoint/hash与manifest绑定。

### 16.2 实施次序

先用当前已开放银行做schema与信息审计，完成最小记忆／support网络；然后加入DFE与router并只做开发选择。物理契约未确认前，不扩大SUT或追加大规模PPO训练。

神经网络有无增益不能由本设计预先保证；弱于FBRT时保留负结果。组件二的纠错、组件三的局部命中应分别有日志证据，不能仅以总分解释所有机制。

---

## 17. 本次对旧措辞的修复

| 原措辞风险 | 本次准确表述 |
|---|---|
| “与维度无关” | 接受变长参数集合，固定token宽度；不保证未见语义/高维效果 |
| “padding不好” | 允许mask padding；反对无语义、无mask的拼零列 |
| “self-attention置换不变” | self-attention等变，PMA/读取后不变 |
| “机制＝事故原因” | 可观察失效模式，因果解释需另有证据 |
| “gate＝历史可靠概率” | 可学习检索/融合权重，不是校准概率 |
| “第二步一定纠正历史” | 训练目标如此，行为与性能需验证 |
| “Thompson自动找到最佳搜索策略” | 根据已观察yield自适应；非平稳相关arms无直接保证 |
| “pass意味着射线外侧都安全” | 仅局部降权，不删候选、不推断未测标签 |
| “新场景参数表意味着模型已支持” | 必须核实schema、runner、encoder、baselines全部接通 |
| “B50仍训练support≤10即可” | support训练覆盖0–49，最终输入分布匹配 |
| “一个新context就是未见新SUT” | 泛化目标不同，不能互相替代 |

---

## 18. 论文贡献如何保持聚焦

候选贡献不是发明Transformer、TabM或Thompson Sampling，而是：

1. **带通过对照及来源约束的历史失效模式基元**：适配变维交通参数且保存可验证证据。
2. **目标证据条件化的历史检索**：少量目标通过／失败不仅改变最终分数，也改变哪些历史经验被读取。
3. **同一收费预算下的全局检索与目标失败局部挖掘**：适配有限Sobol候选，不依赖免费种子或完整边界重建。

它们是待验证贡献。一个对比表的总分不足以分别证明所有新模块优越；没有独立增量时，应缩小主张，而非更改失效定义或删除不利场景。

---

## 19. 来源与可复用实现

以下分清用户材料、代码实况与外部研究。文档中标为“开发起点”的数值由本方案提出，不是引文给出的最优配置。

### 用户材料

- [U1] `FM2_FBT_Research_Plan(1).md`，V7.1，用户上传。本文继承其三组件、per-family任务和目标失效发现目标。
- [U2] `FBRT_Scenario_Parameter_Space_V3.md`，用户提供的场景空间方案。14 族范围、八个已接线声明及 IA/IB 退役沿用其场景设计；本地开发采样量现按维数扩充，以 `methods/failure_memory_regression/configs/scenario_parameter_space.yaml` 为准。

### 当前项目代码

- [C1] [pattern_memory.py，固定快照](https://github.com/SafeDL/META_LEARNING/blob/68db2f767ef1b5495072ecaa94c5d48faea0a6e7/methods/failure_memory_regression/pattern_memory.py)：`active_values`、`semantic_key`、`build_pattern_cards`、`RBFDictionary`。
- [C2] [selector.py，固定快照](https://github.com/SafeDL/META_LEARNING/blob/68db2f767ef1b5495072ecaa94c5d48faea0a6e7/methods/failure_memory_regression/selector.py)：oracle、观测、历史权重及新目标失败记录。
- [C3] [replay_utils.py，固定快照](https://github.com/SafeDL/META_LEARNING/blob/68db2f767ef1b5495072ecaa94c5d48faea0a6e7/methods/failure_memory_regression/replay_utils.py)：有效结果与任务奖励。
- 早期远端快照没有包含场景参数配置；现有本地 `methods/failure_memory_regression/configs/scenario_parameter_space.yaml` 与生成器已经核对。远端快照不代表当前工作区状态。

### 外部研究与代码

- [R1] Huang等，*Identification of Failure Regions for Programs with Numeric Inputs*。[论文](https://arxiv.org/abs/2007.15231)，[作者源码IFR](https://github.com/huangrubing/IFR)。源码以RAR归档；FSB/DSB思想需适配预算和有限池，仓库可读取不等于获得任意重许可权。
- [R2] Wang等，*Dance of the ADS: Orchestrating Failures through Historically-Informed Scenario Fuzzing*，ISSTA 2024。[DOI](https://doi.org/10.1145/3650212.3680344)。本轮阅读用户上传全文中的方法摘要与历史筛选/事故轨迹聚类段落；未复现其CARLA系统。
- [R3] Lee等，*Set Transformer*，ICML 2019。[论文](https://proceedings.mlr.press/v97/lee19d.html)，[作者代码modules.py](https://github.com/juho-lee/set_transformer/blob/master/modules.py)。本轮直接读取 `MAB`、`SAB`、`ISAB`、`PMA`；原实现需添加本项目mask、角色语义与现代Pre-LN适配，不宣称原封不动复现。
- [R4] Kim等，*Attentive Neural Processes*，ICLR 2019。[论文](https://arxiv.org/abs/1901.05761)。借鉴query读取相关support；不继承其回归性能保证。
- [R5] Gorishniy等，*On Embeddings for Numerical Features in Tabular Deep Learning*，NeurIPS 2022。[论文](https://proceedings.nips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html)，[作者代码](https://github.com/yandex-research/rtdl-num-embeddings)。
- [R6] Gorishniy等，*TabR: Tabular Deep Learning Meets Nearest Neighbors*，ICLR 2024。[作者介绍](https://research.yandex.com/publications/tabr-tabular-deep-learning-meets-nearest-neighbors)，[作者核心代码](https://github.com/yandex-research/tabular-dl-tabr/blob/main/bin/tabr.py)。本轮读取了排除自检索、相似性、`label_encoder`、query-context差值及value聚合实现。
- [R7] Gorishniy等，*TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling*，ICLR 2025。[会议论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html)，[作者代码与自定义输入说明](https://github.com/yandex-research/tabm)。本方案采用参数高效head思路，不把通用基准性能搬成驾驶测试结果。
- [R8] Russo等，*A Tutorial on Thompson Sampling*。[论文](https://arxiv.org/abs/1707.02038)。
- [R9] Qi等，*Discounted Thompson Sampling for Non-Stationary Bandit Problems*。[论文](https://arxiv.org/abs/2305.10718)。该文研究的分布／分析条件不同；本方案的折扣Beta路由只借鉴遗忘思想。

复用代码前锁定commit、依赖和许可证，保留归属声明。本文没有分发上述第三方源码或声称已经运行其完整实验。

---

## 20. 最终统一口径

> 在highway-env中，将不同参数维度的功能场景分别作为有限候选池；用具备同源通过对照与可观察行为描述的历史失效基元冷启动；用目标真实反馈改变历史检索和候选排序；发现目标失败后在Sobol候选邻域进行方向多样化挖掘；每族全部过程最多50次目标执行，以真实失效发现数评价。

**本次没有恢复单调版本演化、双向奖励、精确边界识别或TabPFN路线。** 目标明确、预算可审计、组件有来源、有实施接口，也保留未验证和可能失败的边界。
