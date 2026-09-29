# FBRT 统一研究方案 V6.1：highway-env 中的分阶段双向版本变化测试

> GPU环境在系统的： conda activate metadrive
> 原始方案快照的现有实现为 `FBRT-Memory-Exploit-v3`；当时的双向模式、持续更新训练接口及边界差异模型均待实现、待验证。
> 原始方案快照仅核对历史方案、项目源码和官方接口文档，尚未执行新增驾驶训练、物理仿真或完整回放实验。
> **后续实施进度（2026-09-27）**：已实现 NL-IDM 父子 release chain、真实 PPO 权重继承 V0→V1→V2、完整配对物理银行、双向预算回放与族级统计。前三个 NL 确认集和首个 PPO 确认集暴露了原方法的局限，负面结果均保留。第四个 NL 确认集上 margin/frontier 方法在 20 次方向查询内找到 3/3 回退、13/17 改善，优于静态风险的 0/3、6/17；但真值全在 S08，族级差异不显著。随后依据旧银行开发的角色门控法，在全新上下文的 NL 确认中找到 2/2 回退、2/2 改善，在 PPO 确认中找到 V0→V1 的 7/15 回退、3/4 改善及 V1→V2 的 2/6 改善。**这些结果仍不足以证明本文算法整体优越**：普通坐标消融在新确认中与角色门控法持平，PPO 首段静态角色覆盖找到更多回退（14/15），NL 的四个变化仍全在 S08。另五族开发调查仅在 S04 出现 NL 回退，PPO 没有版本翻转；不得作为优越性证据。代码与完整记录见 `methods/failure_memory_regression/bidirectional.py`、`results/method_chains/failure_memory_regression/nl_release/`、`results/method_chains/failure_memory_regression/ppo_release/`、`results/method_chains/failure_memory_regression/cross_sut_confirmation/` 和 `results/method_chains/failure_memory_regression/family_survey/`。
> **四族补充确认**：为检验多区域前沿修订，又在预先冻结的新上下文中完整执行 NL 与 PPO 各 200 个场景、600 次物理仿真。NL 首段回退主方法发现 17/35，优于静态风险 10/35，但与静态角色覆盖及同采集坐标消融均持平；PPO 首段回退发现 5/5、第二段改善 4/4，却漏掉首段 2/2 改善，而静态角色覆盖找到这两个改善。同采集坐标消融在 PPO 全部任务与主方法持平。**因此仍不能诚实地宣称本文算法具有普遍、显著或由有向特征带来的整体优势。**全部正负结果见 `results/method_chains/failure_memory_regression/family_confirmation_v1/`。
> **有向特征诊断**：在另一个冻结的四族×三上下文×7×7 物理银行中，保留角色门控区的有向边界特征后，NL 和 PPO 的发现数均与同采集坐标消融持平；PPO 部分任务的早期面积还更低。该结果否定了“角色门控版已体现有向特征增益”的解释。此前多轮固定余量探测、前沿和角色门控属于探索变体，不能替代本文首版协议。随后按第 2、3、7 节冻结并完成纯共享后验确认：同一目标风险后验直接排序回退与改善，比较 Random、静态风险、静态边界、中心残差、普通坐标、目标响应模型及有向残差，使用 3 族×3 新上下文×11×11 网格；完整结果见 `results/method_chains/failure_memory_regression/core_confirmation/`。核心结果不支持本文首版方法相对坐标消融或静态风险具有一致优越性：NL 的 16 个回退主方法发现为 0，25 个改善发现 13；PPO 的两个改善任务分别发现 2/26、0/20，而相应静态风险为 2/26、0/20。PPO 第二段 `target_only` 达到 3/20 并找到唯一的一个回退，提示父风险 offset 可能负迁移。负面结果保留，不据此宣称方法胜出。
> **第二阶段确认（2026-09-28）**：NL 与 PPO 各完成 1,089 个全新上下文场景、3,267 次配对物理执行。弱化局部截距先验的 UCB 在 NL V1→V2 与 PPO V1→V2 改善方向上优于静态风险和普通坐标残差，但两项均与坐标上下文 UCB 消融持平；它漏掉全部 NL 回归，也未优于非空回归任务上的 target-only。结果支持部分任务中的上下文适应信号，不支持广泛优越性，也不支持有向边增益。细节见 `results/method_chains/failure_memory_regression/contextual_confirmation/findings.md`。
>
> **第三阶段确认（2026-09-28）**：NL/PPO 各完成 1,089 个场景、3,267 次配对物理执行。斜率校准主方法在 PPO V1→V2 回归方向找到 11/21，优于静态风险 8 和普通坐标残差 7，但低于坐标校准消融 13；它漏掉全部 51 个 NL 回归，在 PPO V0→V1 改善方向也低于 target-only（6/46 对 11/46）。阶段三未支持稳定优越性或有向边增益，见 `results/method_chains/failure_memory_regression/offset_calibration_confirmation/findings.md`。
>
> **第四阶段确认（2026-09-28）**：NL/PPO 各完成 1,089 个新场景、3,267 次配对执行。上下文启动将 PPO V0→V1 回归发现提高到 9/10（坐标启动版 10/10），将 PPO V1→V2 回归提高到 10/17；但 NL V1→V2 改善发现为 15/73，低于无启动校准版 20，PPO V1→V2 改善为 10/47，低于 16–17。结果支持“启动有利于回归但牺牲改善查询”的方向不对称，不支持一致双向优越性。详情见 `results/method_chains/failure_memory_regression/context_bootstrap_confirmation/findings.md`。
>
> Stage five confirmation (2026-09-28): NL and PPO each completed 1,089 scenes and 3,267 paired physical executions. Regression-only context bootstrap did not yield a consistent bidirectional advantage: NL regression recall rose over unbootstrapped calibration, but the method lost to role-gated or directed-calibration controls on other transitions. All Holm-adjusted p-values were 1.0 across three family clusters. Full counts: results/method_chains/failure_memory_regression/regression_bootstrap_confirmation/findings.md.
> Stage six replication (2026-09-28): the same predeclared selector and controls were frozen on three fresh fixed contexts per supported family. NL/PPO source, manifest, protocol, and checkpoint hash audits passed; both 3,267-episode physical banks completed with zero new executions during replay. The primary found 45 changes at D@20 across six active directions, below coordinate role-gated search's 54; it won only on PPO V1→V2 regression among those direct comparisons. Three-family tests are descriptive and all Holm-adjusted p-values are 1.0. Post-confirmation exploratory screening found no clear aggregate edge-feature gain. This replication does not establish superiority; see `results/method_chains/failure_memory_regression/superiority_replication_confirmation/findings.md`.
> **Stage seven candidate confirmation (2026-09-28)**: exploratory stage-five/six replays nominated `coordinate_role_gated` for fresh-family testing; that selection is post hoc and the prior banks are development evidence only. The protocol froze this candidate against static, coordinate, directed-bootstrap, and directed-edge role-gated controls across eight executable families × three new contexts × 11×11 scenes for both NL and PPO. Each chain requires 8,712 physical episodes. Hash and overlap audits passed. By explicit user request, physical measurement is paused before evaluation: NL has 4,153 valid rows (2,904 V0, 1,249 V1), PPO 3,951 (2,904 V0, 1,047 V1); neither has V2 results or selector comparisons. Resuming the frozen measurement command reuses cached episodes.

> **场景参数空间修订**：此前准备的单上下文 11×11 二维开发清单（8 族、968 个具体场景）保留为原方案记录，不据此声称各族本来只有两个有效维度。下一轮开发采用按场景机理定义的 3–5 维可执行空间，按维数取 1,024／2,048／4,096 个 Sobol 候选，8 族共 19,456 个，尚无物理执行。其余 6 族也逐一定义了候选轴，但在执行器支持前不进入物理银行。完整设计、取值依据与限制见 `docs/FBRT_Scenario_Parameter_Space_V3.md`；此前冻结且部分测量的三上下文阶段七清单保持原有记录。
> **覆盖关系**：本文件是后续研究的统一执行依据，并覆盖 V6 中“规则控制器仅作为机制对照”的安排。V4/V5 仅作为历史记录。V5 的 CARLA/TransFuser 路线、仅将 Unsafe→Safe 用于事后统计的限制，以及与之对应的成本计划，均不再执行。旧实验结果不得改写为新方案结果。

## 0. 固定六项决策

| 事项 | 固定结论 |
|---|---|
| 环境 | 只使用 highway-env 与项目现有 `highway_sim_env` 扩展；不切换 CARLA、MetaDrive 或其他模拟器 |
| 研究问题 | 有限目标查询预算下，同时主动发现 Safe→Unsafe 的回退与 Unsafe→Safe 的改善 |
| 方法性质 | 历史引导的自适应版本测试；不是驾驶策略训练算法，也不是整体安全认证或事故率估计 |
| 评估真值 | 对预先冻结的有限场景库，完整执行各版本，形成版本×场景×指定随机实现的配对结果矩阵 |
| 被测版本演化 | **先使用同一非学习型 IDM 控制器构造可解释的 V0→V1→V2 软件 release chain，完整跑通双向协议；再使用现有 SB3 PPO 的真实权重继承链验证学习型更新。** highway-env 灰度图+CnnPolicy 仅作为后续输入模态补充 |
| 技术规模 | 一个历史边界表示、一个新版本响应预测模型、一个双池预算调度；PBTR、价值前瞻、深核、元强化学习等不列为首版必需组件 |

**一句话动机**：自动驾驶软件无论通过控制规则/参数修改还是模型权重持续学习进行更新，都可能同时带来局部改善与局部回退；完整重测代价高，因此利用旧版本测试经验和少量新版本反馈，尽早找到“哪里变差、哪里变好”。

**论文建议标题**：*History-Guided Bidirectional Change Discovery for Continually Updated Autonomous Driving Policies*。

中文：**面向持续更新自动驾驶策略的历史引导双向变化测试**。

“安全边界”仅表示给定场景参数和判定规则下的局部通过—失败关系；不是可达性分析得到的形式化安全边界，也不预设所有变化都发生在旧边界附近。

---

## 1. Motivation 与问题定位

### 1.1 完整 motivation

自动驾驶决策软件会经历连续版本演化：早期版本常通过控制规则、阈值和标定参数迭代，学习型版本还会通过新增数据继续更新模型权重。一次更新可能改善某些场景中的表现，也可能破坏以前已经建立的能力。只比较平均奖励，可能掩盖局部回退；只重新执行过去失败的用例，又可能漏掉原来通过的场景发生的退化。因此，版本验证需要同时了解“哪里变差”和“哪里变好”。

然而，每个版本完整重测大量场景需要重复投入。旧版本已经积累了场景级通过、失败及轨迹信息，其中局部失败—通过结构可为新版本测试提供线索；这些线索又可能因更新而失效。本研究关注：如何利用历史局部结构，并根据少量目标反馈修正预测，在有限预算内更早发现回退与改善，而不是从头摸索，或无条件相信旧经验。

上述文字是研究动机和待验证假设，不是新算法已经胜出的实验结论。

### 1.2 两类算法必须分开

- **被测策略 SUT**：PPO 等驾驶策略。持续学习发生在 SUT 的版本之间。一次测试会话期间，其权重、预处理、动作与环境契约固定。
- **本文测试方法**：选择待测场景，根据结果更新场景响应预测。它不输出驾驶动作，不在测试过程中修复 SUT。

PPO 使用强化学习，不等于测试方法是元强化学习。没有跨任务的真实元训练，就不能声称获得了通用元策略。

### 1.3 “发现”而不是“证明”

主问题为 **budget-constrained bidirectional change discovery**，不是整个 ODD 的风险估计。

Safe/Unsafe 是通俗缩写。正式使用 PASS/FAIL，限定于预先注册的终点和执行条件。原始记录保留 `ego_collision`、执行完成、越界、进度、超时、碰撞阶段等字段。碰撞终点上的通过不等于全部驾驶要求满足。

发现 Unsafe→Safe，表示该场景的观察结果改善；只有关联开发者声明的缺陷及检查范围，才构成相应的修复确认证据。发现很多改善不能抵消少量严重回退，更不能说明全部旧问题修复。

### 1.4 当前结果的地位

现有 `task_reward` 只支持 `regression` 与 `cross_agent`；前者奖励父版本通过且目标碰撞，后者奖励目标碰撞。当前没有改善主动发现分支。[C02]

当前 V3 在旧单向冻结结果中有预算 20 的方向性增益；它不是双向测试或持续学习版本更新的验证结果。原报告的累计发现数、选择器重复和有限物理上下文不能当成独立软件缺陷数或新增独立样本。

---

## 2. 正式问题定义：一个场景响应模型，两种变化

### 2.1 场景集与响应

在固定 highway-env 执行契约下，冻结有限具体场景集：

$$
\mathcal T=\{x_1,\ldots,x_N\}.
$$

每个 x 包含功能场景、物理上下文、参数值和明确的随机实现标识。各版本对同一组 x 执行。对主要终点定义：

$$
y_v(x)\in\{0,1,\bot\},
$$

0：该终点下的有效通过；1：有效失败；\(\bot\)：基础设施错误、缺失或不能判定。有效碰撞提前结束不是无效执行。碰撞终点下无碰撞但长期停车的结果，要同时报告功能状态，不能冒充完整修复。

共同有效性的实现必须防止泄漏：候选资格只用预先定义的信息和已知父版本结果；不能看完目标结果后，为选择器事先删去“目标执行异常”条目。目标异常在查询时揭示，照样计费，随后按统一规则处理。评估端另报完整性及异常分布。

### 2.2 四种变化

对父版本 p、目标版本 t，且两者结果确定时：

$$
\mathcal R=\{x:y_p(x)=0,y_t(x)=1\},\quad
\mathcal I=\{x:y_p(x)=1,y_t(x)=0\},
$$

$$
\mathcal P=\{x:y_p(x)=1,y_t(x)=1\},\quad
\mathcal S=\{x:y_p(x)=0,y_t(x)=0\}.
$$

分别为回退、改善、持续失败、通过保持。\(\mathcal R\) 与 \(\mathcal I\) 都进入主动测试目标；\(\mathcal P\) 与 \(\mathcal S\) 不是坏数据，而是模型需要学习的无翻转响应。

### 2.3 查询集合与目标

候选按已知父版本结果拆成：

$$
\mathcal T_0=\{x:y_p(x)=0\},\quad
\mathcal T_1=\{x:y_p(x)=1\}.
$$

一次会话的总查询集合为 \(Q_B\)，无放回查询，并分别记录：

$$
D_R(B)=|Q_B\cap\mathcal R|,\qquad D_I(B)=|Q_B\cap\mathcal I|.
$$

**不以一个未区分方向的总翻转数作为唯一目标。** 主协议分别比较两方向效率；回退与改善都必须单独报告，防止较容易发现的改善掩盖回退漏检。

### 2.4 一个后验，而不是两套互不相关的模型

设当前目标版本失败预测为 \(p_t(x)\)。变化分数为：

$$
s_R(x)=p_t(x),\ x\in\mathcal T_0;\qquad
s_I(x)=1-p_t(x),\ x\in\mathcal T_1.
$$

这两个分数源于同一个目标响应模型。每次实际查询将 **目标结果 \(y_t\)** 用于模型更新；不能将“发现变化=1”的奖励当成目标失败标签，尤其改善场景的目标标签应是 0。

分数互补只是问题定义的自然结果，不单独作为创新。

---

## 3. 最小技术框架：历史结构→局部适应→双向选例

### 3.1 历史局部通过—失败结构

利用父版本完整结果建立同版本、同场景族、同物理上下文的局部对比片段。对一个失败—通过对 \((z_f,z_p)\)，在冻结归一化坐标中定义：

$$
d_e=\|z_p-z_f\|_2,\quad n_e=(z_p-z_f)/d_e,\quad m_e=(z_f+z_p)/2,
$$

$$
u_e(z)=n_e^\top(z-m_e).
$$

加入局部支持特征 \(k_e(z)\) 和有向特征 \(k_e(z)u_e(z)\)。每上下文先限制少量片段，例如最多 4 个；数量、半径在开发集冻结。不能用不同版本的一个失败和一个通过伪称同版本边界。

\(n_e\) 只是方向线索；中点不是真实边界，片段外不作强外推。网格复杂区域可以拆片段，不能假定单一凸区域。

当前源码已有历史模式和附近通过记录基础；新增价值是检验方向信息是否比中心距离更有用。[C01][L02]

### 3.2 一个低维局部版本差异模型

首版采用与现有概率代码相容的带 offset 贝叶斯逻辑回归：

$$
\ell_t(x)=g_p(x)+\phi(x)^\top\delta_t,\qquad
p_t(x)=\mathbb E[\sigma(\ell_t(x))\mid H_t].
$$

- \(g_p\)：只由授权历史拟合的有限、平滑父版本风险 logit；不将确定通过标签当作严格零风险。
- \(\phi\)：少量局部支持及有向边界特征；不堆叠大量自由参数。
- \(\delta_t\)：由少量实际目标查询更新，采用零均值、强收缩先验；既允许增加也允许减少局部风险。
- 父版本拟合在该会话中固定。已作为 offset 使用的历史，不重复注入同一份强残差均值先验。

回退样本在旧通过侧将风险向上修正；改善样本在旧失败侧将风险向下修正。模型输出用于排序，不解释为自然道路场景的真实事故频率。

没有可靠历史边界时使用普通坐标特征和目标反馈回退；具体回退规则事先固定。不得按最终目标收益决定是否开启某个上下文的历史。

**局部独立可信度网络不列为首版必需项。** V4 的 LocalGate 可以作为后续消融，在上述小模型确实出现负迁移时再评估。不要同时叠加 offset、强历史先验、全局混合权重和局部混合权重，导致历史证据重复计入且收益无法归因。

### 3.3 固定双池预算，而不是再训练一个预算分配策略

主协议建议最大总预算 40：20 次用于旧通过池，20 次用于旧失败池；交替执行，先回退方向。报告总查询 10、20、40 的前缀及各方向实际查询数。另报每方向 1、5、10、20 次机会对应结果。

这使每个方向能沿用原有 20 次规模。不能把“双向总预算 40”的成绩与旧 V3 的“单向总预算 20”混比。所有适配基线共用同一调度和反馈权限。

若某父版本池不足 20 个，事先按已知池大小确定用尽该池后的额度转移规则；报告实际方向预算与短池。不能根据未查询目标中是否存在变化来转移预算。

每轮在被安排的池中选对应变化分数最高的尚未查询场景，获取真实目标结果，更新共享后验。风险近似平局时使用冻结物理身份打破平局；不要靠后验 Monte Carlo 噪声制造排序差异。

### 3.4 首版明确不做的事情

不使用目标反馈改变正式候选池；不把 PBTR 当主方法；不强制固定覆盖查询；不加入价值前瞻/DAD/MetaBO 训练；不构造多保真模拟器；不训练 PEARL/SAC 元策略；不增加大型深核/CNP。

新回退可能远离历史边界。正式候选池保留完整规则网格，不能只保留边界薄带。评估中单列“近历史边界／远历史边界”表现；历史方法失败是要报告的结果，不能删掉远端样本。

### 3.5 在线流程

```text
授权父版本历史 → 构建局部对比特征与父版本模型
             ↓
按父版本结果分成两个池（两个池都保留）
             ↓
按固定额度选择当前方向
             ↓
用共享后验给该方向候选打分
             ↓
查询一个目标结果 → 记录回退/改善/无翻转/异常
             ↓
用真实目标标签更新共享后验 → 下一次查询
```

---

## 4. 被测可学习算法：适配 highway-env，而不是迁移驾驶平台

### 4.1 主对象：现有 SB3 PPO 的持续更新链

项目确有：

```text
sut_algorithms/highway_env/ppo_ece.py
sut_algorithms/highway_env/checkpoints/ppo_ece/vd_1_5_trial_1.zip
```

`PPOPolicy.load()` 通过 SB3 PPO 加载，推理使用 CPU、`deterministic=True`、返回整数动作。现有 `PolicyAdapter` 支持指定 checkpoint 路径。仓库目录中确认存在相应 zip，但本轮未加载反序列化模型，也未运行其接口测试。[C03][C04]

统一 runner 当前使用 Kinematics 观测：5 个车辆槽位、`x,y,vx,vy,sin_h,cos_h`，相对坐标、排序、归一化、包含后方；物理频率 20 Hz，PPO BuildSpec 控制频率 5 Hz。[C04][C05]

这不是原始相机/LiDAR 感知到驾驶的模型；准确名称是 **基于运动学观测的闭环学习型驾驶决策策略**。

### 4.2 动作能力必须检查

主接口沿用 `DiscreteMetaAction`，包含换道与目标速度增减；低层转向/速度跟踪由环境车辆控制器执行，不是神经网络直接输出连续油门、转角。[W02]

必须冻结并核查：动作索引顺序、`target_speeds`、上下速度边界、控制频率、是否有有效停车/避让动作、观测语义、归一化和 checkpoint wrapper。

若现有 checkpoint 的速度动作不能实现停车，不能让它在“无避让通道且必须停车”的场景中必然失败，然后称为算法更新的安全回退。S02 可使用预先定义的可横向避让上下文。需更改动作契约时，应在 V0 前完成适应并注册新契约；V0→V1→V2 期间不改变动作含义。

软件训练库与 highway-env 版本按项目契约固定，不因为官方最新文档存在新接口就直接升级项目。当前 runner 字符串标识 `highway-env-1.9.1`；实际安装版本仍须在执行机器核对。[C05]

### 4.3 对此前“端到端”要求的同环境补充

保留一条明确的图像到决策补充：**highway-env 原生灰度图堆叠→CNN→PPO 动作**。不引入 CARLA，不更换测试方法。官方环境支持灰度观测与帧堆叠；SB3 PPO 支持 CnnPolicy。[W03][W04]

它只能称为二维仿真图像到驾驶决策的端到端策略，不是量产视觉 ADS，也不是车辆真实摄像头视角。使用离散元动作时，低层控制仍由环境完成。

状态 PPO 的权重不能直接改名为 CNN 权重。当前项目未核实可直接复用的图像 PPO checkpoint；该补充须显式安排初始化、必要训练和三个连续版本的独立链，不得假称已经运行。优先用已有兼容权重；否则应公开计入视觉根模型的初始化训练成本。在实施预算不足时，论文必须明确缺少这项图像输入证据，不能用状态 PPO 冒充。

这是一项输入模态补充，不再增加第三种学习算法、多个模拟器或多套大型驾驶栈。主方案先完成已有 PPO 状态链的闭环与训练接口。

### 4.4 非学习型版本链不再只是机制对照，而是第一阶段正式验证

**可以，而且建议先做。** 对本文这种“版本变化测试”问题，先使用非学习型控制器模拟连续软件 release 有三个优势：版本差异可解释、全网格建库成本低、可以在没有训练随机性和 checkpoint 选择混杂的条件下验证双向选择器是否真的工作。它不是替代学习型实验，而是更干净的第一层证据。

但可信的非学习型版本不能是“换一个完全不同的控制器”或“为了制造碰撞随意插入故障”。它必须满足：

1. **同一控制器家族与同一接口。** V0、V1、V2 使用同一个 `ProfiledIDMVehicle`（或后续单独定义的同一家族实现），车辆物理、观测、道路和判定规则不变；只修改软件逻辑/标定参数。
2. **明确父子继承。** V1 从 V0 的代码/参数出发修改，V2 再从 V1 修改；保存 change log，而不是把三个独立 profile 事后排序成“版本”。
3. **每次更新有独立开发目的。** 例如效率/舒适性标定、安全距离标定、短 TTC 预测制动补丁。参数通过开发集与需求确定，不根据最终确认库中回退/改善数量挑选。
4. **正式测试前冻结。** 版本参数、场景 manifest 和判定规则在读取最终目标真值前固定。某个版本最终只有改善、只有回退或没有变化，都必须保留。
5. **与故障注入分开。** `merge_blind06`、`merge_brake2`、`slow_front_brake2`、rear-guard bypass 等继续作为受控 stress/机制实验，不能与正常 release chain 混称为同一种版本演化。

当前仓库已经提供了同一 IDM/FVDM 实现的 profile 机制，并存在 `AV-Reference-IDM`、`AV-Calibrated-IDM`、`AV-Predictive-Brake` 等参数化控制器；另外还有 `idm_revision_pilot.py` 的同家族版本 pilot。[C08][C09] **这些现有 profile 证明工程上可行，但它们本身尚不能自动视为一条真实父子 release chain。** V6.1 要在同一个基线 profile 上显式定义增量修改与父关系。

### 4.5 建议的第一条非学习型 release chain

首版建议只做一条纵向/交互控制链，先把方法与实验协议跑通：

| 版本 | 软件更新目的 | 相对父版本的允许修改 | 论文角色 |
|---|---|---|---|
| **NL-V0：Reference IDM** | 稳定基线 | `time_wanted=1.5`、`desired_gap=5.0`、`max_brake=5.0`、无额外 emergency TTC safeguard（可直接承接现有 reference profile） | 历史起点 |
| **NL-V1：Efficiency/response calibration** | 改善通行效率与一般响应 | 在开发集上小范围重标定 headway / desired gap / comfort acceleration / normal braking；不增加独立安全补丁 | 模拟一次正常性能版本更新；允许自然出现 Safe→Unsafe 与 Unsafe→Safe |
| **NL-V2：Predictive-brake patch** | 针对短 TTC 风险增加安全补丁 | **继承 NL-V1 的基础参数**，只增加/调整 `emergency_ttc`、`emergency_brake_gain`、`emergency_max_brake` 等短 TTC 预测制动逻辑 | 模拟后续安全修复 release，重点检验 Unsafe→Safe，同时仍检查其它区域是否新回退 |

为了避免把结果“设计出来”，NL-V1 的具体标定范围和 NL-V2 的补丁阈值只能在**开发场景集**上确定。可以参考当前 `AV-Predictive-Brake` profile 中已有的 short-TTC safeguard 实现作为代码起点，但正式 NL-V2 应当从冻结的 NL-V1 继承，而不是同时把 headway、gap、comfort 参数全部重置成另一个独立 profile。[C08]

这条 release chain 的目的不是声称 IDM 代表现代量产 ADS，而是提供一个**确定性、可解释、可完整遍历的版本演化试验台**。如果本文方法连这条链上的双向变化都不能稳定优于静态历史/随机基线，就不应先投入更昂贵的 PPO 持续更新实验。

### 4.6 可选第二条非学习型链：MOBIL 逻辑更新

只有在 NL-IDM 主链通过后，再考虑一个横向决策链，例如围绕换道收益阈值与 candidate-lane rear safety guard 做“效率标定→安全约束增强”的父子版本。现有 `RearGuardIDMVehicle` 已把 rear guard 逻辑集中在同一类中，说明实现入口存在；但当前 `mobil_rear_guard_off_v2` 是**关闭安全否决的受控退化版本**，不能直接充当“正常 V1 release”。需要另建正常开发目的的 `mobil_release_v0/v1/v2`。[C10]

VI/MCTS 不强制进入版本链；IDM→PPO 属于跨算法比较，不作为连续软件版本。ATSLG 的 FVDM 代理与 ACC+AEB 被测对象也不构成父子版本；只采用其低维响应图、代理—目标差异和有限反馈实验思路。[L01]

---

## 5. 学习型版本链：在非学习型协议稳定后再验证持续更新

**实施顺序固定为：NL-IDM release chain → PPO 权重继承链 →（资源允许时）灰度图 CnnPolicy。** 非学习型链用于确认双向标签、完整银行、历史边界特征和选择器评价没有逻辑错误；PPO 链用于证明方法并非只依赖手工控制器的参数几何。两层使用相同的版本矩阵、目标隐藏、双池预算与指标定义。

### 5.0 进入 PPO 阶段的门槛

只有当下列条件在 NL-IDM 链上完成后，才扩大到 PPO：

- 全版本完整 bank 与四类变化矩阵可复现；
- Safe→Unsafe / Unsafe→Safe 标签与双池查询没有方向错误；
- 有向边界模型相对静态边界/普通坐标基线至少在开发集显示可解释增益或明确失败原因；
- 目标未查询标签泄漏测试、成本账和缓存指纹均通过；
- 版本参数/场景网格不是根据确认库结果反向挑选。

### 5.1 最小主版本链

```text
已有 PPO checkpoint
       ↓ 开发环境的兼容性与能力核验（必要适应计费）
固定 V0
       ↓ 新条件训练分布 + 旧条件重采样
固定 V1
       ↓ 下一组新条件 + 历史条件重采样
固定 V2
```

V1 从 V0 权重继续训练，V2 从 V1 权重继续训练。使用相同模型结构、观测预处理、动作语义、训练奖励定义、物理与判定契约。新增版本主要改变权重及其训练记录，不通过更换模型名称模拟演化。

### 5.2 PPO 的历史保持不能照抄离线经验回放

PPO 的标准实现以当前策略采集的 rollout 训练。[W05] 因此：

**保存并重采样旧训练场景的定义/种子/分布，在这些旧场景上用当前 PPO 策略重新执行，得到新 rollout。** 这叫场景重采样或场景演练，不是把上一策略收集的旧轨迹直接混入 PPO 的 on-policy loss。

一个开发起始混合可以是“新增训练分布 50%＋历史训练分布 50%”；比例仅作开发配置示例，正式冻结前确定。新条件并非新模型从未见过的场景语义，而是新实例或新交互参数分布。不要将训练场景重采样称为本文测试器的历史失效记忆。

不需要在线大规模元强化学习，也不需要从零重新学习全部驾驶能力。增量 PPO 的训练交互仍有实际成本；不能因使用已有 checkpoint 就把训练写成免费。

### 5.3 更新阶段的开发目的

V1 可适应一组新的切入时序/相对速度条件；V2 可适应切入后制动等另一组条件。持续更新的目的来自驾驶能力开发，不是最大化最终回退数量。

对每个阶段，在独立验证集上检查新增条件表现和旧条件保持，并按冻结的步数/验证规则选择 checkpoint。不得从最终全库中挑出让本文测试器最占优的版本。没有回退、没有改善或某方向历史失效，都是可能且应保留的结果。

### 5.4 四类数据身份

| 数据 | 用途 | 禁止行为 |
|---|---|---|
| SUT 训练/场景演练集 | 产生 PPO 新版本 | 回填最终测试轨迹 |
| SUT 验证集 | checkpoint/训练超参数选择 | 根据最终双向发现数选模型 |
| 测试方法开发集 | 边界片段数、正则化、模型因素与预算协议确定 | 反复调参后称其为确认结果 |
| 最终全版本配对库 | 确认方法性能、计算真值 | 把目标未查询结果传给选择器 |

按物理 episode 来源和参数上下文划分，不随机拆散同一场景的相邻帧来制造独立验证。

### 5.5 版本与训练账本

每个版本至少保存：parent ID、checkpoint SHA256、架构、观测与动作契约、归一化状态、训练配置、优化器继承策略、历史场景清单、新条件清单、RNG、实际 policy steps、物理 steps、训练耗时、验证选模规则。

`reset_num_timesteps=False` 只能保持计步等状态，并不自动定义持续学习或保证能力保持。[W04] 学习率计划、优化器状态和每阶段 rollout 的处理必须明确。

---

## 6. 场景组织与完整结果库

### 6.1 原首版三个主场景族

| 显示名 | 规范语义 ID | 现有实现 ID | 原二维参数建议 |
|---|---|---|---|
| 邻车切入 | `scn_cut_in` | `fbrt_cutin` | 初始间隙×换道时间尺度 |
| 前车切出显露静止目标 | `scn_cut_out_static` | `fbrt_cutout_static` | 初始间隙×现有名义静止目标时间参数 |
| 邻车切入后制动 | `scn_cut_in_brake` | `fbrt_cutin_then_brake` | 初始间隙×实测并线后制动时间差 |

原 `research_v2` 中 S08 的并线后制动时间差从 `fixed_context` 读取，原二维建议未实施。V3 已将该时间差接入活动参数，并与间隙、切入车速度、换道时间尺度和减速度组成五维开发候选；旧冻结清单仍按原值执行。[C05]

S02 的状态输入可能提前观测到静止目标，不能称视觉遮挡感知实验。必须检查实际观测和事件日志，按其真实作用解释为交互/避碰条件。规则模型、状态 PPO、灰度 PPO的物理 manifest 可保持相同，但三者观测语义不同，分层报告。

每类主场景设置若干实际不同的固定上下文，如相对速度、避让通道和事件时序。原 S03/S04 等停车场景暂不进入主 PPO 套件，先依据动作能力决定是否单列；S10/S12 不在首版继续扩展。IA/IB 交互支线已从活动代码退役，其 age080 历史阴性结果仅保留为来源记录。

### 6.2 资格与空间冻结

已准备的单上下文二维开发清单固定 8 个可执行场景族，每族 121 个场景、总计 968 个；这是原方案的独立记录。后续开发候选改按 `FBRT_Scenario_Parameter_Space_V3.md` 的场景专属维度取样，不能把 V3 参数空间回写到旧清单。NL 与 PPO 复用相同物理参数组合并分别建立三版本响应库。新增上下文应在新的实验清单中预先确定，不依据单上下文结果筛选有利条件。

资格基于物理有效性、接口能力、事件定义与研究范围。在目标执行前冻结。

不能把 parent-fail 当物理无效；完整库保留父失败。对于怀疑物理不可解的场景，按预先定义的独立见证协议区分可避免、未证实等状态，不能把“某控制器失败”当作不可解证明。

主池使用均匀规则网格。正式库不按目标表现、本文方法胜负、边界附近是否有更多收益调整。若开发阶段发现结构完全不适用，可在冻结之前修改并登记；正式结果中的负收益或零信号不可删除。

### 6.3 全版本矩阵

每个 SUT 版本执行同一份完整 manifest，建立：

$$
Y[v,\text{scenario},\text{realization}].
$$

网格遍历是有限测试集上的完整测量，不是连续空间穷尽验证。每个条目必须有有效结果或明确异常状态。残留未知时，不能声称所有真值已知；统计有效配对数、异常数和适用分母。

若有随机性，固定一次配对实现可以作为“指定实现上的变化”主终点。若要估计场景概率，则需另定重复设计；不能通过只复测翻转点获得整库同质量的概率真值。相同数值 seed 不保证随机过程完全一致，要检查独立随机流与执行路径。

### 6.4 历史可见性与生命周期

当前父版本全库可见，是主实验明确条件。对 V0→V1 测试，V1 未查询结果隐藏；对 V1→V2 测试，完整 V1 可作为父版本，但必须承认在两次会话之间已经补齐 V1 的全库结果。

不能同时说“每代只测 20/40 次，无额外测量”，又在下一代免费使用完整上一代。严格稀疏生命周期不列为首版主张。

场景描述、父版本历史和目标已查询结果可以给选择器；完整目标变化集合、剩余翻转数量、未来版本、内部修改类型不得泄漏。

---

## 7. 实验协议与评价

### 7.1 四个研究问题

1. 固定预算下，回退与改善两方向是否都比强基线更早被发现？
2. 方向性通过—失败结构是否比旧失败中心、静态边界距离更有价值？
3. 少量目标反馈是否带来正确局部适应，是否覆盖多个变化区域？
4. 方法在持续更新、远离旧边界、历史支持弱和零变化任务中的表现与成本如何？

### 7.2 最小主比较

所有双向方法使用同一旧通过/旧失败池、相同方向调度和总预算。主比较限定为：

| 方法 | 要隔离的因素 |
|---|---|
| 分层 Random | 无历史排序的下限；保持方向配额 |
| 父版本静态风险排序 | 用旧响应模型，回退按高旧风险、改善按低旧风险，无目标更新 |
| 静态边界距离 | 同样的历史边界，但没有版本适应 |
| 当前中心特征模型的双向适配版 | 共用新预算，仅把原模型转为两个方向的合法评分；明确不是原 V3 已有实验 |
| 同结构目标数据模型 | 保留方向所必需的父标签，但不用历史轨迹/边界先验；不能称完全无历史 |
| 本文：有向边界＋局部版本差异 | 检验完整主方法 |

旧 HistoryRank-UCB、FailureDistance 和 V3 原协议保留在“旧单向回归基准”中。扩成双向时必须定义对称风险/距离含义，而不是让原先只找碰撞的方法未经适配去参加改善竞赛。

最接近的通用残差对照应在消融中保留：同样的 offset 和预算，只将有向边界特征换为普通坐标/中心特征。ATSLG/DETOUR 启发的实现明确标为适配，不冒充全文复现。

### 7.3 主指标分方向报告

设第 d 方向已消耗 j 个查询机会后累计发现为 \(D_d(j)\)，d 为 R 或 I。对每方向 K=20 定义：

$$
A_d(K)=\frac{2}{K(K+1)}\sum_{j=1}^{K}D_d(j).
$$

这衡量早期发现，不依赖目标全库数量，因此可作为方向主指标。两方向分别为预注册的主要比较，不用一个加权总分掩盖任一方向退化。报告 D@1/5/10/20 和总预算前缀 10/20/40。

完整库支持：

$$
\operatorname{Recall}_R=|Q\cap\mathcal R|/|\mathcal R|,\quad
\operatorname{Recall}_I=|Q\cap\mathcal I|/|\mathcal I|.
$$

某方向真值为空时，召回和相对百分比为 NA，发现数为 0，不能把零分并列算作优势。池不足时报告短池、实际机会数和预先规定的归一化，不伪造 20 个可查询场景。

### 7.4 区域覆盖与图像

二维网格采用冻结的邻接规则（例如 4 邻接），由评估者分别计算回退区域和改善区域的连通分量。区域覆盖为被发现点触及的分量比例。该量是网格与邻接定义下的覆盖，不是独立软件缺陷数；报告至少一个网格分辨率/邻接敏感性诊断。

如做行为覆盖，碰撞对象、事件阶段和轨迹聚类规则必须共用。目标未查询轨迹不得为选择器构造特征。

优先完成六组图：

- 旧/新版本响应和四类变化真值图；
- 查询点按方向及顺序叠加的轨迹图；
- 两方向发现曲线与有限库召回；
- 两方向变化区域覆盖；
- 中心/有向特征、有/无目标适应的消融；
- 版本×训练阶段能力保持矩阵及成本表。

真值图不能假称是少预算算法重建的图。若另评价模型的变化区域预测误差，分开已查询与未查询条目，预先冻结判定阈值，并说明其为辅助预测指标。

### 7.5 统计单位

选择器的十次随机重复只用于观察选择稳定性。先在物理上下文/版本链层面形成配对结果，再报告效应量与区间。相邻版本共享模型、多个链共享 V0、网格点相邻，都存在相关性。

至少预指定两个方向与主要基线的比较，控制多重比较。样本量/训练重复数在开发阶段按成本和方差预定，不因为“尚不显著”逐次增加种子后只保留最后结果。

---

## 8. 分阶段成本：统一环境，不再使用 V5 的 CARLA 预算

以下是拟议的量级规划，尚未测量实际耗时，不是保证训练收敛或统计功效的阈值。

### 8.0 Phase NL：先完成非学习型 release chain

**这是新的第一正式阶段，不需要 GPU 训练。** 先在 3 个主场景族的冻结二维网格上完整执行 NL-V0/V1/V2，建立第一套双向 full bank；用它完成 Random、静态风险、静态边界、普通残差与本文有向边界模型的全部离线回放和消融。

原单上下文二维开发清单为 `8 场景族 × 1 上下文 × 11×11`，即 **968 个具体场景 × 3 个非学习型版本 = 2904 次基础执行**；该数字不适用于新的变维 V3 候选。V3 八族共 19,456 个候选；若全部资格和终点检查通过并冻结，每条三版本链对应 58,368 次基础执行。PPO 可以复用相同物理参数清单，但需要另行执行其三个版本。

Phase NL 的成功标准不是要求两个方向都人为出现，而是：协议可复现、版本变化来自冻结的正常更新、至少有足够有效变化可评价时能公平比较方法；若某方向真值为空则按 NA 规则报告。

### 8.1 先接口与开发，不先大规模建库

**先用非学习型 NL-V0→V1→V2 完成 full-bank、双向 replay、指标和泄漏审计。** 这一步不依赖 PPO 训练接口。

之后再用开发小集合完成 PPO checkpoint 加载、观测语义、可执行动作、单一控制权、训练 step 与评估执行一致性、权重确有更新、没有标签泄漏。随后验证学习型 V0→V1→V2 训练流程与独立验证选模。无论双向变化是否都出现，都不以最终测试输出反向挑选 checkpoint。

### 8.2 主 PPO 配对库规划

原单上下文二维清单：8 场景族×1 物理上下文×11×11 网格=968 个场景。新的变维 V3 仍是开发候选，尚无 PPO 物理测量；如果资格检查后全部 19,456 个候选均被预先冻结，PPO V0/V1/V2 各执行一次将是 58,368 次基础执行。

两条增量训练链共享 V0 时，执行版本为 V0、V1a、V2a、V1b、V2b，基础执行数等于冻结具体场景数乘 5；若采用全部 19,456 个 V3 候选则为 97,280 次。共享 V0 的相关性必须明确。固定 r 次物理重复则乘 r，另计开发/重跑。

这两种是预算层级，不是先完成小样本、看显著性再决定扩样的规则。正式采用哪一层在查看确认目标结果前确定。

### 8.3 图像 PPO 补充

先做小范围但完整的配对库，例如 3 场景族×1 上下文×7×7=147 个场景，3 个版本对应 441 次基础执行。它只承担图像输入兼容性的补充证据，不借小样本宣称视觉 ADS 普遍有效。

图像 V0 获取/初始化、必要训练、持续更新与渲染成本单列。并未确认现有图像预训练权重，因此不得直接把这 441 次写成整个图像实验成本。

### 8.4 成本账

$$
C_{\mathrm{research}}=C_{\mathrm{SUT\ initialization}}+C_{\mathrm{SUT\ updates}}
+C_{\mathrm{full\ bank}}+C_{\mathrm{diagnostics}}+C_{\mathrm{selector}}.
$$

$$
C_{\mathrm{online\ version\ testing}}=C_R+C_I+C_{\mathrm{selection}}.
$$

部署上额外的强制修复确认用例也要计费，不能隐含免费。参考完整银行建设是历史已付成本，不是不存在。不同实验重复读取缓存不是新增物理执行。

训练同时报告 policy decision steps 与 20 Hz physics steps，不能将两者混称环境步。最新 SB3 的 `.learn(total_timesteps=...)` 可能按完整 rollout 超过请求值，预算按实际采样统计。[W04]

---

## 9. 代码落点与最重要的实施风险

### 9.1 当前 runner 不是已完成的 PPO 训练环境

`FBRTUnifiedEnv._advance()` 内部调用冻结 adapter 产生动作，`run_build_episode()` 直接循环 `_advance()` 完成评估。[C05]

因此，不能把此对象直接交给 `PPO.learn()` 就宣称完成持续学习：训练器提供的 action 必须实际控制 ego，reward、terminated/truncated、reset、观测返回和控制频率必须形成规范接口。

建议将物理推进拆成共用内核，分别提供：

- 评估：adapter 产生动作 → 统一 step；
- 训练：当前 PPO 产生动作 → 同一个 step。

同一 policy tick 只能有一个 ego 高层动作所有者；不能在 PPO 动作之后再次被旧 adapter 或 `road.act()` 覆盖。一个 5 Hz 决策 tick 推进 4 个 20 Hz 物理 tick，并按相同规则推进背景事件和低层跟踪。训练环境与评估器在冻结策略、同初始状态下需做逐步一致性测试。

图像帧堆叠在约定时点更新一次；不要因为日志/adapter 多次调用 observe()，导致训练与评估的帧历史不同。

共用内核改变后更新执行契约并做兼容回放。不能直接将改变后的物理记录与旧 bank 混用。

### 9.2 最小工单

| 工单 | 源码位置/新职责 | 验收 |
|---|---|---|
| A. 动作/训练接口 | `highway_sim_env/envs/fbrt_unified_env.py`、新增 training wrapper | PPO action 确实作用；训练评估轨迹同契约一致 |
| B. 版本身份 | `sut_algorithms/highway_env/registry.py`、`fbrt_adapters.py`、训练入口 | V0/V1/V2 不同权重哈希、明确父关系，无观测/动作暗改 |
| C. 双向任务定义 | `replay_utils.py`、`schema.py` | 增加 improvement/bidirectional 合法模式；四类变化单元测试；改善标签为目标 0 |
| D. 全库银行 | `experiment.py`、`replay.py`、archive | 所有版本覆盖完整 manifest，父失败不删，异常明确，目标只经 oracle 揭示 |
| E. 历史与小模型 | `pattern_memory.py`、`bayes_model.py` | 有向对比与普通残差可独立开关；不重复注入历史先验 |
| F. 双池选择 | `selector.py` | 固定预算、无放回、目标共享后验、短池规则、相同基线协议 |
| G. 结果与图 | report/benchmark | 两方向曲线、召回、区域覆盖、四类变化、成本、全任务表 |

路径中未存在的新模块是职责建议，不声称已有命令参数支持。主方法仍放在 `methods/failure_memory_regression/`，通过模式配置区分，不复制多套版本目录。

### 9.3 先修正已有比较问题

原 V3 默认取消固定覆盖槽位，而其他 FBRT 比较含第 10/20 次覆盖；新比较统一预算与调度。零 TTC/间隙不得当缺失，首次发现按预算前缀截尾，概率积分与平局规则稳定，缓存包含全部模型和执行契约因素。工程修复不包装为学术创新。

### 9.4 自动检查

至少覆盖：

1. parent=1,target=0 的记录进入改善统计但不是目标失败；
2. 两个池都能被主动查询，已知无池与真实无变化区别明确；
3. 改写未查询目标标签不会影响当前候选和查询前预测；
4. 目标异常不能在查询前泄漏或免费移除；
5. 查询实际执行成本、短池和缓存计数一致；
6. 修改子版本 checkpoint 确实改变载入哈希，不错误复用旧 adapter 实例；
7. 完整 V1 作为下一次父历史时有对应补测/建库记账；
8. 图像模型没有错误载入状态模型权重，帧堆叠与观测空间一致。

---

## 10. 文献作用与创新边界

| 文献组 | 保留的借鉴 | 不再引入 |
|---|---|---|
| IFR、DETOUR | 失败区域结构、历史邻域的强比较 | “首次使用历史/边界”的主张 |
| ATSLG、Coverage-Aware、AdaTE | 代理/历史失配、局部响应差异与有限反馈 | 事故率无偏估计或安全认证结论 |
| PEARL、CNP、FSBO、CORRO | 条件适应、历史来源与数据分布审计 | 大型元训练及编码器作为必选 |
| DREAM、DAD、MetaBO | 未来可研究查询信息价值 | 首版非短视优化器与额外探索训练 |
| FREA、STRIVE、SAGE、ScenarioFuzz | 场景有效性、可执行性和独立安全见证的边界 | 额外生成器/对抗驾驶策略训练 |
| BehAVExplor、SAMOTA、SDC-Prioritizer、SAVME、SCBO、FST | 覆盖、成本、局部搜索与评估目的区分 | 所有组件同时成为本文算法模块 |

这些文献的详细阅读记录保留在 V4/V5；本文件不因新增双向目标而声称已检索到不存在的双向最优性定理。

潜在贡献：将历史局部失败—通过结构用于更新策略的双向变化排序，并在少目标反馈下进行局部适应；以可信持续更新与完整隐藏真值检查这种结构的增量价值。

双池、互补概率、PPO 续训、完整建库、画四类变化图本身都不是新的算法原理。论文是否成立，取决于有向边界和适应在两方向相对强基线的真实增益，而不是重新命名或模块数量。

---

## 11. 交付与最终表述

最终至少包含：`protocol.json`、`version_lineage.jsonl`、`scenario_manifest.jsonl`、`full_response_bank.jsonl`、`transition_truth.csv`（评估端）、`queries.jsonl`、`summary_by_direction.csv`、`summary_by_context.csv`、`ablations.csv`、`cost_ledger.json`、`tests_report.txt` 与统一命名图表。

固定报告章节：研究范围与判定；版本来源；全库完整性；回退发现；改善发现；模式/区域覆盖；消融；零变化和失败案例；成本；适用范围。

最终方法定位：

> **在 highway-env 中，针对连续演化的自动驾驶软件版本——先是可解释的非学习型控制器 release，再是具有真实权重继承关系的学习型策略——利用旧版本局部通过—失败结构和少量目标反馈，在有限预算下同时主动发现新增回退与实际改善。**

不把它写成整体安全证明、不把图像二维仿真写成真实感知部署、不把当前 V3 的单向旧结果写成新的双向持续学习验证结果。

**最新优先级：先完成 NL-IDM V0→V1→V2 的正常 release chain 和完整配对库；方法逻辑稳定后，再做 PPO 持续更新。** 这能最大限度降低 GPU 与训练不稳定性带来的实验风险，同时保持最终论文仍有学习型策略版本演化证据。

## 来源与核对范围

### 本项目与历史文件

- [C01] `methods/failure_memory_regression/README.md`：当前方法、模块职责与银行回放。
- [C02] `methods/failure_memory_regression/replay_utils.py`、`replay.py`：当前奖励模式、父通过筛选及 evaluator-only 银行。
- [C03] `sut_algorithms/highway_env/ppo_ece.py` 与 `checkpoints/ppo_ece/`：PPO 加载方式和 checkpoint 元数据。
- [C04] `sut_algorithms/highway_env/registry.py`、`fbrt_adapters.py`：受控延迟不是权重更新、PPO 控制频率和 adapter。
- [C05] `highway_sim_env/envs/fbrt_unified_env.py`：20 Hz 物理、运动学观测、脚本场景、内部 adapter 动作与评估循环。
- [C06] `FBRT_Optimization_Objectives_V4_Expanded_5refs.md`：历史边界/残差/可靠性候选及旧协议审查。
- [C07] `FBRT_Research_Plan_V5_FullBank_Continual_E2E.md`：完整有限银行应保留；其平台与单向主目标在 V6 废止。
- [C08] `sut_algorithms/highway_env/idm_profiles.py`：同一 `ProfiledIDMVehicle` / `ProfiledFVDMVehicle` 的参数化控制器、reference/calibrated/predictive-brake profile 与 short-TTC safeguard 实现。
- [C09] `methods/core_mine/idm_revision_pilot.py`：已有同家族 IDM revision pilot，证明同一 controller family 上进行版本差异实验的工程入口已经存在；其 delay/brake 弱化构建仍仅作受控开发证据。
- [C10] `sut_algorithms/highway_env/regression_builds.py`：MOBIL rear-guard 逻辑集中在同一车辆类，现有 bypass/aged-state 构建属于受控差异，不自动等于正常 release chain。

代码均对应本文件开头的固定快照；只做静态核对，不等于已跑通训练接口。

### 用户提供论文

- [L01] *Testing Scenario Library Generation for Connected and Automated Vehicles: An Adaptive Framework*，PDF 第 6–8 页：FVDM、ACC+AEB、二维差异函数、有限适应；不是真实版本链。
- [L02] *Identification of Failure Regions for Programs with Numeric Inputs*，第 3–6 页及假设/局限：失败—边界结构。
- [L03] *Coverage-Aware Active Evaluation for Failure Discovery with Paired Systems*：局部校正和覆盖是相关既有技术，不独占为本文贡献。
- 其余文献标题与章节定位见 [C06] 的逐篇阅读登记；本次是收敛方案，不宣称重新复现所有论文实验。

### 官方接口文档（2026-09-27 核对，接口实际使用以项目固定版本为准）

- [W01] Farama HighwayEnv Getting Started，Training an agent：`https://highway-env.farama.org/quickstart/`。
- [W02] Farama HighwayEnv Actions，Discrete Meta-Actions：`https://highway-env.farama.org/actions/`。
- [W03] Farama HighwayEnv Observations，Kinematics / Grayscale image：`https://highway-env.farama.org/observations/`。
- [W04] SB3 PPO 文档，支持空间、策略类型、CPU、load/learn：`https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html`。
- [W05] SB3 on-policy 实现，collect_rollouts：`https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/on_policy_algorithm.html`。
