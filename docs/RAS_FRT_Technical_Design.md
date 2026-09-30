# RAS-FRT：基于历史场景响应的新 SUT 危险场景发现

> **方法名称**：RAS-FRT（Response-Adaptive Search for Failure-Region Testing）  
> **场景族**：S01 Cut-In  
> **历史库 A**：同一 IDM 控制模型的五组参数配置，在 2048 个场景上的完整测试结果  
> **新 SUT**：完整 FVDM 控制器 `fvdm_safety_speed_23_mps`  
> **候选库 D**：2048 个 S01 场景，固定 FVDM 的完整实测结果  
> **目标查询预算**：每种方法最多 200 次反馈；早期开发比较报告 100 次

本文的问题是：当一个新的驾驶控制器到来时，如何利用 A 中积累的历史测试结果和场景相似性，在受限查询预算内尽早发现真实危险场景。正式实验只使用历史库 A 和新 SUT 候选库 D。第 15～32 节保留 D 上 100 次预算的开发记录；第 33 节在同一个 D 上扩展至 200 次预算和九种方法。D 的真值来自 `highway-env` 执行器中的 FVDM 仿真。

## 1. 为什么需要重新收敛方案

前期研究经历了多个阶段：历史失效边界迁移、双向版本变化发现、FBRT Failure Memory、PatternCard + Transformer + target support，以及 DFE / Global-Local Router。旧 S01 协议曾在同坐标候选库和故障注入目标上观察到 `HistoryRank-UCB-v2` 的 `D@50=50`。该结果只解释为什么需要改变测试问题，不作为本轮 FVDM 目标的基线成绩。

早期“historical SUT 与 target SUT 使用完全相同候选点”的设置，本质更接近“同一个场景历史上是否失败过”的直接查询，而不是“历史知识能否推广到新的场景”。因此当前方案改用独立的 D 坐标，保留 A 的逐场景 IDM 响应，让方法必须从历史结果泛化到新 SUT 的新场景。复杂 PatternCard 压缩和 Transformer 没有体现优势，反而可能损失逐场景历史响应。

因此，后续论文不应继续通过增加网络复杂度去试图超过一个已经达到 50/50 命中的基线，而应把问题重新聚焦到：

> **historical set 与 target candidate set 不完全重合时，如何利用同一 IDM 模型的多组参数响应，对一个完整的 FVDM 目标实现跨场景 few-shot failure testing。**

---

## 2. 最终 Motivation

自动驾驶系统在开发和升级过程中会积累大量历史测试结果。对于一个新的被测驾驶算法，完全从零开始重新探索整个场景空间成本很高。然而，历史数据存在两个根本局限：

1. 历史 IDM 参数配置只测试过场景空间中的有限样本；
2. 即使这些配置共享同一控制律，它们的失效模式也不一定适用于 FVDM 目标。

因此，简单复用历史 failure point、failure distance、failure boundary 或同场景历史风险，只能在历史覆盖充分、历史与目标行为高度一致时有效。

本文最终研究：

> **如何从同一 IDM 模型的多组参数配置的场景—响应数据中学习场景之间的行为相似关系，并通过少量 FVDM 目标测试在线修正历史先验，从而在严格预算下主动发现目标失效并近似识别其失效区域。**

一句话：

> **历史告诉我们“场景之间如何关联”，目标反馈告诉我们“这些历史关系在新系统上哪里失效”，主动选例则决定“下一次测试哪里最值得”。**

---

## 3. 研究任务定义

对固定功能场景族 (f)，定义合法逻辑场景空间 \(\Omega_f\)。本实验使用 S01 四维 Cut-In 参数空间。

### 3.1 历史库 A

\[\mathcal X_A=\{x_1^A,\ldots,x_{2048}^A\}\subset\Omega_f.\]

A 中每个坐标都由同一 IDM 模型的五组参数分别执行，因而每个场景带有五条历史响应。

### 3.2 新 SUT 的候选库 D

\[\mathcal Z_D=\{z_1^D,\ldots,z_{2048}^D\}\subset\Omega_f.\]

D 是独立 Sobol 坐标上的新 FVDM SUT 测试库，与 A 参数范围相同但场景坐标不重合。目标真值定义为 D 上真实碰撞的场景集合：

\[F_D=\{z\in\mathcal Z_D:y_{\mathrm{FVDM}}(z)=1\}.\]

选择器只接收已经查询场景的单个结果。D 全部响应在离线评价中作为完整真值，用于算出所有方法发现了多少危险场景和覆盖了多少危险区域；它们不进入历史训练，也不用于调节方法参数。

---

## 4. 历史知识的正确表示：Multi-Configuration Response Matrix

设有 \(M\) 组历史 IDM 参数配置，所有配置运行同一控制器实现。

历史场景坐标：

\[
X_H\in\mathbb R^{N_H\times d_f}.
\]

历史响应矩阵：

\[
Y_H\in\{0,1\}^{N_H\times M}.
\]

其中：

\[
Y_H[i,j]=y_j(x_i^H).
\]

`1` 表示该 IDM 参数配置在该场景发生有效 failure，`0` 表示有效 pass。

若有缺失执行，则维护：

\[
O_H\in\{0,1\}^{N_H\times M}.
\]

`O_H[i,j]=0` 表示该响应未知。缺失不能当成 pass。

### 4.1 PatternCard 的新定位

PatternCard 继续保留，但主要用于：

- failure cluster 解释；
- collision role 分析；
- maneuver phase 分析；
- TTC / clearance 统计；
- failure report 与可视化。

主预测器应优先保留完整 historical response matrix，不再让 PatternCard 成为全部历史知识的唯一入口。

---

# 5. Component 1：Historical Response Representation

目标：学习一个场景表示，使其能够解释同一 IDM 控制器的不同参数配置在各场景上的响应。

## 5.1 Scene Encoder

对于功能场景族 \(f\)：

\[
x=(x_1,\ldots,x_{d_f}).
\]

每个参数按冻结 bounds 归一化：

\[
\tilde x_j=\frac{x_j-l_j}{u_j-l_j}.
\]

第一版不使用复杂 Transformer，而使用小型 MLP：

```text
input d_f
→ Linear(d_f, 64)
→ GELU
→ Linear(64, 128)
→ GELU
→ Linear(128, 64)
→ GELU
→ Linear(64, 32)
```

得到：

\[
h_\theta(x)\in\mathbb R^{32}.
\]

第一阶段每个功能场景族独立训练，先验证方法机制，再考虑跨 family 共享编码器。

## 5.2 Historical SUT Response Heads

对每个 historical IDM 构建 \(m_j\)，设置独立输出头：

\[
\hat r_j(x)=\sigma(w_j^\top h_\theta(x)+b_j).
\]

组合：

\[
\hat R_H(x)=[\hat r_1(x),\ldots,\hat r_M(x)].
\]

其含义是：给定一个场景，模型预测各组历史 IDM 参数配置分别会如何响应。

## 5.3 Training Objective

使用历史真实响应训练：

\[
\mathcal L_{response}
=
\frac{
\sum_{i,j}O_H[i,j]\,BCE(\hat r_j(x_i^H),Y_H[i,j])
}{
\sum_{i,j}O_H[i,j]
}.
\]

这里不是训练“总体危险程度”，而是训练一个能同时解释多组 IDM 参数配置行为差异的场景表示。

---

# 6. 历史场景相似度

历史相似度由两部分共同决定：

1. learned historical response similarity；
2. physical locality。

## 6.1 Historical Response Distance

\[
d_R(x,z)=\frac{\|\hat R_H(x)-\hat R_H(z)\|_2}{\sqrt{M}}.
\]

## 6.2 Physical Distance

\[
d_X(x,z)=\|\tilde x-\tilde z\|_2.
\]

## 6.3 Combined Similarity

\[
\boxed{
S_H(x,z)=
\exp\left[
-\frac{d_R(x,z)^2}{2\sigma_R^2}
-\frac{d_X(x,z)^2}{2\sigma_X^2}
\right]
}
\]

其中 \(\sigma_R\) 和 \(\sigma_X\) 只能通过 historical pseudo-target validation 选择。预先允许 \(\sigma_R=\infty\) 作为物理距离单独决定相似度的退化设定；若五组历史 IDM 参数配置的响应不能提供额外区分信息，验证应选择该设定，而不是强行保留响应距离。不能查看 D 的完整真值后再决定是否使用响应项。

响应相似度回答的是“同一 IDM 模型的多组参数在这两种条件下是否呈现相似的通过 / 失效模式”，物理距离回答的是“四维切入参数是否接近”。两个参数点即使距离接近，也可能跨过某些控制器的制动或间距阈值，导致历史响应不同；只看坐标距离会把一次目标反馈传播到行为不同的条件。反过来，许多相距很远的场景可能都有相同响应，因此不能只看响应相似度。\(S_H\) 把两种证据同时作为局部传播的约束。

响应相似度的额外价值是**待验证假设**：若历史 IDM 的响应几乎相同，或预测头无法稳定预测未见坐标，它可能比单纯物理距离更差。历史伪目标验证应检查响应预测质量与邻近关系；主实验结果不能仅凭该表示的设计就归因于“学到了行为相似性”。

### 6.4 为什么不能只按历史标签是否相同定义相似

本轮有同一 IDM 模型的 5 组参数配置，二值响应组合最多只有 \(2^5=32\) 种，而且实际组合可能更少。大量物理上相距很远的场景可能都呈现 `[0,0,0,0,0]`。因此必须加入物理局部性，避免一次 target failure 被错误传播到整个历史安全区域。

---

# 7. Historical Prior

对任意 target candidate：

\[
z\in\mathcal Z,
\]

即使历史系统从未执行过该 exact point，响应模型仍可得到：

\[
\hat R_H(z).
\]

定义 historical prior：

\[
\boxed{
p_H(z)=\sum_{j=1}^{M}\pi_j\hat r_j(z)
}
\]

且：

\[
\sum_j\pi_j=1.
\]

第一版使用 source-balanced uniform weight：

\[
\pi_j=\frac1M.
\]

首轮 A/D 协议没有精确重合，不使用同场景历史真实响应替代模型预测。若后续允许重合，应把这些点单独分层报告。

注意：\(p_H(z)\) 只是 historical risk prior，不是 target SUT 的真实 failure probability。

---

# 8. Component 2：Few-Shot Target Residual Correction

这是整个方案的核心 target adaptation。

## 8.1 Target Execution

第 \(t\) 次执行：

\[
z_t.
\]

target SUT 返回：

\[
y_*(z_t)\in\{0,1\}.
\]

## 8.2 Target Residual

\[
\boxed{
e_t=y_*(z_t)-p_H(z_t)
}
\]

含义：

- \(e_t>0\)：historical prior 低估 target 风险；
- \(e_t<0\)：historical prior 高估 target 风险；
- \(e_t\approx0\)：target 与历史判断一致。

## 8.3 Similarity-Weighted Residual Propagation

对于未查询场景 \(z\)：

\[
\delta_t(z)=
\frac{
\sum_{i=1}^{t}S_H(z,z_i)e_i
}{
\lambda+\sum_{i=1}^{t}S_H(z,z_i)
}.
\]

然后：

\[
\boxed{
p_t(z)=clip[p_H(z)+\delta_t(z),0,1]
}
\]

其中 \(\lambda>0\) 是 no-correction strength，防止一个非常遥远的 target observation 被强制传播到整个空间。

---

# 9. 目标残差的局部传播

第 8 节的修正项将已观察到的历史—目标误差，按场景相似度传播到尚未查询的点。相似度越低，单条反馈的影响越小；分母中的 \(\lambda\) 使远离全部已测点的区域回到历史先验。这里使用明确的核加权平均，不把 Cross-Attention 作为论文方法名称或贡献。

**与下一步选例的连接**：每次查询后，先用真实 PASS/FAIL 更新所有未查询点的 \(p_t\)，再用同一相似度 \(S_H\) 计算这些点离已查询集合还有多远，最后将两项送入第 11 节的分数。前者回答“现在哪些点可能失败”，后者回答“哪些点还缺少相近的实测证据”。因此第 11 节不是另一套独立预测模型，而是使用第 8 节最新风险排序，再对相似点密集重复查询施加软惩罚。覆盖缺口不等于预测不确定性；一次 PASS 也能改变邻近风险及后续查询顺序。

---

# 10. 可选第二阶段：Target-Conditioned Similarity

第一阶段暂不实现。

只有显式 residual correction 确认有效后，才考虑让当前 target support 轻量改变场景相似度，例如学习：

\[
a_t\in\mathbb R^{32}_{+}
\]

并定义目标条件化表示距离：

\[
d_{t}(x,z)=\|\sqrt{a_t}\odot(h(x)-h(z))\|_2.
\]

其含义是：target 少量反馈可以改变场景表示中哪些方向更重要。

该部分是后续增量，不进入首轮主方法。

---

# 11. Component 3：Risk-Aware Coverage-Gap Search

目标：优先测试预测风险高、同时与已测目标场景相似度低的候选点。此处选的是**候选点本身**，不再对整个参考集计算“新增代表性覆盖质量”。

## 11.1 已查询场景的相似覆盖

对尚未查询的 \(x\in\mathcal Z\setminus Q_t\)，定义：

\[
c_t(x)=\max_{x_i\in Q_t}S_H(x,x_i),\qquad c_0(x)=0.
\]

覆盖缺口：

\[
g_t(x)=1-c_t(x)\in[0,1].
\]

\(g_t\) 只是相对于已查询点的相似度缺口，不能直接解释为目标结果未知的概率、预测方差或真实场景曝光率。

## 11.2 风险与覆盖缺口的选例分数

对有效 PASS/FAIL 反馈更新第 8 节的 \(p_t(x)\) 后，计算：

\[
\boxed{\operatorname{Score}_t(x)=(1-\beta)p_t(x)+\beta g_t(x),\qquad 0\le\beta<1.}
\]

并选择：

\[
\boxed{x_{t+1}=\arg\max_{x\in\mathcal Z\setminus Q_t}\operatorname{Score}_t(x).}
\]

两项均在 \([0,1]\)；首轮固定 \(\beta=0.2\)，不根据 D 真值调节。当前精简实验比较完整 RAS-FRT 与随机、历史排序和 Transfer-UQ，不单独消融覆盖项。因此结果可评价完整方法的发现与覆盖表现，但不能单独归因于某个分数组件。

---

# 12. 选例分数的三种行为

## 12.1 高风险区域

若 \(p_t(x)\) 增加，其他条件相同时，该点分数增加。

## 12.2 重复测试抑制

若候选点与任一已查询点高度相似，\(c_t(x)\) 增加、\(g_t(x)\) 降低。即使它仍有较高风险，也不会被硬性排除。

首轮不加入独立 DFE direction diversity；是否仍需要 DFE 留待首轮结果诊断。

## 12.3 低历史风险、低覆盖点仍有机会

当 \(\beta>0\) 时，低风险点可凭较大的覆盖缺口提高分数；它是否最终被选中取决于其他候选。不能声称加性分数保证每个低覆盖区域都能得到测试。

---

# 13. 完整在线流程

```text
Input:
    Historical scene set X_H
    Historical multi-configuration response matrix Y_H
    Target candidate set Z
    Budget B = 200 for the independent confirmation

Offline:
    Train historical response encoder
    Train source-specific response heads
    Freeze encoder
    Compute historical prior p_H(z)
    Compute and cache similarity S_H(z_i, z_j) on Z

Online:
    Q = {}
    residuals = {}

    for t = 1 ... B:

        1. compute residual-corrected target risk p_t(z)
           for all unqueried z

        2. compute coverage gap g_t(z) = 1 - c_t(z)

        3. compute Score_t(x) = (1-beta) p_t(x) + beta g_t(x)
           for each unqueried candidate x

        4. select x_t = argmax Score_t(x)

        5. execute target SUT on x_t

        6. if valid PASS/FAIL:
               observe y_t
               e_t = y_t - p_H(x_t)
               add (x_t, y_t, e_t)

        7. if inconclusive:
               consume budget
               do not fabricate binary label
```

---

# 14. 为什么暂时移除 DFE / Thompson Router

新分数直接组合当前目标风险与已查询场景的相似覆盖缺口。首轮只实现这一种新选择器；旧 DFE 与两臂路由代码保留，但不重新适配到新候选库。

若 failure 附近仍有高风险候选，其风险项仍可使其继续被选择；若相似候选已查询，覆盖缺口会减小。D 上通过与其他整体搜索方法比较发现曲线和危险区域覆盖，评估这一取舍，不把结果解释为覆盖项单独的因果效应。

---

# 15. A→D 开发实验协议（B=100）

实验只使用两个场景库。A 是历史知识库，包含同一 IDM 控制模型的五组参数变体；D 是新 FVDM SUT 的独立候选场景库。两库都属于 S01 Cut-In，场景坐标由不同的冻结 Sobol 种子生成。

## 15.1 历史库 A

| IDM 历史构建 | `time_wanted` (s) | `desired_gap` (m) | `max_brake` (m/s²) |
|---|---:|---:|---:|
| `idm_source_short_headway` | 1.0 | 6.0 | 5.0 |
| `idm_source_nominal` | 1.5 | 6.0 | 5.0 |
| `idm_source_long_headway` | 2.0 | 6.0 | 5.0 |
| `idm_source_limited_brake` | 1.5 | 6.0 | 3.0 |
| `idm_source_strong_brake` | 1.5 | 6.0 | 8.0 |

五组均为同一 `ProfiledIDMVehicle` 控制器，只改变表中参数；其余控制参数、车辆物理、场景执行器、仿真种子和 PASS/FAIL 定义相同。A 有 2048 个场景坐标，每个 IDM 变体均有完整实测响应，共 10240 条历史响应。A 的五个冻结响应模型及历史排序参数仅由 A 训练和确定。

## 15.2 新 SUT 与场景库 D

D 使用另一个独立 scrambled Sobol 样本，种子 `43105`，包含 2048 个 S01 场景。被测对象是完整 `ProfiledFVDMVehicle`，参数为 `target_speed=23 m/s`、`max_brake=8`、`desired_gap=8`、`fvdm_sensitivity=0.6`、`fvdm_velocity_gain=1.0`、`fvdm_transition_gap=8`；不使用故障注入。该 FVDM 参数配置在 D 采样前已由既有控制器校准固定；D 的场景坐标和结果没有用于挑选该配置。D 上 2048 次 FVDM 执行全部有效，其中 116 个真实碰撞场景。

A 的五组 IDM 只提供历史经验，不在 D 坐标上额外执行。D 上完整 FVDM 响应已作为离线基准真值保存；每种选择方法仍只通过 `TargetOracle.query()` 逐次获得所选场景的单个标签，最多 100 个。完整真值仅供最后核对发现数、危险场景空间覆盖和全库 AP。

## 15.3 基线、重复与指标

- **Random**：均匀随机选点，5 个固定种子。
- **HistoryRank-adapted**：只按 A 的历史失败风险排序，确定性运行 1 次。
- **RAS-FRT**：A 学得的目标风险预测，加上场景相似响应和已查询区域覆盖项；5 个模型种子。
- **Transfer-UQ**：基于 A 的风险先验与目标逐次反馈更新迁移残差不确定性；5 个模型种子。

各方法使用相同的 D、100 次查询上限和反馈接口。在预算 10、30、50、100 时记录已发现的真实 FVDM 危险场景数。多样性用已发现危险场景覆盖到的 S01 四维参数网格单元数衡量：每个归一化参数轴等宽分成 4 档。D 的 116 个危险场景共占 33 个单元。另报告基于完整 D 标签计算的 AP，作为全库排序诊断，而非预算内发现数。

**实验口径**：D 已经完整执行，因此这里是对真实执行结果进行的离线预算回放。回放严格隔离未查询标签，适合公平比较选择策略，但本次结果不能表述为实际只花了 100 次仿真就得到了完整真值。

---

# 16. 真实危险场景评价

真值集合为 D 上全部有效 FVDM 响应中的碰撞场景：

\[
F_D=\{z\in\mathcal Z_D:y_{\mathrm{FVDM}}(z)=1\},\qquad |F_D|=116.
\]

## 16.1 预算内危险场景发现

对每条逐次查询序列，在预算 10、30、50、100 处记录累计发现数 (D@k) 和 (D@k/116)。随机方法和依赖模型种子的方法报告均值、标准差；确定性 HistoryRank 报告单次结果。

## 16.2 危险场景多样性

把四个归一化 S01 参数轴各等宽切分为 4 档，以四元组网格单元标记场景位置。报告每个预算点发现的危险场景覆盖到多少个不同网格单元。D 的 116 个真实危险场景共占 33 个网格单元。该量衡量参数空间覆盖，不代表不同的物理碰撞机理。

## 16.3 全库风险排序诊断

对输出全库风险分数的方法，使用 D 的完整标签计算 Average Precision。AP 是离线诊断指标，不是预算内的查询发现数；Random 不产生风险分数，因此不报告 AP。

---

# 18. 对比方法

所有方法使用相同的 A 历史库、D 候选集、FVDM 反馈和 (B=100) 查询预算。

| 方法 | 作用 | 重复 |
|---|---|---:|
| Random | 不用历史经验的随机搜索基线 | 5 个种子 |
| HistoryRank-adapted | 只按 A 中历史失败风险进行近邻迁移排序 | 确定性 1 次 |
| RAS-FRT | A 学到的风险先验加场景相似响应与已查询区域覆盖项 | 5 个模型种子 |
| Transfer-UQ | 用 A 风险先验和逐次 FVDM 反馈更新迁移残差不确定性 | 5 个模型种子 |

RAS-FRT 参数由 A 上的伪目标验证确定；Transfer-UQ 的核和权重固定在方法定义中。D 标签不用于拟合或改动任何方法参数。所有方法共享 `TargetOracle` 逐点揭示反馈，完整 D 标签只在回放结束后供评价器使用。

---

# 19. 这组对比回答的问题

1. 相比随机查询和纯历史排序，RAS-FRT 是否更早发现 D 上真实危险的 FVDM 场景？
2. 在相同查询预算下，RAS-FRT 是否覆盖更多不同的危险参数区域？
3. 到预算 100 时，RAS-FRT 与 Transfer-UQ 的危险场景总数、区域覆盖和全库 AP 如何取舍？

本轮只做这一组必要的整体对比，不追加消融矩阵。

---

# 20. 训练数据与标签隔离

响应编码器、历史风险排序和 RAS-FRT 超参数均只使用 A 的五组 IDM 响应进行伪目标验证与训练。D 的 FVDM 标签不进入模型训练、参数选择或场景排序初始化；在线回放时，选择器通过查询接口逐次获得所选 D 场景的 FVDM 结果。

Transfer-UQ 使用其固定的物理核与采集权重，并依据已查询反馈更新残差分布。D 全量标签仅在预算回放记录全部冻结后用于离线评价。由于 D 已完整执行，本轮是保留这种标签隔离机制的回顾性离线回放，不能将它描述为预先注册的盲测。

---

# 21. 数据量与模型规模

当前历史 IDM 参数配置数量有限，第一版采用小模型：

```text
Scene Encoder:
4 → 64 → 128 → 64 → 32

M source heads:
32 → 1 each
```

暂时不要引入：

- Set Transformer history memory；
- large cross-attention；
- TabPFN；
- GP / GPC；
- Neural Process；
- Meta-RL；
- 新的 PPO/SAC 训练。

只有小模型已经显示明确不足，才升级。

---

# 22. 与 FST 文献的关系

FST：

\[
\text{surrogate responses}
\rightarrow
\text{learn scenario representativeness}
\rightarrow
\text{weighted few-shot evaluation}
\rightarrow
\text{overall crash-rate estimate}.
\]

本文：

\[
\text{historical multi-configuration responses}
\rightarrow
\text{learn response similarity}
\rightarrow
\text{few-shot target residual correction}
\rightarrow
\text{active failure-region testing}.
\]

共同点：

- 使用多个 prior / surrogate systems；
- 利用场景之间的关系；
- 工作在严格测试预算下；
- 避免纯随机测试。

区别：

- FST 的测试集在 target 测试前固定；
- 本文逐次使用 target feedback；
- FST 目标是总体事故率估计；
- 本文目标是 failure discovery + failure-region approximation；
- 本文显式处理 historical untested target coordinates。

本方案采用单点风险与覆盖缺口的加性选例分数，不再以 FST 的 reference-mass / representative-coverage 目标函数作为主方法。文献关系用于界定研究问题，不作为该分数有效性的证据。

---

# 23. 当前代码与 A/D 资产

```text
methods/ras_frt/protocol.py             # 冻结历史库 A 的校验
methods/ras_frt/banks.py                # 读取 IDM 历史响应
methods/ras_frt/training.py             # A 上的响应模型与历史排序参数
methods/ras_frt/response_encoder.py     # 响应模型
methods/ras_frt/coverage_selector.py    # HistoryRank、RAS-FRT 与查询接口
methods/ras_frt/transfer_uncertainty.py # Transfer-UQ 对比方法
methods/ras_frt/d_experiment.py         # D 上的预算回放和评价
```

历史库 A 位于 `results/method_chains/ras_frt/historical_idm/`；D 位于 `results/method_chains/ras_frt/s01_uniform_fvdm_speed_23_mps_confirmation/`。复现实验的入口为 `conda run -n metadrive python -m methods.ras_frt.d_experiment`。结果目录包含冻结的对比协议、逐次查询记录、预测分数、汇总指标和 D 上全部 116 个危险场景的显式清单。

---

# 24. 开发实验执行顺序（B=100）

1. 校验 A 的 2048 个历史场景、五组 IDM 响应、五个冻结模型和内容哈希。
2. 校验 D 的 2048 个场景、固定 FVDM 构建指纹和完整执行结果；检查 A、D 场景坐标无重合。
3. 冻结预算、随机种子、对比方法、发现曲线和危险区域覆盖指标。
4. 用同一个逐次查询接口回放四种方法，每次只公开一个被选场景的 FVDM 结果，最多 100 次。
5. 用 D 的全部真实结果离线统计每个预算点发现的危险场景数、覆盖的危险参数网格单元数和全库 AP；保存危险场景清单以便逐个检查。

---

# 25. 必须保留的旧负结果

共享2048 candidate protocol 中：

```text
HistoryRank D@50 = 50
FM²-FBT D@50 = 36
```

该结果必须保留，不能覆盖。

新协议回答的是：

> historical untested target scenarios 上的知识迁移。

旧协议回答的是：

> 相同候选库上的历史风险优先级。

二者不是同一个实验问题；本轮历史来源重新定义为同一 IDM 模型的五组参数，目标也换成完整 FVDM 控制器。旧 `D@50=50` 不能当作新设置下 HistoryRank-adapted 的成绩，本轮无需运行旧故障注入构建。

---

# 26. 核心技术假设

本文依赖：

> **同一 IDM 模型的多组参数响应结构，对不同控制律的 FVDM 目标仍具有一定局部可迁移性。**

但不要求：

> target 是 historical response 的精确线性组合。

Target residual correction 专门处理 historical prior 与 target response 的局部偏差。

若 FVDM 目标与 IDM 历史配置都没有可迁移关系，则历史知识可能没有帮助；这应当作为实验结果，而不是被方法结构掩盖。

---

# 27. 开发实验完成判据与解释

原始对比的完成条件是：A 中五组 IDM 的 2048 场景历史响应和冻结模型可校验；D 中 2048 条 FVDM 实测响应全部有效；四种原始方法的 100 次预算回放记录完整；发现数、多样性和全库 AP 能从逐次记录与真实标签重建。第 32 节的融合方法有独立协议、逐次记录和评价文件。

结果分别回答发现速度、危险区域覆盖和 100 次时的总发现数。结论限于 S01 Cut-In、当前 A 历史来源和这一固定 FVDM 配置。由于 D 全量响应已预先取得，查询预算结果是离线回放比较，不能当作本次实验实际减少了 FVDM 仿真执行成本。

---

# 28. 论文贡献候选

当前结果支持的论文贡献候选及其边界。融合方法的权重是在查看 D 的早期结果后确定的；第 33 节仍使用同一个 D，因此属于扩展开发比较，不能称为独立确认：

## Contribution 1 — Historical Multi-Configuration Response Representation

把同一 IDM 控制器多组参数配置的完整场景响应作为新 FVDM SUT 的历史先验来源。当前证据限于五组 IDM 配置、S01 和一个 FVDM 参数配置。

## Contribution 2 — Few-Shot Target Residual Adaptation

通过逐次 FVDM 反馈修正历史风险先验。D 上的离线回放显示，历史经验有助于早期查询；但 D 已完整执行，不能把目标标签查询上限当作实际仿真调用次数。

## Contribution 3 — Uncertainty-Aware Risk and Coverage Search

在唯一候选库 D 上，融合方法用 100 次查询平均找到 79.8/116 个碰撞，高于 Transfer-UQ 的 70.0、纯目标 GP-UCB 的 71.8 和原 RAS-FRT 的 62.2。到 200 次时，纯目标 GP-UCB 平均找到 115.4/116，融合方法与 Transfer-UQ 都为 114.2/116，接近有限候选库的发现上限。融合权重在查看 D 的早期结果后确定，因此这是同一库上的开发证据；现阶段不能主张独立样本上的显著优势或多种物理失效机理。

---

# 29. 推荐标题

首选：

**Historical Response Transfer for Few-Shot Failure-Region Testing of Autonomous Driving Systems**

备选：

- **Response-Adaptive Few-Shot Testing for Autonomous Driving Systems**
- **From Historical Responses to Target Failure Regions: Few-Shot Testing of Autonomous Driving Systems**
- **Few-Shot Failure-Region Testing with Historical IDM Configuration Responses**

---

# 30. 当前统一表述

本文不再研究：

> 如何在历史场景库中重新挑出过去最危险的测试点。

也不再以：

> PatternCard Transformer

作为主要创新。

本文最终研究：

> **已有算法的历史测试结果为新算法在未测场景上提供初始风险先验；新算法的少量真实测试不断纠正这个先验并更新迁移不确定性；测试方法随后选择当前预测风险高、可能补充危险区域且能提供有效校正信息的场景，以在有限预算下发现并近似刻画新算法的失效区域。**

技术框架：

\[
\boxed{
\text{Historical Prior}
\rightarrow
\text{Target Feedback Correction and Uncertainty}
\rightarrow
\text{Risk-Aware Failure-Coverage Search}
}
\]

实验框架：

\[
\boxed{
\text{Single-Target Full-Bank Evaluation}
+
\text{Nine-Method Independent Confirmation}
}
\]

目标预算：

\[
\boxed{B=200}
\]

最终必须始终区分：

- **真实执行确认的 failures**；
- **模型预测的 failure region**；
- **尚未经过目标执行确认的区域**。

第 31～32 节是 D 上 100 次预算的开发结果；第 33 节在同一 D 上扩展到 200 次和九种方法。两轮都是完整取得 FVDM 真值后的离线逐次反馈回放，200 次是选择器可见的目标标签上限，并非本轮实际 FVDM 执行次数。

---

# 31. A→D 原始四方法实验结果

## 31.1 D 的危险场景真值

D 上 2048 次 FVDM 执行全部有效，发现 **116 个危险场景**，总体碰撞率为 **5.66%**。危险场景清单、完整 FVDM 响应、冻结协议和候选场景见[D 实验目录](../results/method_chains/ras_frt/s01_uniform_fvdm_speed_23_mps_confirmation/README.md)。116 个场景覆盖四维等宽网格中的 33 个单元。这个网格表示参数空间分布，不等同于 33 种独立的物理碰撞机理。

## 31.2 受限预算下的发现与多样性

下表为每个预算点已发现的真实危险场景数，以及 100 次预算内发现的危险网格单元数。随机、RAS-FRT 和 Transfer-UQ 各运行 5 个种子；HistoryRank-adapted 确定性运行 1 次。数值为均值 ± 标准差。

| 方法 | 危险场景 @10 | @30 | @50 | @100（占全部116个） | 危险网格单元 @100（占33个） | 全库 AP |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.2 ± 0.4 | 1.2 ± 1.1 | 2.4 ± 2.3 | 4.6 ± 2.9（4.0%） | 4.2 ± 2.2（12.7%） | — |
| HistoryRank-adapted | 2 | 3 | 4 | 32（27.6%） | 20（60.6%） | 0.416 |
| RAS-FRT | **4.8 ± 1.3** | **16.6 ± 1.7** | **30.4 ± 0.9** | 62.2 ± 2.6（53.6%） | **28.4 ± 2.1（86.1%）** | 0.720 ± 0.024 |
| Transfer-UQ | 0.0 ± 0.0 | 9.6 ± 0.9 | 28.6 ± 1.5 | **70.0 ± 0.7（60.3%）** | 24.0 ± 0.7（72.7%） | **0.915 ± 0.006** |

在原始四方法中，RAS-FRT 在早期预算点发现更多危险场景，并在 100 次查询内覆盖最多的危险网格单元；到预算 100 时，Transfer-UQ 找到的危险场景总数更多，全库 AP 也更高。这是设计第 32 节融合方法的依据。两项指标回答不同问题，应同时报告。

## 31.3 复现与解释边界

逐次选择记录、方法预测、预算曲线和汇总指标见 [`method_replay.json`](../results/method_chains/ras_frt/s01_uniform_fvdm_speed_23_mps_confirmation/method_replay.json)、[`method_predictions/`](../results/method_chains/ras_frt/s01_uniform_fvdm_speed_23_mps_confirmation/method_predictions/) 和 [`method_evaluation.json`](../results/method_chains/ras_frt/s01_uniform_fvdm_speed_23_mps_confirmation/method_evaluation.json)。D 上全部 116 个危险场景另存于 [`all_fvdm_dangerous_scenarios.jsonl`](../results/method_chains/ras_frt/s01_uniform_fvdm_speed_23_mps_confirmation/all_fvdm_dangerous_scenarios.jsonl)。

这些是已完整执行 D 后的离线回放结果：查询器看不到未查询 FVDM 标签，评价器在回放结束后使用全量真值。它们量化了不同策略在同一个真实标注库上、预算最多为 100 时的反事实选择表现；完整 D 真值的取得本身使用了 2048 次物理执行。若要主张一次新的部署仅需 100 次仿真就能找到这些场景，还需按在线流程运行新的盲测库，并在全部查询结束后才补齐真值。

---

# 32. RAS-FRT-UQ：历史响应与迁移不确定性的探索性融合

## 32.1 选择机制

该方法保留 A 上训练的五组 IDM 响应模型及联合相似度 (S_H)，并用目标 FVDM 已查询结果的残差 (y-p_H) 更新物理核上的高斯后验。设 (p_{\mathrm{loc},t}(z)) 是原 RAS-FRT 的局部残差校正风险，(p_{\mathrm{GP},t}(z)) 是高斯残差后验风险，(v_t(z)\in[0,1]) 是剩余后验方差，则当前查询风险为

\[
p_t(z)=v_t(z)p_{\mathrm{loc},t}(z)+[1-v_t(z)]p_{\mathrm{GP},t}(z).
\]

反馈较少的位置由局部历史响应校正主导；后验方差下降的位置更多使用目标残差模型。令 (g_t(z)=1-\max_{q\in Q_t}S_H(z,q)) 表示与已查询场景的覆盖缺口，(m_t(z)=1) 表示该候选所在的四维参数网格尚无已确认碰撞，(I_t(z)) 是 Transfer-UQ 的加权方差减少量。查询分数为

\[
a_t(z)=0.95\{0.8p_t(z)+0.2g_t(z)+0.2p_t(z)m_t(z)\}+0.05I_t(z).
\]

首次查询尚无覆盖缺口；尚未发现碰撞时不启用新网格奖励。网格只由已知场景坐标构造，(m_t) 只用已查询碰撞结果更新。选择器从不读取未查询的 FVDM 标签。这里的方差减少项是未来信息价值的计算近似，不是精确的多步前瞻；二元碰撞标签的高斯残差方差也不构成校准后的置信保证。

## 32.2 同一 S01 D 库上的结果

五个模型种子、每次 100 个不同查询、与原始四方法相同的 A 和 D。下表列出原方法和主要对照；完整基线见第 31 节。数值为均值 ± 样本标准差。

| 方法 | 危险场景 @10 | @30 | @50 | @100（共116个） | 危险网格 @100（共33个） | 全库 AP |
|---|---:|---:|---:|---:|---:|---:|
| RAS-FRT | 4.8 ± 1.3 | 16.6 ± 1.7 | 30.4 ± 0.9 | 62.2 ± 2.6 | 28.4 ± 2.1 | 0.720 ± 0.024 |
| Transfer-UQ | 0.0 ± 0.0 | 9.6 ± 0.9 | 28.6 ± 1.5 | 70.0 ± 0.7 | 24.0 ± 0.7 | 0.915 ± 0.006 |
| **RAS-FRT-UQ** | **6.0 ± 0.0** | **20.0 ± 0.0** | **37.0 ± 0.7** | **79.8 ± 1.3（68.8%）** | **31.8 ± 0.4（96.4%）** | **0.924 ± 0.015** |

在当前 D 库上，融合方法的平均发现数在四个预算点均最高，100 次时平均找到 79.8 个真实碰撞场景，并覆盖 31.8/33 个危险网格。换成每维 3 档和 5 档的网格后，融合方法分别覆盖 13/15 和 42.8/57 个真实危险单元；原 RAS-FRT 为 12.6/15 和 39.4/57。全库 AP 包含已查询场景，只作辅助诊断；融合方法对各自未查询剩余场景的 AP 为 0.473 ± 0.091，不宜与其他方法不同的剩余场景直接比较。

完整的[`fusion_protocol.json`](../results/method_chains/ras_frt/s01_uniform_fvdm_speed_23_mps_confirmation/fusion_protocol.json)、[`fusion_replay.json`](../results/method_chains/ras_frt/s01_uniform_fvdm_speed_23_mps_confirmation/fusion_replay.json)、[`fusion_evaluation.json`](../results/method_chains/ras_frt/s01_uniform_fvdm_speed_23_mps_confirmation/fusion_evaluation.json)和[`fusion_predictions/`](../results/method_chains/ras_frt/s01_uniform_fvdm_speed_23_mps_confirmation/fusion_predictions/)可复现逐次轨迹、真实反馈、风险预测与汇总值。

**证据边界：** 融合机制与 0.2、0.2、0.05 的权重是在已查看 D 基线结果后探索确定的。它在当前 S01 D 库上是开发结果，不能作为同一 D 上预先冻结的独立验证，也不能外推至其他 SUT 或场景族。第 33 节在同一 D 上扩展预算和对照方法，沿用这一证据边界。

---

# 33. 同一 D 库上的 200 次预算与九方法比较

## 33.1 执行条件与信息边界

本节沿用第 31～32 节的**同一个 D**：scrambled Sobol 种子 `43105` 产生 2048 个 S01 场景，固定 FVDM 在 `FBRTUnifiedEnv` 中执行后得到 116 个真实碰撞（5.66%）。执行器使用 `highway-env` 1.9.1 的道路与车辆动力学，仿真频率为 20 Hz。没有生成第二个正式候选库。A 与 D 的场景坐标不重合。

完整 D 已执行 2048 次 FVDM，用来建立评价真值。九种方法只通过逐次查询接口看到自己选中场景的标签，每种方法最多查询 **200** 个不同场景；完整标签仅供事后评价。因而 200 是回放中的可见标签预算，不是这次实验实际使用的 FVDM 仿真次数。九种方法共享 D 的候选点、SUT、标签接口和 10、30、50、100、150、200 六个检查点。

除原有的 Random、HistoryRank-adapted、RAS-FRT、Transfer-UQ、RAS-FRT-UQ，还比较四个对照：`Farthest-First` 按场景参数选分散点；`Target-GP-UCB` 只用已查询的 FVDM 标签；`History-only` 只按 A 的先验风险排序；`Residual risk-only` 保留 RAS-FRT 的残差校正而去掉覆盖奖励。GP-UCB 将二元碰撞视为带噪数值反馈，不继承原算法的理论置信保证。HistoryRank-adapted 是一次确定性运行，其余方法各运行五个固定种子。

## 33.2 碰撞发现与查询效率

下表给出已发现的**真实碰撞场景数**，五种子方法报告均值；全库分母固定为 116。逐种子记录和标准差见 [D 的 200 次比较结果](../results/method_chains/ras_frt/s01_uniform_fvdm_speed_23_mps_confirmation/README.md)。

| 方法 | @10 | @30 | @50 | @100 | @150 | @200 |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.2 | 1.2 | 2.4 | 4.6 | 7.8 | 10.8 |
| Farthest-First | 2.0 | 4.2 | 5.8 | 9.6 | 12.4 | 16.0 |
| HistoryRank-adapted | 2 | 3 | 4 | 32 | 72 | 77 |
| History-only | 6.2 | 17.0 | 28.2 | 54.0 | 79.2 | 92.6 |
| Residual risk-only | 6.8 | 20.0 | 33.8 | 65.6 | 83.8 | 95.6 |
| RAS-FRT | 4.8 | 16.6 | 30.4 | 62.2 | 83.0 | 93.2 |
| Transfer-UQ | 0.0 | 9.6 | 28.6 | 70.0 | 103.0 | 114.2 |
| Target-GP-UCB | 0.6 | 15.6 | 31.8 | 71.8 | 106.8 | **115.4** |
| RAS-FRT-UQ | 6.0 | **20.0** | **37.0** | **79.8** | **109.6** | 114.2 |

**查询碰撞率**是已发现碰撞数除以已查询数，**碰撞召回率**是已发现碰撞数除以 D 上全部 116 个碰撞。下表由平均发现数计算。

| 方法 | @100 查询碰撞率 | @100 碰撞召回率 | @200 查询碰撞率 | @200 碰撞召回率 |
|---|---:|---:|---:|---:|
| Random | 4.6% | 4.0% | 5.4% | 9.3% |
| Farthest-First | 9.6% | 8.3% | 8.0% | 13.8% |
| HistoryRank-adapted | 32.0% | 27.6% | 38.5% | 66.4% |
| History-only | 54.0% | 46.6% | 46.3% | 79.8% |
| Residual risk-only | 65.6% | 56.6% | 47.8% | 82.4% |
| RAS-FRT | 62.2% | 53.6% | 46.6% | 80.3% |
| Transfer-UQ | 70.0% | 60.3% | 57.1% | 98.4% |
| Target-GP-UCB | 71.8% | 61.9% | **57.7%** | **99.5%** |
| RAS-FRT-UQ | **79.8%** | **68.8%** | 57.1% | 98.4% |

融合方法在 100 次查询时平均比 Transfer-UQ 多找到 9.8 个碰撞，比纯目标 GP-UCB 多 8.0 个；五个相同种子上的优势方向均一致。到 200 次时，纯目标 GP-UCB 略高于融合方法（115.4 对 114.2），三种较强方法都接近 D 的 116 个碰撞上限。融合方法的收益主要体现在 50～150 次的中期预算。原 RAS-FRT 在 100、200 次时均低于 `Residual risk-only` 的碰撞发现数，因而原覆盖项没有显示出事故发现收益。

## 33.3 四档网格覆盖与证据边界

每个 S01 参数轴等宽分成 **4 档**；D 的 116 个碰撞占据 33 个四维网格。**危险网格覆盖率**等于已找到碰撞所在的网格数除以 33。括号给出五次运行的平均网格数；HistoryRank-adapted 只运行一次。

| 方法 | @100 危险网格覆盖率 | @200 危险网格覆盖率 |
|---|---:|---:|
| Random | 12.7%（4.2/33） | 27.9%（9.2/33） |
| Farthest-First | 29.1%（9.6/33） | 47.3%（15.6/33） |
| HistoryRank-adapted | 60.6%（20.0/33） | 90.9%（30.0/33） |
| History-only | 83.6%（27.6/33） | 98.2%（32.4/33） |
| Residual risk-only | 83.6%（27.6/33） | 98.2%（32.4/33） |
| RAS-FRT | 86.1%（28.4/33） | 97.0%（32.0/33） |
| Transfer-UQ | 72.7%（24.0/33） | **100%（33.0/33）** |
| Target-GP-UCB | 73.9%（24.4/33） | **100%（33.0/33）** |
| RAS-FRT-UQ | **96.4%（31.8/33）** | **100%（33.0/33）** |

融合方法在 100 次时同时取得最多碰撞和最高四档网格覆盖，但这两项比较都发生在用于设计融合权重的 D 上。五个种子共享同一个 D，不能充当五个独立的场景库重复。因此本节支持同一 S01/FVDM 候选库上的开发效果，不能据此宣称跨场景库统计显著、在线仿真节省或多种物理失效机理。四档网格表示参数空间分布，不等于物理机理分类。
