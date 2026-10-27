# 共享概率模型与推断细节

本节给出[历史响应迁移方法](../history_transfer/paper_section.md)使用的场景核、伯努利高斯过程推断及训练设置，可作为方法论的数学补充。高斯过程分类、Laplace 近似与深核构造采用标准建模方式；本文通过历史响应与目标反馈的组织方式，将这些计算用于跨系统碰撞场景发现。

## 场景表示与核函数

对第 $q$ 个连续场景参数 $u_q$，利用预先确定的物理范围 $[l_q,h_q]$ 得到 $x_q=(u_q-l_q)/(h_q-l_q)$；类别变量按各自允许的取值进行独热编码，拼接为输入 $\mathbf x$。这一表示只依赖执行前可知的场景参数。

联合响应模型和目标基础模型分别使用具有独立参数的深核。记任一模型的特征映射为 $e_\vartheta(\mathbf x)$，定义

$$
\begin{aligned}
k_\vartheta(\mathbf x,\mathbf x')
&=\sigma_k^2\left(1+\sqrt5 r+\frac{5r^2}{3}\right)e^{-\sqrt5 r},\\
r&=\frac{\|e_\vartheta(\mathbf x)-e_\vartheta(\mathbf x')\|_2}{\ell}.
\end{aligned}
\tag{A1}
$$

这是在学习表示上计算的 Matérn-$5/2$ 核。特征映射由四个输出宽度为 $32$ 的全连接层组成，相邻层之间采用 ReLU，最后一层直接输出特征；均值、核幅值、长度尺度和网络参数共同学习。核幅值与长度尺度采用对数参数化，计算时将长度尺度下界设为 $0.01$。深网络与核函数的这种组合遵循[深核学习](https://proceedings.mlr.press/v51/wilson16.html)的基本构造。

原始场景参数模型使用相同的 Matérn 形式，但令 $e_\vartheta(\mathbf x)=\mathbf x$，并直接根据已查询标签适配均值、幅值和长度尺度。其固定初始化为均值 $0$、核标准差 $1$、长度尺度 $0.3$。它提供基于原始参数关系的预测，与基于学习表示的目标分量共同构成混合模型。

## 伯努利观测下的条件预测

下面将历史来源条件模型、独立目标模型和原始参数模型统一写为有限候选集上的高斯先验 $\mathbf f\sim\mathcal N(\mathbf m,\mathbf K)$。对已观测索引集合 $O$，使用

$$
p(\mathbf y_O\mid\mathbf f_O)
=\prod_{i\in O}\sigma(f_i)^{y_i}\bigl[1-\sigma(f_i)\bigr]^{1-y_i}.
\tag{A2}
$$

Laplace 近似以潜变量后验的众数及其局部曲率构造高斯近似，是[高斯过程分类](https://gaussianprocess.org/gpml/chapters/RW3.pdf)的标准推断方法。针对当前先验，众数为

$$
\widehat{\mathbf f}_O
=\underset{\mathbf f_O}{\operatorname{arg\,min}}\left\{
\frac12(\mathbf f_O-\mathbf m_O)^\top\mathbf K_{OO}^{-1}(\mathbf f_O-\mathbf m_O)
-\log p(\mathbf y_O\mid\mathbf f_O)
\right\}.
\tag{A3}
$$

记 $\widehat p_i=\sigma(\widehat f_i)$，并令

$$
\begin{aligned}
\mathbf W&=\operatorname{diag}\!\left(\widehat p_i(1-\widehat p_i)\right),\\
\mathbf L\mathbf L^\top&=\mathbf I+\mathbf W^{1/2}\mathbf K_{OO}\mathbf W^{1/2},\\
\mathbf a&=\mathbf K_{OO}^{-1}(\widehat{\mathbf f}_O-\mathbf m_O).
\end{aligned}
\tag{A4}
$$

候选集上的近似后验均值与协方差为

$$
\begin{aligned}
\boldsymbol\mu&=\mathbf m+\mathbf K_{\mathcal X O}\mathbf a,\\
\boldsymbol\Sigma&=\mathbf K-\mathbf V^\top\mathbf V,\\
\mathbf V&=\mathbf L^{-1}\mathbf W^{1/2}\mathbf K_{O\mathcal X}.
\end{aligned}
\tag{A5}
$$

式（A5）说明了目标反馈怎样作用于未测场景：已测点与候选点之间的协方差控制预测修正，所有已查询标签共同决定修正方向与大小。程序使用 Cholesky 分解和三角求解实现这些式子；式中逆矩阵用于数学表达。

对候选 $\mathbf x$，从 $\mu(\mathbf x)$ 和 $v(\mathbf x)=\Sigma_{\mathbf x\mathbf x}$ 计算碰撞概率时，对潜变量不确定性进行积分。设 $z_q,\omega_q$ 为 $Q$ 点 Gauss–Hermite 求积的节点与权重，则

$$
p(\mathbf x)
\approx\frac{1}{\sqrt\pi}\sum_{q=1}^{Q}
\omega_q\,\sigma\!\left(\mu(\mathbf x)+\sqrt{2v(\mathbf x)}\,z_q\right),
\qquad Q=32.
\tag{A6}
$$

当尚无目标标签时，直接对各分量的先验边际使用式（A6）。历史响应的边际方差、目标核的不确定性和已查询标签的条件化结果因而通过同一概率计算进入场景排序。

## 联合历史响应的条件化

历史模型同时使用最小间距与碰撞观测。为明确其计算顺序，将历史点和候选点上两通道的全部潜变量记为 $\mathbf z$，其联合均值与协方差为 $\mathbf m_z,\mathbf K_{zz}$；将历史记录对应的间距潜变量索引记为 $G$，变换后的间距观测记为 $\mathbf r$。先对高斯间距通道条件化：

$$
\mathbf m_z^{(d)}
=\mathbf m_z+\mathbf K_{zG}
\left(\mathbf K_{GG}+\sigma_d^2\mathbf I\right)^{-1}
(\mathbf r-\mathbf m_G),
\tag{A7}
$$

$$
\mathbf K_{zz}^{(d)}
=\mathbf K_{zz}-\mathbf K_{zG}
\left(\mathbf K_{GG}+\sigma_d^2\mathbf I\right)^{-1}
\mathbf K_{Gz}.
\tag{A8}
$$

随后取出碰撞通道，以 $\mathbf m^{(d)},\mathbf K^{(d)}$ 为先验，利用历史碰撞标签执行式（A2）至式（A5）。候选部分即为主方法中的 $\mathbf m_s,\mathbf V_s$。联合响应模型中的通道相关系数决定间距如何影响这一先验，碰撞观测进一步约束失效位置。部署保存 $\mathbf V_s$ 的对角元素；训练计算使用同样的边际方差表示。

联合核的通道矩阵 $\left[\begin{smallmatrix}1&\rho\eta\\\rho\eta&\eta^2\end{smallmatrix}\right]$ 在 $\eta>0,|\rho|<1$ 时为正定矩阵，因而与有效的场景核结合后得到有效联合协方差。历史边际方差矩阵与目标核的加和保持半正定，为来源条件推断提供一致的概率模型。

## 核学习与数值计算

目标基础核训练及在线核适配使用近似负对数边际似然

$$
\mathcal L_{\rm GP}
=\frac12(\widehat{\mathbf f}_O-\mathbf m_O)^\top
\mathbf K_{OO}^{-1}(\widehat{\mathbf f}_O-\mathbf m_O)
-\log p(\mathbf y_O\mid\widehat{\mathbf f}_O)
+\sum_i\log L_{ii}.
\tag{A9}
$$

优化时按标签数归一化。每次在线适配从固定基础参数开始，在当前全部标签上执行 $10$ 步 Adam；学习表示的目标模型学习率为 $0.003$，原始参数模型学习率为 $0.01$。核适配的证据计算采用 $10$ 轮潜变量迭代。

用于历史后验和各分量最终条件预测的求模计算采用白化坐标下带步长调整的 Newton 方法，以最大梯度绝对值 $10^{-10}$ 为收敛判据，最多迭代 $100$ 次。计算使用双精度，并在观测协方差的对角加入 $10^{-5}$ 的数值稳定项。来源权重在对数域递推，评分时将似然下界设为 $10^{-15}$，之后通过对数归一化得到权重。

下表汇总用于复现当前方法的主要设置。候选数与预算定义测试任务，模型设置定义同一反馈前缀下的预测与适配计算。

| 部分 | 设置 |
|---|---|
| 公开候选集与目标预算 | 每项任务 $N=512$，$B=128$，前 $b_0=8$ 次为共同初始化 |
| 联合响应观测 | 历史碰撞标签与最小间距；在线适配使用目标碰撞标签 |
| 深核结构 | 四个宽度为 $32$ 的全连接层，层间 ReLU；Matérn-$5/2$ 核 |
| 两阶段历史优化 | 基础核与联合响应模型各 $1500$ 步，Adam 学习率 $0.001$，梯度范数上限 $5$ |
| 联合响应初始化 | 共享核参数由已训练的基础核复制；间距均值 $m_h=0.5$、通道尺度 $\eta=0.3$、相关系数 $\rho=-0.5$ |
| 模拟目标支持长度 | $r\in\{8,16,32,64,128\}$ |
| 模拟目标查询集 | 每个片段 $32$ 个独立查询点 |
| 训练历史上下文 | 每个允许的非目标来源采样 $64$ 条记录，排除该片段全部支持与查询坐标 |
| 开发选择 | 每 $100$ 步评价；基础核按适配后查询负对数似然选择，联合响应模型按查询负对数似然选择，同值时比较 Brier 分数 |
| 目标适配 | 每轮重置后执行 $10$ 步 Adam；学习核学习率 $0.003$，原始参数核学习率 $0.01$，梯度范数上限 $5$ |
| 历史与目标的组合 | 历史后验取边际方差，目标差异核系数为 $1$ |
| 权重初始化 | 历史来源共占 $0.75$、等权分配；独立目标分量占 $0.25$，组内权重随后递推 |
| 原始参数模型 | 在最终混合中固定占 $0.1$，其余分量合计占 $0.9$ |
| 概率积分 | $32$ 点 Gauss–Hermite 求积 |

基础核训练使用历史碰撞证据，联合响应训练使用模拟目标查询损失。开发阶段使用独立开发系统及其支持、查询划分选择模型；正式目标标签不参与上述选择。控制家族指规则控制、优化规划、搜索规划或学习策略等决策机制分组。在严格留家族设置下，同时从历史训练、响应上下文和开发选择中排除目标所属控制家族，其他训练与推断规则保持一致。

## 实现对应关系

| 数学内容 | 实现入口 |
|---|---|
| 场景归一化与参数组定义 | [scenarios.py](../../experiments/scenarios.py) 中的 `features`、`dedup_key` |
| 深核、证据计算与目标适配 | [gp_classifier.py](gp_classifier.py) 中的 `ClassificationKernel`、`laplace`、`fit_target` |
| 收敛的分类后验推断 | [gp_laplace.py](gp_laplace.py) 中的 `laplace`、`posterior` |
| 碰撞概率积分 | [classification.py](classification.py) 中的 `torch_moments` |
| 联合响应核与间距变换 | [joint_response.py](../history_transfer/joint_response.py) 中的 `JointResponseKernel`、`margin_values` |
| 历史后验与训练前缀预测 | [domain_consistency.py](../history_transfer/domain_consistency.py) 中的 `historical_prior`、`query_probabilities` |
| 两阶段训练 | [domain_training.py](../history_transfer/domain_training.py) 中的 `train_target`、`train` |
| 来源条件预测 | [converged_response.py](../history_transfer/converged_response.py) 中的 `component_probabilities` |
| 权重递推与在线选例 | [domain_discovery.py](../history_transfer/domain_discovery.py) 中的 `select` |
