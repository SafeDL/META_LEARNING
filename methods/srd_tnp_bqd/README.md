# SRD-TNP-BQD

原 S01 单一 cut-in 基准的唯一论文实现。运行参数为 [s01.yaml](configs/s01.yaml)，场景和控制器由 [s01.py](s01.py)及共享基准原协议固定。当前设置与结果见[实验汇总](../../results/srd_tnp_bqd/report.md)。

默认主线统一为：匹配历史源 → 冻结 TNP 均值与局部历史风险 → A 内独立均值读出 → 联合均值校准与多尺度非平稳差异 GP → Bayesian QD。采集保持一轮风险档案预期改进、两轮后验风险均值循环。

原 A/D 各 2048 个 S01 场景、原 FVDM 目标和 ego 初始 25 m/s 不变。原五个期望速度 27 m/s 的 IDM 历史源用于冻结 h；均值使用原 A 上补充的期望速度 23 m/s、制动上限 8 m/s² 的 IDM 源。源规格仅按已知目标期望速度和制动上限确定，未用 D 标签选源。

`m=a*m_TNP+b*m_local+c`。局部风险取归一化四维坐标的 8 个近邻，以 Matérn-5/2 相关性加权连续 R 后转 logit。仅在 A 内训练三个读出系数：1638 个训练点四折交叉预测、410 个验证点、1000 步固定训练，风险重建与尾部成对排序损失，以 A 的 R>0.5 排序 AP 选择检查点。它是场景局部相关性读出，尚未实现通用 SUT 相似度编码。

在线使用 `z=offset+scale*m+discrepancy`；offset/scale 先验均值 0/1、方差 1/0.25，与差异 GP 联合条件化。h 保留原五源完整 context，继续参与响应关联和尺度门，历史模型及核参数冻结；目标反馈仅更新联合后验。scale 是响应系数，不是可信度概率。

|模块|职责|
|---|---|
|s01、audit、benchmark|原 A/D、控制器、四维坐标、真值隔离和基准核验|
|risk、measurements|同步 TTC/DRAC/车身间距风险和被动测量|
|tokens、scan_attention、krblock、historical|整体场景 token、完整 context 历史模型|
|train|A 内两阶段训练和外层留源开发；已有五个冻结模型直接复用|
|nonstationary_kernel、discrepancy|h 关联、尺度门、独立物理通道和 GP 后验|
|qd、acquisition|已查询风险档案、求积和第一批混合采集|
|oracle、session|逐次计费、跨进程反馈隔离|
|baseline、experiment|八个通用/文献基线、原 RAS 和当前主线统一运行|
|diagnostics、evaluate|完整碰撞真值、覆盖、排序诊断和复现核验|
|visualize|已发现碰撞的原配置 GIF 回放，逐帧核对原轨迹和风险测量|
|matched_history、mean_readout|补充源的物理响应与 A 内独立均值读出；由默认实验入口调用|

```powershell
conda activate metadrive
python -B -m methods.srd_tnp_bqd.audit
python -B -m methods.srd_tnp_bqd.train
python -B -m methods.srd_tnp_bqd.experiment
python -B -m methods.srd_tnp_bqd.experiment --online
python -B -m methods.srd_tnp_bqd.evaluate
python -B -m methods.srd_tnp_bqd.visualize
python -B -m pytest -q -p no:cacheprovider
```

默认运行新主线和九个基线的固定池回放，已完成的轨迹直接复用。匹配源响应及读出缺失时由实验入口准备；训练器仅在原冻结模型缺失时执行 A 内两阶段训练，不使用 D 选型。`--online` 对当前主线种子 11 的完整 200 次查询执行真实仿真并核对缓存响应；当前新主线成绩来自固定池回放，旧物理在线核验仅属于第一批。

输出统一在[comparison](../../results/srd_tnp_bqd/comparison/README.md)，当前方法名称为 **SRD-TNP-BQD**。均值校准单独试验、原 TNP 匹配均值试验、前100次只选风险试验及无收益读出不再提供运行分支，指标和实际成本集中在[历史开发摘要](../../results/srd_tnp_bqd/reference/development_history.json)。

可视化入口从当前主线种子 11 的前50次查询中取三个不同失效 cell 的碰撞，逐帧核对原轨迹和风险。已有[第一批 GIF](../../results/srd_tnp_bqd/reference/first_batch/visualizations/README.md)作为历史示例保留；本次清理没有新增可视化物理执行。

当前主线 F50/F100/F200 为40.2/74.2/113.6，召回97.93%、失效 cell 覆盖99.39%；第一批为25.4/60.8/109.4。本文 F50 高于九个正式基线，F100 和最终碰撞召回仍略低于 RAS-FRT-UQ。原始测量、模型、九个基线与[第一批参照](../../results/srd_tnp_bqd/reference/first_batch/README.md)保留；旧消融不能作为当前主线消融。

正式比较共九个基线。下表中的代表性方法采用描述名称；文献方法采用原名称，并明确当前适配范围。

|名称|依据|当前实现|
|---|---|---|
|Uniform random|代表性随机测试|在 D 无放回均匀抽样|
|Farthest-first|[Gonzalez，1985](https://www.sciencedirect.com/science/article/pii/0304397585902245)|四维归一化参数的最远优先遍历；首点由种子决定|
|kNN historical-risk ranking|代表性历史邻域回归|每个历史源取 8 个最近邻风险均值，再对五源平均；确定性排序一次|
|GP-UCB|[Srinivas 等，ICML 2010](https://icml.cc/Conferences/2010/papers/422.pdf)|连续 R 的物理 Matérn-5/2 GP；有限候选集 UCB|
|GP-EI|[Jones、Schonlau、Welch，1998](https://link.springer.com/article/10.1023/A:1008306431147)|同一物理 GP；改进基准为已测最大 R|
|BOP-Elites|[Kent 等，TEVC 2025](https://wrap.warwick.ac.uk/id/eprint/183754/)|已知参数 cell 的候选池适配；区域归属概率为 1，按各 cell 档案阈值计算 EI|
|BAS|[Sinha 等，CoRL 2024／PMLR 2025](https://proceedings.mlr.press/v270/sinha25a.html)，[作者论文](https://amansinha.com/docs/SinhaNiPaWh24.pdf)|论文单保真 Bayesian adaptive sampling；最大化全 D 池预期 Bernoulli 方差下降，事件为 R>0.5|
|RF surrogate BO|[Huang、Sun、Tian，T-ITS 2025](https://ieeexplore.ieee.org/document/10759101/)|RF 均值、十折删除 Jackknife 方差、EI；固定参数和固定池枚举适配|
|RAS-FRT-UQ|本仓库已有方法，Git cc7c1b32|原四维模型、原二值碰撞反馈、原融合权重；[实现](../ras_frt_uq/README.md)|

BAS 实现作者论文附录 A.2 的单步预期阈值事件方差下降；在均匀的 2048 点池上积分，完整使用候选协方差。这里没有把五种历史控制器当成同一目标的仿真保真层，因此不命名为 BAMS。RF 用 50 棵树、深度 10、最小分裂数 2；省略原论文的 TPE 调参与连续 DE 搜索，不能报告为完整作者程序复现。BOP-Elites 使用已知参数描述符，未训练论文中的未知描述符 GP。

四个 GP 基线和 RF 均采用同种子共有的 10 个随机初始查询，计入 200 次预算。GP 固定均值 0.5、方差 0.25、长度尺度 0.2、噪声 0.001、jitter 1e-8；UCB 的 delta 为 0.1。全部参数在本轮比较前固定。本文和八个通用／文献基线只用已披露连续 R 选例，RAS 保留原二值反馈。完整 D 碰撞标签只由计费 oracle 和评价器持有。

碰撞召回的分母为 D 的 116 个真实碰撞场景；碰撞 cell 覆盖的分母为 D 的 33 个真实失效 cell。每维划分 4 档，共 256 个几何 cell，但 256 不作为碰撞覆盖分母。风险 QDScore 为各 cell 已测最大正质量 max(R−0.5,0) 的总和，与碰撞 cell 覆盖分别报告。原 D 曾参与 RAS 融合开发，当前比较属于同库开发证据。
