# 配对系统主动评估：代理校正与覆盖选择

本方法用历史系统构造风险代理，再结合已查询的目标反馈校正局部风险，利用覆盖信息选择一批值得测试的场景。每批执行完成后更新目标模型。下文 CV 指控制变量校正，MC 指蒙特卡罗采样，MI 指互信息。

原文：Anjali Parashar等，*Coverage Aware Active Evaluation for Failure Discovery with Paired Systems*，[arXiv:2608.13719v1](https://arxiv.org/abs/2608.13719v1)，2026年8月13日。全文优先取自[用户目录](https://drive.google.com/file/d/1MZnLp9iqhZ1ogucTboDaJon2U73u6yEi/view)，核查第3–5节、算法1及附录A、C、D、I中的推导、局部区域、支持模型和实验设置。复现记录中未定位到作者实现，代码核对采用全文公式和独立数值例。

## 原文与实现对应

| 原文 | [实现](method.py) | 机制 |
|---|---|---|
| §4.1，式(2)，附录A | `control_variate` | 局部配对代理/目标预测样本，加独立代理样本；保留 `k/(k+n)`、代理方差及目标/代理协方差，允许负相关 |
| §4.2，式(3) | `support_information` | 计算支持状态与支持发现变量间的互信息，使用等价形式 `H(R)-E_Z H(R|Z)` |
| 附录D，式(19)–(24) | `support_model` | K-means、簇RMS半径、新支持状态、距离权重、数据自适应新颖性半径与软化系数 |
| 附录C，式(17)–(18) | `choose_batch` | 由MC均值梯度/标准差决定自适应局部区域半径 |
| §4.3、算法1 | `choose_batch`、`run` | 先取高互信息候选，再按CV风险阈值筛选，再聚类取中心附近的批量候选；批次完成后才更新目标代理 |
| 附录I.1、表6的nuPlan设置 | `TargetSurrogate` | 96→24→6、ELU、LayerNorm、dropout=0.1，Adam 0.001，1000个epoch |

预测阶段通过 MC dropout 采样风险，CV 计算使用这些随机样本；采样时保持模型参数固定。数值测试同时核对有正/负相关及恒定代理的CV、用联合分布直接计算的互信息、重复几何、零/全碰撞、批次边界和预算。

## 本项目的数据与预算适配

- 风险量为二值碰撞标签，对应碰撞概率阈值0.5，使用二元交叉熵拟合目标代理。原文 nuPlan 任务使用连续 TTC 测量及384维输入，本项目使用公开场景参数编码。
- 代理来源固定为允许历史对象的响应预测器等权平均，带宽0.25，使用项目既有 `HistoricalPredictor`。来源范围在目标查询前确定。历史代理采用已完成测试记录的核平滑预测，构建时无新增仿真调用；原文附录 I.2 也讨论了使用代理模型近似代理执行的方式。
- [构造函数](../../../experiments/domain_paired_validation.py)保存来源标识、档案指纹、构造时间；新增代理执行为0。代理按传入的历史来源范围构建；当前正式比较使用全来源数据。
- 使用共同随机初始化8次及128次总预算；之后每批5次，共24批。初始化及批次结果只有在付费后进入目标代理；已选出的同批候选不提前使用批内尚未完成的标签。
- 原文未给出的项目值固定为：MC预测32次、高MI候选64个、局部最多64个样本、其中最多16个配对预测样本；K在1至最多10且不超过已观测点数一半/不同几何数的范围内用弦线肘点选择。附录D未给出 `q_k` 的唯一函数，本项目采用 `0.01*(1-w_k)`，保持其靠近簇中心为0、远处截至小常数的条件。
- 局部平方距离阈值下界取公开库最近邻平方距离的中位数，上界取公开库最大平方距离。至少4个公开邻居保证局部样本协方差可估计。附录D的新颖性半径/软化系数另使用已观测支持的欧氏最近邻距离中位数/标准差，与局部平方距离半径区分。
- 原文未定义阈值筛选后不足一批的处理。本项目按固定预算要求，以同一高MI候选集中的CV风险降序补齐；若不同几何数不足批量大小，先覆盖不同几何，再选不同候选索引补齐。两种补齐量逐次记录，不免费查询标签判定补齐对象。
- 目标代理按批次从当前参数继续训练，使用全部已付费碰撞标签。概率、CV风险、CV系数、支持互信息、簇数及批次标识随查询保存。CV 风险是校正估计量，可超出[0,1]；预测概率与校正风险分别保存。

这些是有限候选库、二值风险及历史代理条件下的适配；局部代理校正、支持感知互信息和批量多样性三项核心均保留。

## 验证入口

```powershell
conda activate metadrive
python -B -m pytest tests/experiments/test_paired_system_active_evaluation.py -q -p no:cacheprovider
python -B -m experiments.domain_paired_validation
```

原始执行和完整候选库的碰撞分母由评估端读取。开发期验证记录位于 `experiment_outputs/paired_system_active_evaluation/collision/discovery/domain_comparison/stage_a/`。开发比较和独立验证均已完成，独立验证包含全来源条件下的200项运行，结果见 [独立验证报告](../../../experiment_outputs/benchmark/collision/discovery/independent_validation/report.md)。
