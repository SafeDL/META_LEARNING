# 配对系统主动评估

本方法用历史系统构建风险代理，用已付费目标响应进行局部控制变量校正，再通过支持互信息和聚类选择一批候选。每批执行后更新目标代理。

依据 Parashar 等的 *Coverage Aware Active Evaluation for Failure Discovery with Paired Systems*，[arXiv:2608.13719v1](https://arxiv.org/abs/2608.13719v1)，[核对全文](https://drive.google.com/file/d/1MZnLp9iqhZ1ogucTboDaJon2U73u6yEi/view)。实现按正文第 3–5 节、算法1及附录 A、C、D、I 的公式核对，以独立数值例验证。

| 原文 | [method.py](method.py) | 作用 |
|---|---|---|
| §4.1，式(2)，附录A | `control_variate` | 以局部配对样本和独立代理样本校正风险，保留方差、协方差与 `k/(k+n)` 因子 |
| §4.2，式(3) | `support_information` | 用 `H(R)−E_Z H(R|Z)` 计算支持互信息 |
| 附录D，式(19)–(24) | `support_model` | 构建簇半径、距离权重和新支持状态 |
| 附录C，式(17)–(18) | `choose_batch` | 从预测均值梯度和标准差确定局部区域 |
| §4.3、算法1 | `choose_batch、run` | 高互信息筛选、风险阈值过滤和聚类批量选择 |
| 附录I.1 | `TargetSurrogate` | 96→24→6、ELU、LayerNorm、dropout 0.1 的目标代理 |

## 历史代理与目标反馈

历史代理为允许来源的 `HistoricalPredictor` 等权平均，带宽为 0.25。构建只读取历史档案，保存来源标识、指纹和构建耗时。目标代理使用碰撞标签和二元交叉熵，以 Adam 0.001 训练 1,000 轮；各批次从当前参数继续训练全部已付费记录。

预测采用 32 次 MC dropout，固定参数后计算局部控制变量风险；校正系数允许负相关。校正风险可超出 [0,1]，与碰撞概率分别记录。原文的连续测量和输入在本项目中映射为二值碰撞与公开参数特征，也可传入 `feature_fn=domain.features`。

## 固定候选库适配

共同初始化计入总预算，之后每批 5 次。同批候选全部选定后才接收其实际反馈。默认高互信息候选为 64 个，局部样本至多 64 个，其中配对预测样本至多 16 个。

簇数在 1–10 范围内按弦线肘点选择，并受已观测点数一半和不同几何数约束。新支持权重采用 `0.01*(1−w_k)`。局部平方距离阈值以下界为公开库最近邻平方距离中位数、上界为公开库最大平方距离，并保留至少 4 个邻居；新颖性半径与软化系数另由已观测支持的最近邻距离确定。

阈值过滤后不足一批时，按同一高互信息候选集的校正风险补齐；不同几何不足时先覆盖不同几何，再补足候选索引。补齐量、局部风险、系数、互信息、簇数和批次均写入查询记录。

## 接入与验证

[build_proxy](method.py)构建历史代理，`run(candidates, oracle, proxy=..., initial_indices=..., budget=200, seed=11)`执行测试，由 [experiments.pilot](../../../experiments/pilot.py)接入。

~~~powershell
python -B -m pytest tests/experiments/test_paired_system_active_evaluation.py -q
~~~

测试覆盖控制变量数值、互信息、正负相关、重复几何和批次反馈隔离。结果见[论文报告](../../../artifacts/runs/joint_response/paper/reports/results.md)。
