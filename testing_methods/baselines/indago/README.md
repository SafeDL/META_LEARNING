# Indago：分类器与显著性搜索

本方法用历史场景及碰撞标签训练分类器，再根据输入梯度指导场景变异，通过爬山搜索选择预测碰撞概率较高的候选。预算内分类器保持固定。

原文：Matteo Biagiola、Paolo Tonella，*Testing of Deep Reinforcement Learning Agents with Surrogate Models*，TOSEM 2024，[DOI](https://doi.org/10.1145/3631970)，[作者全文](https://matteobiagiola.github.io/assets/pdfs/testing-deep-reinforcement-learning-agents-with-surrogate-models.pdf)。复现依据为作者全文第4节、算法1–2、第5节实验与超参数，以及[作者复现包](https://github.com/matteobiagiola/drl-testing-experiments)，`ts` 分支、提交 `b1afed834f2a82d6f8a267217b31f8e375855ed1`。

## 论文与代码对应

| 原文/作者代码 | 本项目实现 | 保留机制 |
|---|---|---|
| §4.1、§5.2.1，类别权重公式(1)；`avf_policy.py:get_model_architecture/loss_function` | [FailurePredictor、train](method.py) | 32单元的隐藏层、BatchNorm、ReLU，多层时 dropout=0.5；双类别交叉熵及逆类别频数权重 |
| §4.2，失败概率 | `FailurePredictor.probabilities` | 双类别 softmax 的失败类概率 |
| §4.3、算法1 | [PoolSearch.propose](pool.py) | 随机起点、固定分类器、包含当前解的邻域、保留最佳概率的爬山 |
| §4.3.1；`parking_env_configuration.py:mutate_hot` | `PoolSearch.saliency` | 作者代码按绝对梯度加权抽样输入坐标，并使用梯度符号；连续参数与类别映射分开处理 |
| `avf.py:sample_test_env_configuration_hc` 的无时间限制分支 | `SearchConfig` | 邻域50、最多100轮、停滞20轮、概率变化阈值0.005 |
| §4.3.1，合法参数变异 | [SearchSpace.mutate](mutation.py)、`PoolSearch.project` | 先在合法公开场景参数上变异，再投影到未查询候选，不查看目标结果 |

作者论文文字将显著性描述为选择最大梯度坐标，而复现包实际按绝对梯度加权抽样。本实现采用作者代码的抽样方式；测试检查9:1梯度下的选择频率及梯度符号。显著性由**失败类 logit** 对输入的梯度计算。

## 历史数据与候选库适配

原文使用目标策略训练过程中积累的场景与结果。本项目使用8个历史驾驶系统的场景参数和碰撞标签训练分类器。历史系统均已完成训练，因此按场景坐标划分训练与验证数据，不使用策略训练进度过滤。

训练使用四个32单元隐藏层，对应原文 Driving 分类器的结构；使用作者训练入口默认的20个 epoch、Adam 学习率0.0003、批量64，类别权重取 `N/(2*class_count)`。采用20%历史验证，选择验证交叉熵最低的模型；同一配置在不同历史系统中的响应放在同一分区，避免相同公开坐标同时出现在训练和验证中。不使用开发或独立目标标签选择该模型。

训练函数也支持只有一个观测类别的历史档案：保留两个 logit，缺失类权重为0，唯一观测类权重为1，训练记录保存实际类别计数。当前正式 Indago 比较使用全来源数据，保存5类场景各一个分类器。

候选库中的 Sobol 配置通常不会恰好只相差一个参数。每次变异先使用项目的合法连续/类别变异，再以归一化公开特征距离投影到最近的未查询候选；同分按场景标识确定。这保留显著性指导的变异及爬山步骤，同时声明了有限候选库的投影限制。使用随机起点，不免费使用目标训练失败种子。

每次查询后从可用集合移除该候选；分类器保持固定。共同随机初始化8次计入128次预算。目标标签翻转测试核对了选择过程只由历史分类器、公开候选和随机种子决定。

## 构建、核查与结果

```powershell
conda activate metadrive
python -B -m pytest tests/experiments/test_indago_pool.py -q -p no:cacheprovider
python -B -m experiments.domain_indago_validation
```

[验证入口](../../../experiments/domain_indago_validation.py)先对作者固定提交中的网络构造函数与本项目1–4层结构进行相同输入、相同权重对比，再从切入、切出历史响应构建独立分类器，最后执行一个开发库的128次隔离付费回放。作者结构对比误差为0；模型保持固定、初始化计费、预算与不重复检查通过。结果见[审计](../../../experiment_outputs/indago/collision/discovery/domain_comparison/stage_a/audit.json)。

独立分类器由 `python -B -m experiments.independent_models classifiers` 构建，保存于 [independent_models](../../../experiment_outputs/indago/collision/independent_models/)。训练与加载函数为 `method.py:train/load`。已完成的独立比较包含全来源条件下的200项 Indago 运行，结果见 [独立验证报告](../../../experiment_outputs/benchmark/collision/discovery/independent_validation/report.md)。
