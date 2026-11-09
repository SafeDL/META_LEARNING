# Indago：历史分类器与显著性搜索

本目录保留 Indago 的历史档案适配，供补充对照及原配置复现。分类器由历史碰撞标签训练，查询阶段保持固定，再用输入梯度指导变异和爬山搜索。

依据 Biagiola、Tonella 的 *Testing of Deep Reinforcement Learning Agents with Surrogate Models*，TOSEM 2024，[DOI](https://doi.org/10.1145/3631970)，[作者全文](https://matteobiagiola.github.io/assets/pdfs/testing-deep-reinforcement-learning-agents-with-surrogate-models.pdf)。核对正文第 4–5 节及[作者实现](https://github.com/matteobiagiola/drl-testing-experiments)的 `ts` 分支，提交 `b1afed834f2a82d6f8a267217b31f8e375855ed1`。

| 依据 | 实现 | 作用 |
|---|---|---|
| §4.1、§5.2.1，类别权重式(1) | [FailurePredictor、train](method.py) | MLP 分类器与逆类别频数加权交叉熵 |
| §4.2 | `FailurePredictor.probabilities` | 双类别 softmax 的失败概率 |
| §4.3，算法1 | [PoolSearch.propose](pool.py) | 随机起点、邻域比较与爬山 |
| §4.3.1，作者 `mutate_hot` | `PoolSearch.saliency` | 以失败类 logit 梯度的绝对值加权抽样坐标，并保留梯度符号 |
| 合法参数变异 | [SearchSpace.mutate](mutation.py)、`PoolSearch.project` | 变异后投影到最近未查询候选 |

## 历史档案适配

原文使用目标策略训练过程中的测试记录；本项目使用八个历史 ADS 的参数和碰撞响应，按功能场景分别训练。相同坐标的记录放在同一训练或验证分区。

默认分类器为四层 32 单元 MLP，使用 BatchNorm、ReLU 和 dropout 0.5；Adam 学习率 0.0003、批量 64，训练 20 轮，以 20% 历史验证集的交叉熵选择模型。类别权重为 `N/(2*class_count)`；单类别档案保留两个输出，使用观测类别训练。

搜索沿用作者实现的梯度加权抽样方式。默认邻域 50、最多 100 轮、停滞 20 轮、概率变化阈值 0.005。连续和类别参数先合法变异，再按归一化距离投影到未查询候选；同分按场景 ID。每次执行后移除该候选，初始化计入统一预算。

## 构建与验证

[method.py](method.py)提供 `train/load`，[pool.py](pool.py)提供 `run`。统一构建通过 `historical_classifiers(history, destination, names=('indago',))`，模型位于 `artifacts/runs/joint_response/pilot/models/<场景>/indago/classifier.pt`。

~~~powershell
python -B -m pytest tests/experiments/test_indago_pool.py -q
~~~

测试覆盖梯度方向及抽样频率、合法变异、固定分类器、目标反馈隔离和查询预算。已有结果见[原配置固定历史报告](../../../artifacts/runs/joint_response/fixed/reports/results.md)。
