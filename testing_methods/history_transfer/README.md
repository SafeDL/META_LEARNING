# 历史响应驱动的跨系统碰撞场景发现

本方法利用已有驾驶系统的碰撞和最小间距记录，预测新系统上哪些场景更容易发生碰撞；再用逐次获得的目标反馈修正预测，选择下一次测试。每类功能场景分别建立模型。

## 方法流程

1. 读取允许使用的历史与开发档案。历史数据用于训练，开发数据用于选择检查点；严格留家族时，目标所属控制方法类别同时退出这两组数据。
2. 训练碰撞预测的基础核，再训练碰撞与间距联合响应编码器。编码器训练轮流将历史系统作为模拟目标，检查它能否在给定反馈后预测新的查询点。
3. 在公开候选坐标上，为每个历史系统计算碰撞响应后验。独立验证采用各候选的历史边际方差。
4. 每轮根据已查询的目标碰撞标签适配目标模型，并更新各历史来源和独立目标分量的可靠性。权重更新使用获得该次反馈前的预测似然。
5. 将各分量的碰撞概率合并，在尚未测试的候选中选择概率最高者，直到用完预算。

历史来源最初等权；历史组与独立目标分量的初始比例为0.75/0.25，随后根据反馈更新。原始场景参数模型在最终混合中固定占0.1，代码中称为 `physical`。目标模型与原始参数模型每轮从固定初始化出发，使用全部已查询标签适配10步，学习率分别为0.003和0.01。

## 训练与运行

从项目根目录构建模型：

```powershell
python -B -m testing_methods.history_transfer.domain_training
python -B -m experiments.independent_models classifiers
python -B -m experiments.independent_models priors
```

第一条命令从允许的历史与开发档案构建基础核和编码器，后两条构建基线分类器和历史后验。已有独立验证的基础核复用由 `python -B -m experiments.independent_models main` 管理；其复用来源和训练配置保存在模型记录中。

在线选择入口是 [domain_discovery.select](domain_discovery.py)，实验调度见 [experiments](../../experiments/README.md)。现有模型位于 [uniform_models](../../experiment_outputs/gcrt/collision/uniform_models/)，共5类场景、5种来源范围、25组。

## 代码与论文对应

| 文件 | 用途 |
|---|---|
| [domain_training.py](domain_training.py) | 基础核训练、编码器训练与开发检查点选择 |
| [domain_consistency.py](domain_consistency.py) | 按在线预测流程计算训练目标 |
| [domain_discovery.py](domain_discovery.py) | 历史迁移、目标适配、来源权重更新和逐次选例 |
| [joint_response.py](joint_response.py) | 碰撞与间距的联合概率模型 |
| [converged_response.py](converged_response.py)、[residual_scale.py](residual_scale.py) | 分类后验推断、目标残差与可靠性计算 |
| [domain_build.py](domain_build.py)、[joint_training.py](joint_training.py)、[converged_search.py](converged_search.py) | 开发期模型构建与协方差对照 |

完整公式与算法见 [paper_section.md](paper_section.md)，统一问题定义见 [方法概述](../paper_section.md)。`H1F0` 表示静态历史排序，`H0F1` 表示聚合历史教师加目标反馈；后者仍包含历史信息。机制对照的具体范围见 [实验设置](../../experiments/README.md)。

实验产物中的 `gcrt` 是本方法的登记标识。当前证据见 [独立验证报告](../../experiment_outputs/benchmark/collision/discovery/independent_validation/report.md) 和 [方法定型记录](../../experiment_outputs/benchmark/collision/discovery/domain_comparison/diagonal_controls/finalization.json)。
