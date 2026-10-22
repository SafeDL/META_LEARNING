# FSBO 碰撞分类适配

FSBO 是 Few-Shot Bayesian Optimization（少样本贝叶斯优化），是一项已有方法。它先从多个历史任务学习共享模型，再用少量新任务反馈进行适配。本目录依据 [FSBO 原论文](https://arxiv.org/abs/2101.07667) 实现面向碰撞标签的分类适配。

本项目把每个历史驾驶系统当作一个独立任务，学习四层场景表示与高斯过程代理，使用开发系统选择模型。目标测试的每一轮，都从同一预训练模型开始，用全部已查询标签微调，再选择预测碰撞概率最高的未测场景。

该适配使用 Bernoulli 二分类观测和 Laplace 近似；输出为碰撞概率，选例按碰撞概率进行。它与原论文面向连续目标优化的设置有区别，报告中统一称“FSBO 碰撞适配”。

[method.py](method.py) 提供训练与加载，[build.py](build.py) 可单独构建全来源和严格留类别模型；完整实验构建入口为 `python -B -m benchmark.build_collision_models complete`。已完成 20 个 FSBO 模型及正式比较，模型与结果索引见 [基线结果说明](../../../results/baselines/README.md)。
