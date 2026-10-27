# 开发期深核训练组件

[method.py](method.py) 提供任务级深核训练和模型加载，由 [domain_build.py](../../history_transfer/domain_build.py) 用于方法开发比较。模型依据 [FSBO](https://arxiv.org/abs/2101.07667) 的共享深核思路，在本项目中使用碰撞标签、伯努利观测和 Laplace 近似。

本目录是共享模型组件。当前六个测试基线见 [baselines](../../baselines/README.md)。本文最终模型由 [domain_training.py](../../history_transfer/domain_training.py) 构建，训练顺序见 [主方法说明](../../history_transfer/README.md)。

开发期检查点及训练记录位于 [domain_comparison/models](../../../experiment_outputs/gcrt/collision/discovery/domain_comparison/models/)，用于复核当时的实验。
