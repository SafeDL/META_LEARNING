# 混合历史数据预训练与目标微调

本方法简称 Pooled-FT。它把允许使用的历史系统测试记录混合起来，训练一个预测碰撞的多层感知机，再用少量目标系统反馈微调。

预训练模型通过独立开发系统选择。目标测试的每一轮从同一预训练权重开始，用全部已查询的目标标签微调，选择预测碰撞概率最高的未测场景。严格留一家族时，该家族同时退出历史训练和开发选参。

[method.py](method.py) 实现训练、微调和加载，[build.py](build.py) 可单独构建全来源和留类别模型。完整构建入口为 `python -B -m benchmark.build_collision_models complete`；统一查询由 [collision.py](../response_baselines/collision.py) 执行。

20 个模型及正式比较已完成。数据与预算说明见 [项目说明](../../../README.md)，保存位置见 [基线结果说明](../../../results/baselines/README.md)。
