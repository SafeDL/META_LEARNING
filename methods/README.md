# 方法

[历史引导风险测试](history_guided_testing/README.md) 是当前论文主链。六个机制不同的历史算法提供风险先验，目标反馈校正差异 GP，再按预期风险选择场景。历史库与目标库均含切入 1024 个、前车急刹 1024 个；默认入口统一运行准备、训练、九基线比较和组件实验。

[RAS-FRT-UQ](ras_frt_uq/README.md) 已加入统一实验，共享六源历史库和场景预算，沿用二值碰撞反馈、原网络结构与融合权重。当前入口统一运行两类场景；原 S01 实验的模型与数据已归入可恢复压缩包。

[当前结果](../results/history_guided_testing/README.md) · [研究归档与恢复说明](../archives/README.md)
