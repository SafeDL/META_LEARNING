# 方法代码

这里存放选择测试场景的算法，与负责驾驶车辆的 [controllers](../controllers/README.md) 分开。研究问题、历史与目标系统、场景、预算和指标见 [项目说明](../README.md)。

| 目录 | 当前用途 |
|---|---|
| [gcrt](gcrt/README.md) | 根据历史测试经验和新反馈预测碰撞，逐次选择场景，并支持提出新坐标 |
| [common](common/README.md) | 各方法共用的测试档案、场景表示、概率模型和模型复用检查 |
| [baselines](baselines/README.md) | 七个候选库对照，以及 Indago、SAMOTA、DeepHyperion 连续搜索适配 |

构建使用 `python -B -m benchmark.build_collision_models core` 和 `complete`，输入是已采集的历史与开发档案。运行比较、控制查询预算和生成报告由 [benchmark](../benchmark/README.md) 负责。这里的方法不能直接读取未查询的目标标签。

已实现方法与尚待补充的文献基线在 [对比方法说明](baselines/README.md) 中分别列出。旧内部方法、停止评价及旧风险报告已归档，不作为当前代码入口。
