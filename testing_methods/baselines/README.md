# 测试基线

本目录实现自动驾驶失效测试的基线，并将其搜索过程接入公开有限候选库和统一付费查询接口。

| 方法 | 核心机制 |
|---|---|
| [Random](response_baselines/README.md) | 按固定种子随机查询 |
| [RF-EI](av_bo_rf_ei/README.md) | 随机森林、Jackknife 方差与期望改进 |
| [SAMOTA](samota/README.md) | 多目标代理集成与全局／局部搜索 |
| [ScenarioFuzz](scenario_fuzz/README.md) | 图风险模型、变异过滤和种子调度 |
| [配对系统主动评估](paired_system_active_evaluation/README.md) | 历史代理校正、支持互信息与批量选择 |
| [BehAVExplor](behavexplor/README.md) | 时序行为抽象、能量调度和场景变异 |
| [Indago](indago/README.md) | 历史分类器与显著性爬山搜索，保留为补充对照 |

当前主比较采用前六项。各目录说明论文出处、对应函数、参数和任务适配。所有方法使用相同的候选、共同初始化及总预算，实际碰撞统一作为失效标签。

[experiments.pilot](../../experiments/pilot.py)负责接入，共用组件位于 [common/](../common/README.md)。结果见[当前论文报告](../../artifacts/runs/joint_response/paper/reports/results.md)，含 Indago 的原配置比较见[固定历史报告](../../artifacts/runs/joint_response/fixed/reports/results.md)。
