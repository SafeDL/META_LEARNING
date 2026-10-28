# 自动驾驶测试基线

本目录包含六种碰撞场景发现基线及其训练、搜索和验证代码。所有方法在相同候选库和查询预算下比较，场景、反馈权限与指标见 [实验设置](../../experiments/README.md)。

| 方法 | 核心做法 | 实现说明 |
|---|---|---|
| Random | 按固定随机顺序测试 | [response_baselines](response_baselines/README.md) |
| RF-EI | 用随机森林预测风险，按期望改进选择场景 | [av_bo_rf_ei](av_bo_rf_ei/README.md) |
| SAMOTA | 使用多目标代理集成与全局、局部搜索 | [samota](samota/README.md) |
| ScenarioFuzz | 用图模型过滤变异场景，结合驾驶质量调度种子 | [scenario_fuzz](scenario_fuzz/README.md) |
| 配对系统主动评估 | 结合历史代理、目标反馈与覆盖信息，批量选择场景 | [paired_system_active_evaluation](paired_system_active_evaluation/README.md) |
| Indago | 用历史分类器和显著性指导的爬山搜索寻找高风险场景 | [indago](indago/README.md) |

各方法 README 保留原论文、作者代码、主要步骤、超参数及有限候选库适配说明。共用模型在 [common](../common/README.md)，被测驾驶控制器在 [controllers](../../controllers/README.md)。

## 已完成的比较范围

全来源独立验证包含本文方法和全部六个基线。严格留家族比较包含本文方法、ScenarioFuzz、Random、RF-EI 和 SAMOTA；后三者不使用历史来源，复用相同的全来源运行。配对系统主动评估与 Indago 的正式结果属于全来源比较。

原独立验证保留25个 ScenarioFuzz 模型和5个 Indago 模型；模型位置及来源后验读取见 [recorded_models.py](../../experiments/recorded_models.py)，构建源码保存在原登记归档中。其余方法在运行时根据允许的数据构建代理或更新模型。运行入口见 [实验指导](../../experiments/README.md)，比较结果见 [独立验证报告](../../experiment_outputs/benchmark/collision/discovery/independent_validation/report.md)。
