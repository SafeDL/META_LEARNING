# 回归测试

在项目根目录的 `metadrive` 环境中，使用 `python -B -m pytest --collect-only -q` 查看测试，使用 `python -B -m pytest 路径` 运行指定文件或目录。实验暂停期间只运行已确认不会启动仿真采集、训练或正式评估的维护用例，例如 `python -B -m pytest tests/experiments/reporting/test_paired_statistics.py`。[实验协议](../docs/experiment_plan.md) · [pytest 配置](../pyproject.toml)

| 测试目录 | 覆盖范围 |
|---|---|
| [history_transfer/](history_transfer/) | 共享–专属核、联合反馈推断、十步适配、局部可信度、持续归档与 LOSO 门控 |
| [baselines/](baselines/) | Target-only 与 Pooled GP 对照、分组及 CUDA 推断、ScenarioFuzz、配对系统选例与驾驶质量 |
| [controllers/](controllers/) | 决策接口、规划控制器、离散策略训练契约及连续 SAC 动作映射 |
| [simulator/](simulator/) | HighwayEnv 原生时钟、碰撞、观测遮挡与场景交互；MetaDrive 的独立测试位于[平台目录](../simulator/metadrive/tests/) |
| [experiments/](experiments/) | 通用场景域、RSS 与失效指标、存储、查询账本、并行恢复、预算及设备调度 |
| [experiments/scene_design/](experiments/scene_design/) | 正式参数域、开发坐标排除、避撞参考与次生碰撞 |
| [experiments/collection/](experiments/collection/) | 驾驶准入、分阶段采集、历史响应、执行归档及已付费轨迹回放 |
| [experiments/transfer/](experiments/transfer/) | 历史构建、准入与固定配置、目标质量、少反馈预测、跨任务积累和账本审计 |
| [experiments/regression/](experiments/regression/) | 同谱系修复版本、固定套件回放、排序基线与配对指标 |
| [experiments/reporting/](experiments/reporting/) | 整块重采样区间、统计汇总、科研插图与 GIF 时钟 |

[fixtures/](fixtures/) 存放合成查询反馈、配对统计样例和数值回归数据。迁移执行与审计测试共用的原生轨迹由 [transfer/conftest.py](experiments/transfer/conftest.py) 按需创建。
