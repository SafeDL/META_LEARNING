# 碰撞场景发现实验

本目录负责采集驾驶系统响应、构建模型、安排查询、统计结果和核对实验记录。测试场景选择算法位于 [testing_methods](../testing_methods/README.md)，驾驶控制器位于 [controllers](../controllers/README.md)，场景执行由 [simulator](../simulator/README.md) 提供。

## 系统与场景

当前研究使用8个历史系统、4个开发系统和8个独立目标系统。历史系统提供训练数据，开发系统用于选择模型和参数，独立目标用于最终评价。系统来自规则控制、优化规划、搜索规划和学习策略四类，每类有两条实现路线；配置见 [系统注册表](../controllers/highway/catalog.py)。

| 场景 | 主要考察的驾驶交互 |
|---|---|
| 前车制动 | 初始间距、速度差、制动强度和开始时刻变化下的跟车制动 |
| 邻车切入 | 切入方向、间距、速度差和横向移动时间变化下的避让 |
| 前车切出暴露障碍 | 遮挡解除后的障碍识别、制动与换道 |
| 换道后车冲突 | 慢前车、快速后车和邻道空间共同约束下的换道 |
| 邻道开放／受限切入 | 相同切入参数下，邻道交通约束对驾驶结果和测试分配的影响 |

场景参数与取值范围见 [scenarios.py](scenarios.py)。物理仿真频率20 Hz、驾驶决策频率5 Hz，单场景最长10秒。当前结果来自 HighwayEnv；MetaDrive 的实现与运行说明见其 [README](../simulator/metadrive/README.md)。

## 查询与数据使用

每个目标系统、每类场景有512个公开候选，单次方法运行的预算为128次，包含共同的8次随机初始化。五个算法种子使用配对设置。邻道开放与受限切入的两组条件共用同一预算。

[test_records](../test_records/README.md) 保存仿真执行结果。选择器通过 [query.py](query.py) 逐次查询，只能获得已经计入预算的目标反馈；完整标签和未查询轨迹由评估端持有。各方法获准使用的碰撞、间距、TTC、撞击量或驾驶质量反馈写入 [冻结协议](../experiment_outputs/benchmark/collision/discovery/independent_validation/protocol.json)。

历史档案用于训练，开发档案用于模型选择。严格留家族时，目标所属控制方法类别同时退出历史训练和开发选择，模型按该来源范围重新构建。

## 已完成的实验

| 实验 | 范围 | 实际运行数 |
|---|---|---|
| 全来源比较 | 8个独立目标 × 5类场景 × 5个种子 × 7种方法 | 1,400 |
| 严格留家族 | 相同目标、场景和种子，重新运行本文方法与 ScenarioFuzz | 400 |
| 交互机制对照 | 邻道开放／受限切入上的静态历史、聚合历史教师加反馈 | 80 |

合计1,880项实际运行、240,640次回放查询。严格留家族中的 Random、RF-EI 和 SAMOTA 另以600条引用复用全来源运行，形成2,480条逻辑比较记录。上述查询在预先采集的40个目标库中逐次揭示反馈，目标库共有20,480次仿真执行。

开发期另保留400项主比较运行，以及协方差与匹配机制控制，用于解释方法定型依据。开发期的四组机制对照与独立验证的两个交互对照分别记录。

## 评价方式

| 指标 | 含义 |
|---|---|
| 去重碰撞数 | 预算内发现的不同场景参数组的碰撞数量 |
| 有限库召回率 | 已发现碰撞组占完整候选库碰撞组的比例 |
| 发现曲线与 AUC | 随查询推进找到碰撞的速度与持续性 |
| 行为覆盖 | 已发现碰撞涉及的驾驶动作和交互类型 |
| 成本 | 数据采集、模型训练、目标适配、选例和场景执行耗时 |

碰撞以仿真事件标签判定。无碰撞候选库仍参与数量与曲线统计，召回率记为不适用。统计先在同一实现路线内汇总，再对路线等权汇总，区间使用配对分层自助法；算法种子用于重复运行。

## 使用入口

已有结论可直接查看 [独立验证报告](../experiment_outputs/benchmark/collision/discovery/independent_validation/report.md)、[开发比较](../experiment_outputs/benchmark/collision/discovery/domain_comparison/report.md) 和 [方法定型](../experiment_outputs/benchmark/collision/discovery/domain_comparison/diagonal_controls/finalization.json)。

| 模块 | 用途 |
|---|---|
| [collect.py](collect.py) | 采集历史、开发系统的场景响应与档案 |
| [independent_preparation.py](independent_preparation.py) | 登记独立系统，准备驾驶资格、策略与交互数据 |
| [independent_target_banks.py](independent_target_banks.py) | 采集独立目标的40个候选执行库 |
| [independent_models.py](independent_models.py) | 构建主方法模型、基线分类器和历史后验 |
| [independent_validation.py](independent_validation.py) | 执行预算内查询和结果复用 |
| [independent_validation_audit.py](independent_validation_audit.py) | 核对已查询反馈、预算、选例与复用引用 |
| [independent_validation_report.py](independent_validation_report.py) | 汇总统计、成本、曲线和报告 |

从根目录运行完整工作流：

```powershell
python -B -m experiments.independent_workflow
```

该入口核对已完成的数据与模型，继续构建和查询流程，并写入工作流、审计及报告产物。查看已有结果直接打开报告即可。主方法从历史档案开始的训练说明见 [history_transfer](../testing_methods/history_transfer/README.md)。

## 实验记录与源码

实验产物位于 [experiment_outputs](../experiment_outputs/README.md)。冻结协议与原始源码归档保存实验执行时的定义，当前源码的整理记录见 [source_maintenance.json](../experiment_outputs/benchmark/collision/discovery/independent_validation/model_construction/source_maintenance.json)。其中的 `obsolete_output_cleanup` 列出已清理的过时文件及核对依据。

开发期复核使用 [原源码归档](../experiment_outputs/benchmark/collision/discovery/domain_comparison/source_snapshot.zip) 和对应模型。原记录中的 `benchmark`、`gcrt` 和旧根路径按登记信息解释，当前运行入口使用 `experiments` 与 `testing_methods`。
