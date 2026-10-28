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

连续安全度量的文献依据、RSS归一化裕度定义与追尾覆盖见 [安全度量说明](../docs/rss.md)。RSS的模型路径、来源范围及对角协方差配置统一定义在 `testing_methods/history_transfer/training.py`。训练直接使用历史／开发数据和源码中的配置，不要求已有旧实验模型或协方差决策文件。完整25组RSS模型与新的独立比较尚未完成。

`python -B -m experiments.render_scenarios` 从原方法已查询的独立目标记录生成五类 GIF；`python -B -m experiments.analyze_rss` 对对应完整库作事后诊断。已冻结结果仍属于原间距响应方法。

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
| [train.py](train.py) | 从历史轨迹派生RSS响应，训练25组模型并导出来源后验 |
| [check_training.py](check_training.py) | 用缩小配置核对五类真实数据上的训练与隔离查询；临时模型随检查结束清理 |
| [recorded_models.py](recorded_models.py) | 读取原冻结实验的模型与来源后验 |
| [independent_validation.py](independent_validation.py) | 执行预算内查询和结果复用 |
| [independent_validation_audit.py](independent_validation_audit.py) | 核对已查询反馈、预算、选例与复用引用 |
| [independent_validation_report.py](independent_validation_report.py) | 汇总统计、成本、曲线和报告 |

从根目录构建当前RSS方法：

```powershell
python -B -m experiments.train
```

快速核对真实数据上的训练与查询链路：

```powershell
python -B -m experiments.check_training
```

该检查将模型写入临时目录，只保留检查报告。主方法的训练顺序与超参数见 [history_transfer](../testing_methods/history_transfer/README.md)。复现旧研究使用原登记源码归档和冻结协议；当前源码保留结果读取、核验与统计工具。

## 实验记录与源码

实验产物位于 [experiment_outputs](../experiment_outputs/README.md)。冻结协议与原始源码归档保存实验执行时的定义；原目录整理记录见 [source_maintenance.json](../experiment_outputs/benchmark/collision/discovery/independent_validation/model_construction/source_maintenance.json)。当前RSS代码的合并、删除与回归检查记录见 [self_check.json](../experiment_outputs/benchmark/collision/rss_response/self_check.json)。

开发期复核使用 [原源码归档](../experiment_outputs/benchmark/collision/discovery/domain_comparison/source_snapshot.zip) 和对应模型。原记录中的 `benchmark`、`gcrt` 和旧根路径按登记信息解释，当前运行入口使用 `experiments` 与 `testing_methods`。
