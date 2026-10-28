# 仿真测试记录

本目录保存驾驶系统执行交通场景后得到的参数、碰撞标签、安全响应、轨迹和调用记录。历史与开发记录还汇总为供模型读取的档案；模型、方法选例过程和比较报告保存在 [experiment_outputs](../experiment_outputs/README.md)。

| 位置 | 当前内容 |
|---|---|
| [history](history/) | 8个历史系统 × 5类场景，共40份档案 |
| [development](development/) | 4个开发系统 × 5类场景，共20份档案 |
| [physical](physical/) | 上述档案对应的30,720次仿真执行与轨迹 |
| [independent_targets](independent_targets/) | 8个独立目标 × 5类场景，共40个执行库、20,480次仿真执行 |

每份档案或执行库包含512个候选场景。历史档案用于训练，开发档案用于模型选择；独立目标库由评估端管理，测试选择器通过 [查询接口](../experiments/query.py) 按预算逐次获得反馈。

因此，“存放其他自动驾驶算法在给定测试场景下的结果用于建模”准确描述了历史档案的用途。这里训练的是本文的**场景响应与失效发现模型**，学习场景参数与不同驾驶系统的碰撞、间距响应之间的关系。被测自动驾驶算法本身的策略训练另有入口。`independent_targets/` 中完整的512条目标结果由评估端持有，不能作为历史训练数据直接读入选择器；一次运行只能逐次付费查询128条，其中包含8条初始化查询。

例如，`history/cut_in/history_beam.npz` 中一行对应该驾驶系统在一个切入场景参数组下的一次执行：`metadata` 记录场景、系统及来源，`collision` 给出实际碰撞标签，`safety` 的两列依次为最小车身间距和最小 TTC。对应的原始轨迹位于 `physical/`，并不是另一套独立样本。当前共有51,200次物理仿真执行：30,720次历史／开发执行和20,480次独立目标执行。

## 记录形式与采集入口

`history/` 和 `development/` 中的 `.npz` 汇总场景参数、碰撞标签、最小间距、TTC 与来源信息。`physical/` 和 `independent_targets/` 保留逐次执行结果、轨迹和调用记录，供回放、成本统计与审计使用。

RSS重算直接读取历史／开发原始轨迹，派生响应保存在 [rss_response](../experiment_outputs/benchmark/collision/rss_response/construction.json)，不覆盖上述原始档案。每个派生文件固定评价对象为ADS，记录统一的RSS参数和原始档案来源；新训练读取独立的 `rss_margin` 字段。

| 入口 | 用途 |
|---|---|
| [collect.py](../experiments/collect.py) | 采集历史、开发系统记录 |
| [independent_preparation.py](../experiments/independent_preparation.py) | 补充邻道开放／受限切入的历史与开发记录 |
| [independent_target_banks.py](../experiments/independent_target_banks.py) | 采集独立目标候选库 |
| [independent_data_audit.py](../experiments/independent_data_audit.py) | 核对历史与开发档案和原始执行的一致性 |

数据位置由 `experiments.protocol.TEST_RECORDS_ROOT` 定义。系统、场景与预算见 [实验设置](../experiments/README.md)，独立数据建设要求见 [准备协议](../experiment_outputs/benchmark/collision/discovery/independent_validation/preparation_protocol.json)。
