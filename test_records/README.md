# 仿真测试记录

本目录保存驾驶系统执行交通场景后得到的参数、碰撞标签、安全响应、轨迹和调用记录。历史与开发记录还汇总为供模型读取的档案；模型、方法选例过程和比较报告保存在 [experiment_outputs](../experiment_outputs/README.md)。

| 位置 | 当前内容 |
|---|---|
| [collision/history](collision/history/) | 8个历史系统 × 5类场景，共40份档案 |
| [collision/development](collision/development/) | 4个开发系统 × 5类场景，共20份档案 |
| [collision/physical](collision/physical/) | 上述档案对应的30,720次仿真执行与轨迹 |
| [collision/independent_targets](collision/independent_targets/) | 8个独立目标 × 5类场景，共40个执行库、20,480次仿真执行 |

每份档案或执行库包含512个候选场景。历史档案用于训练，开发档案用于模型选择；独立目标库由评估端管理，测试选择器通过 [查询接口](../experiments/query.py) 按预算逐次获得反馈。

## 记录形式与采集入口

`history/` 和 `development/` 中的 `.npz` 汇总场景参数、碰撞标签、最小间距、TTC 与来源信息。`physical/` 和 `independent_targets/` 保留逐次执行结果、轨迹和调用记录，供回放、成本统计与审计使用。

| 入口 | 用途 |
|---|---|
| [collect.py](../experiments/collect.py) | 采集历史、开发系统记录 |
| [independent_preparation.py](../experiments/independent_preparation.py) | 补充邻道开放／受限切入的历史与开发记录 |
| [independent_target_banks.py](../experiments/independent_target_banks.py) | 采集独立目标候选库 |
| [independent_data_audit.py](../experiments/independent_data_audit.py) | 核对历史与开发档案和原始执行的一致性 |

数据位置由 `experiments.protocol.TEST_RECORDS_ROOT` 定义。系统、场景与预算见 [实验设置](../experiments/README.md)，独立数据建设要求见 [准备协议](../experiment_outputs/benchmark/collision/discovery/independent_validation/preparation_protocol.json)。
