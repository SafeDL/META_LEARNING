# 实测数据

本目录保存已有测试记录。模型权重、查询轨迹与统计报告位于 [results](../results/README.md)。

| 位置 | 内容 |
|---|---|
| `history/scenarios.json`、`history/responses.npz` | 六种历史 IDM／FVDM 配置的场景参数与风险记录 |
| `scenarios/pool_0/`、`scenarios/pool_1/`、`scenarios/pool_2/` | 三个场景测试集；各含切入、前车紧急制动 1024 个场景 |
| `scenarios/pool_*/scenarios.json` | 场景参数和场景 ID |
| `scenarios/pool_*/<控制器>/responses.npz` | 场景输入、实测风险与碰撞记录 |
| `scenarios/pool_*/<控制器>/measurements.jsonl` | 逐场景测量结果与执行耗时 |
| `scenarios/pool_*/<控制器>/physical_cost.json` | 累计执行耗时与控制器配置 |

三个测试集的物理参数范围相同，Sobol 采样点不同。`pool_0`、`pool_1`、`pool_2` 是沿用的测试集标识。

当前实验按功能场景读取数据，每个任务有 1024 个候选场景。测试方法只通过[风险接口](../benchmark/risk_channel.py)获取实际查询结果；全部危险场景数量、未查询风险、碰撞和执行耗时由评价代码读取。

历史来源与被测驾驶控制器见 [controllers](../controllers/README.md)，读取路径见 [benchmark/data.py](../benchmark/data.py)。
