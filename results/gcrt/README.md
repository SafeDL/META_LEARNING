# 收益校正风险测试的结果

已完成车辆切入、前车紧急制动两类模型，36 条完整测试轨迹、提前停止评价和一次周期更新。全部使用已有实测响应回放。

| 实验 | 评价范围 | 主要内容 |
|---|---|---|
| [完整实验](comparison.md) | 36 条完整轨迹；预设 IDM、IDM＋MOBIL 的 12 组停止评价 | 发现能力、停止效果、周期更新与完整复用成本 |
| [发现效果与测试效率](retention_efficiency/README.md) | 车辆切入的 IDM、IDM＋MOBIL，紧急制动的 MCTS，共 9 组配置 | 与均匀随机测试比较发现数量、覆盖率、调用次数和耗时 |
| [发现数分布校正](calibration_effectiveness/README.md) | 相同 9 组配置的 342 个共同检查点 | 校正前后的预测误差、可靠性与停止结果 |

补充实验的种子固定为 11。MCTS 紧急制动为查看原结果后增加的探索性评价，使用其他控制器重新校准停止参数；原预设 12 组结果单独保留。

## 主要结果

下表为效率实验 9 组配置的等权平均。危险用例按 `R > 0.5` 统计，碰撞按实测标签统计；覆盖率以对应测试集全部同类用例为分母。耗时按实测记录重建，包含仿真执行和算法计算，不含离线训练、校准与维护。

| 方法 | 平均危险用例发现数量 | 平均碰撞用例覆盖率 | 平均仿真调用次数 | 平均执行总耗时（秒） |
|---|---:|---:|---:|---:|
| 均匀随机测试 | 16.56 | 19.70% | 200.00 | 266.39 |
| 本发明方法（GCRT） | 86.00 | 92.23% | 129.44 | 118.91 |

这 9 组配置中，GCRT 累计发现 774 个危险用例，比均匀随机测试少用 35.28% 的仿真调用，重建执行总耗时下降 55.36%。逐控制器结果见[效率报告](retention_efficiency/report.md)。

原 IDM 紧急制动及无碰撞正例的 IDM＋MOBIL 结果见[原独立评价表](retention_efficiency/original_evaluation_table.md)和[场景适用性核查](retention_efficiency/scenario_audit.json)。包含构建、校准和维护的复用成本见[完整实验报告](comparison.md)。

## 文件索引

| 位置 | 内容 |
|---|---|
| [retention_efficiency/](retention_efficiency/README.md) | 效率图表、逐配置统计、补充评价协议与原预设结果 |
| [calibration_effectiveness/](calibration_effectiveness/README.md) | 配对预测指标、偏差分组、可靠性图、校正系数与交底书第 8.3 节填写内容 |
| [comparison.md](comparison.md)、`summary.json` | 两类发现能力、停止效果与完整成本 |
| [cutin/comparison.md](cutin/comparison.md)、[brake/comparison.md](brake/comparison.md) | 各类逐控制器结果、预测误差与复用成本 |
| `cutin/`、`brake/` 下的 `kernel/`、`encoder/` | 基础核、历史编码器与构建记录 |
| 各类 `model.pt`、`training.json` | 剩余发现网络及历史验证选择 |
| 各类 `historical/`、`trajectories/` | 历史训练轨迹、正式查询记录与每步已知信息的特征 |
| 各类 `stopping.json`、`calibration.json` | 停止结果与阈值 |
| 各类 `update.json`、`updated.pt` | 周期更新的验证记录与发布权重 |
| 各类 `build_cost.json`、`historical_cost.json`、`pool_cache/` | 构建、历史运行和候选准备成本 |
| [full_horizon/comparison.md](full_horizon/comparison.md) | 完整剩余预算前瞻对照 |
| `protocol.json`、`source_inputs.zip` | 正式实验设置与源码、输入快照 |
| `audit.json`、`numerics.json` | 查询、数据隔离与数值检查 |

大体积特征和逐步预测缓存不纳入版本管理。首次拉取项目后，可直接阅读已有报告；需要重算时先运行构建与评价入口生成缓存。

重新汇总：`python -B -m methods.gcrt.report`。运行说明见[方法 README](../../methods/gcrt/README.md)。
