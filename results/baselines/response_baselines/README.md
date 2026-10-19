# 响应模型基线的权重与输入

本目录供历史响应高斯过程（HR-GP）和行为响应基线读取模型。正式比较见 [CG-BRT 结果表](../cg_brt/comparison.md)。

| 位置 | 内容 |
|---|---|
| `history_models/` | 五种子的历史响应网络与训练记录 |
| `behavior_models/` | 五种子的行为条件响应网络与训练记录 |
| `protocol.json`、`lock.json`、`locked_inputs.zip` | 原始实验设置、输入核验记录与快照 |
| `source_inputs.zip` | 保留的推断源码与模型快照 |
| `round_two_protocol.json` | 历史行为假设设置 |

方法实现与基线运行说明见 [methods/baselines/response_baselines](../../../methods/baselines/response_baselines/README.md)。
