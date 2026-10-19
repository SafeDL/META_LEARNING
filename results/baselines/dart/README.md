# DART 实验结果

决策相关风险迁移（DART）已完成前两个场景测试集的 420 条比较轨迹，以及第三个测试集的 120 条比较轨迹。第三个测试集的实测响应位于 [data/scenarios/pool_2](../../../data/scenarios/pool_2/)。

相对 CG-BRT，前两个测试集的发现数／覆盖率保留比例为 **98.8%／98.3%**，第三个测试集为 **99.4%／100.1%**，均达到预设 90% 门槛。

| 位置 | 内容 |
|---|---|
| [development_comparison.md](development_comparison.md) | 前两个测试集的完整比较与消融 |
| [confirmation_comparison.md](confirmation_comparison.md) | 第三个测试集的四种方法比较 |
| `models/`、`historical_support/` | 历史编码器、训练记录和伪目标支持轨迹 |
| `development/`、`confirmation/` | 正式查询轨迹 |
| `protocol.json`、`frozen_inputs.zip` | 实验设置与原始源码、权重、历史输入快照 |
| `verification.json`、`status.json` | 查询核验与完成状态 |
| `*_results.png`、`*_results.pdf` | 发现能力与机制诊断图 |

运行命令见[评价说明](../../../methods/baselines/dart/evaluation/README.md)。本文方法 GCRT 复用 DART 的历史响应关联编码器，结果见 [GCRT](../../gcrt/README.md)。
