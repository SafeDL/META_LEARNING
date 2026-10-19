# CG-BRT 实验结果

已完成六种驾驶控制器、两个场景测试集、五个种子的 60 条 CG-BRT 轨迹。两类场景共同使用 200 次测试预算，只接收连续风险反馈。

| 指标 | 五种子平均结果 |
|---|---:|
| 危险场景发现数量 H200 | 141.25 |
| 宏平均危险场景覆盖率 | 72.695% |

危险场景按 `R > 0.5` 判定。此处为联合场景实验；按类分别 200 次的结果见 [LRT 按类比较](../lrt/by_family/README.md)，本文方法结果见 [GCRT](../../gcrt/README.md)。

| 位置 | 内容 |
|---|---|
| [comparison.md](comparison.md)、`comparison.json` | 代表性基线、逐控制器结果和预算上限 |
| `driving/` | 正式查询、风险反馈和来源更新记录 |
| `protocol.json`、`baseline_catalog.json` | 实验设置与基线信息 |
| `verification.json`、`status.json` | 核验与完成状态 |
| `source_inputs.zip` | 原始源码、历史数据与基础核快照 |

汇总命令：`python -B -m methods.baselines.cg_brt.evaluation.compare`。实现见[方法说明](../../../methods/baselines/cg_brt/README.md)。
