# LRT 实验结果

本目录保存局部风险迁移（LRT）的模型、历史训练标签、主比较和机制诊断。本文最新方法见 [GCRT](../../gcrt/README.md)。

联合场景实验使用五个种子，两类场景共同分配 200 次测试预算。已完成 1470 条主比较与失配实验轨迹，以及 1350 个状态的分支收益审计。

| 场景测试集 | LRT H200 | LRT 宏平均覆盖率 | 相对 CG-BRT 的发现数／覆盖率保留比例 |
|---|---:|---:|---:|
| 前两个 | 141.417 | 72.197% | 100.118%／99.315% |
| 第三个 | 140.333 | 71.899% | 101.080%／100.982% |

20 份价值网络均完成 4000 次训练更新。主要结果与全部预设消融见 [comparison.md](comparison.md)。

| 位置 | 内容 |
|---|---|
| `models/`、`historical/` | 价值网络、训练记录、历史轨迹与分支标签 |
| `main/`、`stress/` | 主实验与档案失配条件的正式轨迹 |
| `diagnostics/`、`diagnostics.json` | 固定反馈和评价端分支收益诊断 |
| `comparison.md`、`comparison.json` | 逐控制器结果、消融与配对统计 |
| `protocol.json`、`frozen_method.zip` | 实验设置与源码、权重快照 |
| `verification.json`、`numerical_validation.json` | 查询、损失预算与数值核验 |
| [by_family](by_family/README.md) | 切入、制动分别 200 次的单种子比较与完整成本 |
| [adaptive](adaptive/README.md) | 基于固定 LRT 查询顺序的计数停止对照 |

联合预算与按类预算分别报告。所有实验使用已有实测响应回放，未查询风险保存在评价进程。实现和入口见 [methods/baselines/lrt](../../../methods/baselines/lrt/README.md)。
