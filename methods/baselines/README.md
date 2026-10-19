# 对比方法

本文方法见 [GCRT](../gcrt/README.md)。本目录统一保存具名对比方法，使用相同的场景测试记录和风险反馈接口。

| 目录 | 方法 |
|---|---|
| [lrt](lrt/README.md) | 局部风险迁移（LRT）：局部来源后验、动作价值学习与替换损失约束 |
| [cg_brt](cg_brt/README.md) | 一致性门控贝叶斯风险迁移（CG-BRT） |
| [dart](dart/README.md) | 决策相关风险迁移（DART）：历史关联编码器与条件前瞻 |
| [hgt](hgt/README.md) | 历史引导高斯过程风险测试（HGT）；同时提供 GP-EI、GP-UCB、RF-BO、随机测试等搜索基线 |
| [response_baselines](response_baselines/README.md) | 历史响应高斯过程（HR-GP）及行为响应基线 |
| [hem](hem/README.md) | 历史专家混合（HEM） |
| [saem](saem/README.md) | 尺度自适应专家混合（SAEM） |
| [ras_frt_uq](ras_frt_uq/README.md) | RAS-FRT-UQ 碰撞反馈基线 |

```powershell
python -B -m methods.baselines.cg_brt.evaluation.compare
python -B -m methods.baselines.dart.evaluation.report
python -B -m methods.baselines.lrt.evaluation.family_report
```

各方法的正式权重与结果位于 [results/baselines](../../results/baselines/README.md)，共用的标准搜索轨迹位于 [results/benchmark](../../results/benchmark/README.md)。历史来源近邻估计、核训练等基础组件也供 GCRT 直接复用。

既有记录 ID `frozen_original`、`previous_best`、`previous` 分别对应 HGT、HR-GP、DART；展示名称由 [catalog.py](../common/catalog.py) 给出。CG-CBM-FD 使用额外碰撞反馈，其已有比较结果保留在 [CG-BRT 比较表](../../results/baselines/cg_brt/comparison.md)，项目未保留其可执行源码。
