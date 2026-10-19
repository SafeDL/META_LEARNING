# 局部风险迁移

局部风险迁移（LRT）作为具名对比方法保留，采用局部来源后验、CG-BRT 参照建议、动作价值学习与替换损失约束。本文当前方法为 [GCRT](../../gcrt/README.md)。

| 位置 | 职责 |
|---|---|
| `posterior.py`、`fantasy.py`、`selection.py` | LRT 的局部风险后验、虚拟更新与场景选择 |
| `teacher.py`、`value.py`、`data.py`、`train.py` | 参照策略分支回放、动作价值训练数据与价值网络 |
| `adaptive/` | 使用 LRT 查询顺序的计数停止对照 |
| `evaluation/` | 联合场景比较、按类实验、成本报告与数值核验 |

CG-BRT 位于同级 [cg_brt](../cg_brt/README.md)，历史先验和 Student-t 工具位于 [common](../../common/README.md)。

```powershell
python -B -m methods.baselines.lrt.build
python -B -m methods.baselines.lrt.evaluation.by_family
```

结果见 [联合场景实验](../../../results/baselines/lrt/README.md)、[按类实验](../../../results/baselines/lrt/by_family/README.md)和[计数停止对照](../../../results/baselines/lrt/adaptive/README.md)。
