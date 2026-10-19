# 一致性门控贝叶斯风险迁移

一致性门控贝叶斯风险迁移（CG-BRT，Concordance-Gated Bayesian Risk Transfer）是具名对比方法，也为 LRT、DART 提供参照建议。LRT 的动作价值标签通过继续运行 CG-BRT 生成。

方法先用历史来源建立风险预测，再用真实测试结果更新来源权重、误差尺度和排序可信度，结合联合抽样选择下一场景。默认预算为 200 次，含前 10 次初始化；危险判定为 `R > 0.5`。

| 文件 | 用途 |
|---|---|
| [posterior.py](posterior.py) | 来源风险后验与参考高斯过程 |
| [gate.py](gate.py) | 根据实测排序表现更新迁移可信度 |
| [decision.py](decision.py)、[ranking.py](ranking.py) | 联合抽样、集合排序与相关阈值概率 |
| [selection.py](selection.py) | 逐次测试入口 |
| [reference.py](reference.py) | 根据实际测试历史提出参照建议，供 LRT、DART 使用 |
| [config.py](config.py) | 预算、阈值与数值设置 |

```python
from methods.baselines.cg_brt import select
```

基础核和历史风险预测由 [HGT](../hgt/README.md)、[历史专家混合](../hem/README.md)提供。构建这些组件的入口包含在 `python -B -m methods.baselines.lrt.build` 中。

实验命令见 [evaluation](evaluation/README.md)，已有比较见 [results/baselines/cg_brt](../../../results/baselines/cg_brt/README.md)。本文当前方法见 [GCRT](../../gcrt/README.md)。
