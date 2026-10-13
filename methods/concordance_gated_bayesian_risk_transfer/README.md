# CG-BRT：一致性门控的贝叶斯风险迁移

目标是在 200 次测试内尽量多发现 `R > 0.5` 的场景。选择器只接收已查询的风险分数，前 10 次初始化计入预算。

方法分三步：

1. 六个历史来源分别预测风险；每次得到目标系统的风险反馈后，更新来源权重、预测误差和不确定性。Student-t 模型允许各来源的误差尺度不同。
2. 比较迁移模型与参考 GP 在测试前给出的排序，再用实测高风险标签更新迁移可信度。可信度为零时使用参考 GP；该 GP 也使用历史平均风险作为先验。
3. 从联合预测中抽样，估计场景进入剩余预算测试集合的机会，并检查替换场景可能带来的损失，确定下一次测试。

| 文件 | 作用 |
|---|---|
| [posterior.py](posterior.py) | 风险后验、动态来源权重与参考 GP |
| [gate.py](gate.py) | 根据排序表现更新迁移可信度 |
| [decision.py](decision.py) | 联合抽样、集合排序与替换检查 |
| [ranking.py](ranking.py) | 两个场景的相关阈值概率计算 |
| [selection.py](selection.py) | 在线选择循环 |
| [config.py](config.py) | 阈值、预算和数值设置 |

代码入口：

```python
from methods.concordance_gated_bayesian_risk_transfer import select
```

模型细节见[设计文档](../../docs/Failure_Discovery_Design.md)，实验与结果见[实验说明](../../experiments/concordance_gated_bayesian_risk_transfer/README.md)和[比较表](../../results/concordance_gated_bayesian_risk_transfer/comparison.md)。
