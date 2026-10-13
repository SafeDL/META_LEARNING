# Student-t 风险尺度与混合基线

每个历史来源单独估计预测误差尺度。目标风险反馈同时更新来源权重、GP 残差和尺度，使预测不确定性随反馈变化。

[posterior.py](posterior.py) 提供 Student-t 密度和尺度混合模型；[CG-BRT](../concordance_gated_bayesian_risk_transfer/README.md) 复用密度公式。[selection.py](selection.py) 保留原基线选例，结果在[种子 11 比较表](../../results/concordance_gated_bayesian_risk_transfer/comparison.md)中列出。

初始自由度为 5、先验期望尺度为 1；理论参考 [Student-t Processes as Alternatives to Gaussian Processes](https://proceedings.mlr.press/v33/shah14.html)。
