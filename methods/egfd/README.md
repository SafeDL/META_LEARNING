# EGFD：冻结碰撞发现对照

EGFD（Evidence-Guided Failure Discovery）使用已测试场景的风险 `R` 和碰撞 `C`，更新历史响应模型与碰撞预测。它作为已有对照保留；当前 [CG-BRT](../concordance_gated_bayesian_risk_transfer/README.md) 只使用风险反馈。

```python
from methods.egfd import select
```

`select(x, risk_prior, margin_prior, oracle, seed=11)` 的反馈接口返回风险和碰撞。预算 200、初始化 10，见 [config.py](config.py)；[selection.py](selection.py) 负责选例，[evidence.py](evidence.py) 更新模型证据，[controls.py](controls.py) 提供高斯和 Student-t 对照。

已有轨迹保存在[统一基准](../../results/driving_benchmark/README.md)，结果见[比较表](../../results/concordance_gated_bayesian_risk_transfer/comparison.md)。
