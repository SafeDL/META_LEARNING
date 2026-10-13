# 公共风险模型、历史输入与标准基线

本目录为 [CG-BRT](../concordance_gated_bayesian_risk_transfer/README.md) 和基线提供公共代码。

| 文件 | 作用 |
|---|---|
| [config.py](config.py)、[history.py](history.py) | 六历史来源、固定配置、数据读取与训练划分 |
| [kernel.py](kernel.py)、[train.py](train.py) | 风险核训练与检查点读取 |
| [gp.py](gp.py) | GP 更新与风险选例公式 |
| [baseline.py](baseline.py)、[archive.py](archive.py) | 标准基线与 BOP-Elites 场景档案 |
| [search.py](search.py)、[oracle.py](oracle.py) | 基线测试流程与查询反馈接口 |
| [scenarios.py](scenarios.py)、[risk.py](risk.py)、[measurements.py](measurements.py) | 场景定义、风险计算与物理执行 |
| [io.py](io.py) | 文件读写 |

历史数据和模型见[输入目录](../../results/history_guided_testing/README.md)。当前六系统实验由 [CG-BRT 实验目录](../../experiments/concordance_gated_bayesian_risk_transfer/README.md) 组织。
