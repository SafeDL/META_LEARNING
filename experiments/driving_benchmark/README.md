# 共享响应池与基线

本目录维护六个被测系统、两个场景池及基线。当前方法的评价与比较见 [CG-BRT 实验入口](../concordance_gated_bayesian_risk_transfer/README.md)。

| 文件 | 作用 |
|---|---|
| [config.py](config.py)、[pools.py](pools.py) | 系统、场景池与预算设置 |
| [prepare.py](prepare.py) | 构建物理实测响应池 |
| [interface_audit.py](interface_audit.py)、[validate_banks.py](validate_banks.py) | 核对执行接口与响应数据 |
| [baselines.py](baselines.py) | 核对冻结输入并补齐缺失基线轨迹，不重新训练 |
| [evaluation.py](evaluation.py) | 发现数、覆盖率与发现曲线面积 |

已有响应池与基线见[数据目录](../../results/driving_benchmark/README.md)。仅查看结果时运行主方法的 `compare`，无需重新采集响应或运行基线。
