# 决策相关风险迁移

决策相关风险迁移（DART，Decision-Aware Risk Transfer）是对比方法。它在 CG-BRT 的同历史建议上增加历史关联核、条件前瞻和替换损失预算。本文 GCRT 复用其历史响应关联编码器。

DART 用虚拟反馈估计一次查询对后续发现的影响，再决定是否替换 CG-BRT 建议。两种状态共同使用实际执行场景的风险，每轮只执行一个场景。默认预算 200 次、初始化 10 次、后验替换损失预算 20。

```python
from methods.baselines.dart import select

result = select(x, covariance, priors, probability, fixed_prior, oracle,
                proposal_covariance=learned_covariance)
```

`proposal_covariance` 为学习后的关联核，`covariance` 为 CG-BRT 原核。`enabled=False` 用于查询序列回归；`lookahead=False`、`protection=False`、`spatial=False` 对应前瞻、损失保护和空间关联消融。

| 文件 | 用途 |
|---|---|
| [encoder.py](encoder.py) | 按来源编码历史集合，形成固定关联核 |
| [objectives.py](objectives.py)、[train.py](train.py) | 决策相关训练与历史验证 |
| [fantasy.py](fantasy.py)、[acceleration.py](acceleration.py) | 虚拟条件更新与 Student-t 数值计算 |
| [planning.py](planning.py)、[selection.py](selection.py) | 候选筛选、条件前瞻和风险查询 |
| [config.py](config.py) | 网络、训练和决策设置 |

已完成前两个场景测试集的开发比较和第三个测试集的确认比较。结果与运行入口见 [evaluation](evaluation/README.md)、[results/baselines/dart](../../../results/baselines/dart/README.md)。GPU 数值依赖见 `requirements.txt`。
