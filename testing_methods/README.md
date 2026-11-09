# 测试场景选择方法

本目录实现有限候选集中的预算约束测试。输入为公开场景参数、获准历史档案和当前任务已查询反馈，输出选择顺序、查询前预测及更新记录。

| 位置 | 职责 |
|---|---|
| [history_transfer/](history_transfer/README.md) | 联合响应模型、历史来源后验、目标适配和风险选例 |
| [baselines/](baselines/README.md) | 基线的训练、搜索与有限候选库适配 |
| [common/](common/README.md) | 响应档案、共享核和分类推断工具 |
| [selection.py](selection.py) | 检查初始化索引及预算是否合法 |

各方法通过 [QueryLedger](../experiments/query.py) 获取已付费反馈，并自行处理并列候选。驾驶控制器位于 [controllers/](../controllers/README.md)，统一调度和报告入口见 [experiments/](../experiments/README.md)。
