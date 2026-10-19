# 历史来源风险预测与高斯混合基线

本目录为 [CG-BRT](../cg_brt/README.md) 提供六个历史来源的风险预测，也保留固定平均和高斯来源混合基线。

| 文件 | 作用 |
|---|---|
| [experts.py](experts.py) | 同类场景中取 8 个近邻预测风险；初始权重按机制组均衡分配 |
| [posterior.py](posterior.py) | 根据风险反馈更新高斯来源混合 |
| [selection.py](selection.py) | 基线选择入口 |
| [config.py](config.py) | 近邻、噪声与预算设置 |

当前主方法使用六个历史来源；旧基线的来源设置以冻结记录为准。结果见[比较表](../../../results/baselines/cg_brt/comparison.md)。
