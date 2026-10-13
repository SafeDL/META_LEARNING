# 冻结响应基线

本目录保留统一比较使用的响应模型推断与选例代码。

| 方法 | 作用 |
|---|---|
| `previous_best` | 六来源响应网络与场景相似度核 |
| `behavior_posterior` | 根据反馈更新驾驶行为假设并排序 |
| `risk_conditioned` | 用风险反馈修正行为模型的碰撞预测 |
| `matched_collision_gp` | 在同一历史预测器下使用碰撞得分 GP |

[selectors.py](selectors.py) 提供选择入口。模型与协议见[输入目录](../../results/response_baselines/README.md)；行为类基线额外使用 48 个历史 IDM/FVDM 配置，见[比较表](../../results/concordance_gated_bayesian_risk_transfer/comparison.md)。

`python -B -m experiments.driving_benchmark.baselines` 复用已有轨迹，缺失时运行选例，不重新训练。该入口的来源核验需要本地 [method_selection.zip](../../archives/method_selection.zip)；仅查看已有比较无需此归档。
