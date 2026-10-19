# 响应模型基线

本目录保留统一比较使用的响应模型和选例实现。

| 方法名 | 记录 ID | 方法 |
|---|---|---|
| HR-GP：历史响应 GP | previous_best | 六来源响应网络、响应相似度核及风险／碰撞代理双输出 GP；使用风险反馈更新 |
| Behavior Posterior | behavior_posterior | 根据反馈更新驾驶行为假设并排序 |
| Risk-Conditioned Response GP | risk_conditioned | 用风险反馈校正行为模型的碰撞预测 |
| Matched Collision GP | matched_collision_gp | 在匹配的历史预测器下使用碰撞得分 GP |

HGT（历史引导 GP 风险测试）的记录 ID 为 `frozen_original`，选择入口同样通过 [selectors.py](selectors.py) 调用，模型实现位于 [hgt](../hgt/README.md)。

模型与协议见[输入目录](../../../results/baselines/response_baselines/README.md)。部分行为响应基线额外使用 48 个历史 IDM/FVDM 配置，反馈与历史条件见[完整比较](../../../results/baselines/cg_brt/comparison.md)。

`python -B -m benchmark.baselines` 复用已有轨迹，缺失时运行选例。来源核验使用原始输入锁与 `source_inputs.zip`，包名迁移通过统一快照核验工具处理。
