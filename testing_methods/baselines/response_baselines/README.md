# 响应预测与随机查询基线

[collision.py](collision.py)按算法种子生成候选随机顺序，先执行给定的共同初始化，再依次查询未执行候选，直到达到预算。每个候选最多查询一次。

调用入口为 `run(candidates, oracle, initial_indices=..., budget=200, seed=11)`。选择仅依赖公开候选和随机种子；查询返回的碰撞与其他反馈用于记录。

E01 已完成 Pooled GP 的少样本比较。E02 的 Random 对照复用 E01 中相同种子下的 200 条共同反馈，每个场景与目标 ADS 组合各有一条随机发现曲线。[E01 结果](../../../results/k_shot_adaptation/README.md) · [E02 结果](../../../results/budget_discovery/README.md)

[pooled_gp.py](pooled_gp.py)实现 K-shot 方案中的 Pooled historical GP：在冻结历史表征上，将同场景的原始历史碰撞–RSS 与共同目标反馈序列合并，拟合单个联合 GP。不同历史 ADS 的重复坐标及冲突响应保留。每个登记 K 从同一基础核重新优化六个标量，Adam 10 步、学习率 0.003，再用全部合并响应条件化预测。

当前比较设计见[长期实验计划](../../../docs/experiment_plan.md)。历史记录与目标反馈由评估端分别提供，不在方法内读取目标档案。检查点独立冷启动，K=0 仅以历史和公共候选建立场景缓存，K>0 只开放该点反馈序列。

正式入口 `experiments.history_transfer_experiment` 显式使用 [grouped_gp.py](grouped_gp.py) 的重复输入计算。碰撞次数／成功数与 RSS 均值／组内残差保留全部记录的似然，训练总量仍按原始记录计数；分类稳定项按唯一输入表示。逐记录求解器保留为数值等价对照，回归见 [test_grouped_pooled_gp.py](../../../tests/baselines/test_grouped_pooled_gp.py)。正式方案为原始六标量十步，学习率 0.003，不开展额外开发选参；保留必要数值修复。

`predict_pooled`／`select` 可显式接收 `fit_settings=FitSettings(...)`，两求解器均使用登记的参数、步数与学习率；设置随回放身份保存。当前主实验由 experiments.transfer_settings.native_settings() 固定六参数、十步配置；[transfer_settings.py](../../../experiments/transfer_settings.py) 直接登记固定默认配置，不依赖独立开发实验。

`select(device="cuda")` 在隔离的选择器进程内复制基础模型并转入 GPU；调用方的基础模型不变。正式入口由 [parallel.py](../../../experiments/parallel.py) 的 `POOLED_DEVICE="cuda"` 传递设备，同一 GPU 池只运行一个 Pooled 任务；底层函数默认 CPU，供独立数值对照使用。两种求解器均保留 float64 与原联合似然，资源记录包含同步计时和 GPU 分配显存峰值。恢复要求设备与已登记协议一致。小型合成响应的 CPU／CUDA 概率、拟合参数和反馈隔离见 [test_pooled_cuda.py](../../../tests/baselines/test_pooled_cuda.py)；完整历史规模的成本见 [E01 结果](../../../results/k_shot_adaptation/README.md)。
