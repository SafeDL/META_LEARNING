# BehAVExplor：行为多样性引导的测试

本方法从付费轨迹提取行为特征，以行为新颖性和危险度更新种子能量，再生成变异并投影到未查询候选。

依据 Cheng、Zhou、Xie 的 *BehAVExplor: Behavior Diversity Guided Testing for Autonomous Driving Systems*，ISSTA 2023，488–500，[DOI](https://doi.org/10.1145/3597926.3598072)，[核对全文](https://drive.google.com/file/d/19ZSg8MOQfVD7zs4P09dmUf5waCqhvnk_/view)。作者实现核对版本为 [7ee9cdde](https://github.com/MingfeiCheng/BehAVExplor/tree/7ee9cdde4673cb194eab2da70eecc408cf2a67b7)。

| 依据 | 实现 | 作用 |
|---|---|---|
| §3.2.1，`feature.py` | [motion_trace、temporal_features](behavior.py) | 0.1 秒重采样，10 帧滑窗提取 64 维行为特征 |
| §3.2.1，式(7)–(9) | `BehaviorMiner、trace_distance` | K-means 行为标签与含长度差的 Hamming 距离 |
| §3.2.2，`get_fitness` | [run](pool.py) | 以截断净间距衡量危险度 |
| §3.3，`energy_step/energy_init` | `Seed.update` | 更新种子能量并按能量抽样 |
| 算法2 | `mutate、run` | 高能量局部变异，低能量全域重采样 |

## 行为抽象与种子管理

行为使用速度、带符号加速度模、车体纵横向速度与加速度、偏航角速度和航向，共 8 项属性。每项提取均值、方差、最小、最大、平均变化、平均绝对变化、c3 和 cid_ce，共 64 维；统计公式对应 [tsfresh](https://github.com/blue-yonder/tsfresh/blob/main/tsfresh/feature_extraction/feature_calculators.py)。导数由相邻实测帧计算，尾部窗口重复末状态补齐。

默认聚类数为 10，新颖阈值为 0.4；样本不足时按不同特征行数确定可用簇数。新颖安全行为触发重新聚类并重标注旧轨迹。危险度使用 `clip(clearance, 0, 1)`，越小越危险；碰撞进入发现记录，安全执行按危险度改善或行为新颖性进入种子库。

能量采用作者发布代码：初始安全种子为 1.5，新安全种子为 `max(0, 1+(危险度改善+行为距离)/2)`。局部更新结合有效率与 `tanh(危险度改善/(1−行为距离+1e−10))`，两项权重均为 0.5，再扣除 0.05；无效项为 `−0.1*(1−累计有效率)`。全域变异扣除 0.1，能量下限为零；全零时均匀抽样。这些设置沿用作者代码的有界奖励与种子管理。

## 候选库适配

能量高于 0.5 时局部高斯变异，否则在同类参数域均匀重采样。每个参数以 0.5 概率变异，速度标准差为 6 m/s、距离为 12 m，其他参数为声明范围的 10%。提案先满足公开物理约束，再按归一化距离投影到最近未查询候选，同分按场景 ID。

入口为 `run(candidates, oracle, initial_indices=..., budget=200, seed=...)`，可通过 `domain` 指定参数定义。共同初始化的安全执行构成种子库；若全部碰撞，继续在预算内随机查询，直至出现安全种子或预算耗尽。反馈仅来自当前已付费执行，行为聚类与参数空间多样性指标分别计算。

## 验证入口

~~~powershell
python -B -m pytest tests/experiments/test_behavexplor.py -q
~~~

测试覆盖短轨迹、统计量、含时长的距离、聚类重标注、能量、变异和预算隔离。统一接入见 [experiments.pilot](../../../experiments/pilot.py)，结果见[论文报告](../../../artifacts/runs/joint_response/paper/reports/results.md)。
