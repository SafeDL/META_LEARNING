# ScenarioFuzz：图模型与场景变异

本方法用历史记录训练图风险模型，生成并筛选变异场景；根据已查询场景的碰撞和驾驶质量反馈调整搜索种子与局部变异中心。

依据 Wang 等的 [ScenarioFuzz 全文](https://doi.org/10.1145/3650212.3680344)第3节、算法1、图4和实验配置，以及[作者代码](https://github.com/AtongWang/ScenarioFuzz)提交 `49360794846786b133799a62827528bc4644bc62`。本文保留图风险模型、先全局后邻域的变异、预测过滤、驾驶质量引导和种子调度。场景固定为 HighwayEnv 的公开直线路网和有限候选库；天气、行人及地图采集不属于该任务。

## 对应关系与适配

| 原文机制 | 本实现 |
|---|---|
| §3.2–3.3 地图、任务及背景对象路径图 | `graphs.py:graph`：公开道路路点、主车任务和预定背景路径，不读取执行轨迹 |
| §3.5、图4 双图注意力流与天气编码 | `model.py:ScenarioRiskNetwork`：两层节点GAT、两层边GAT、天气MLP、均值读出；隐藏64、4头、BN、dropout 0.5 |
| 历史数据、80/20划分、1000轮训练 | `model.py:train`：全来源为8个历史系统、4096条记录；按场景坐标分组划分，Adam 0.001、batch 16，验证焦点损失选择检查点 |
| 算法1第16–24行：变异与SEM过滤 | `pool.py:run`：每轮100个变异，首轮全局、后续以成功执行的最低驾驶分数场景为中心，所有连续属性在±5个公开精度步内变异 |
| §3.5 置信度大于0.5、排序后执行 | 每轮最多3次付费执行，每个种子最多3轮；碰撞立即结束该种子的本次搜索 |
| §3.6 驾驶质量分数 | `driving_quality.py`及评估端`experiments/domain_risk.py`：同次已付费安全执行的急加减速、急转、模糊过/欠转和最小车心距惩罚 |
| 种子频率轮盘重入队列 | 碰撞后以 `1−该种子选取次数/总搜索次数` 重入队列；队列空时随机选公开种子，与作者`config.py:check_seed_entropy`一致 |
| 每新增1000条训练数据更新SEM | 全来源历史4096条、留家族历史3072条；128次查询均未达到下一个1000条更新点，预算内保持图模型固定 |

图的原6维节点特征扩展为8维，增加公开预定时刻和速度，保留切入/切出任务中不能由静态几何完全表达的事件差别。天气8维固定为0。归一化仅拟合历史训练部分；常数维使用单位尺度。

连续变异通过公开特征的欧氏距离投影至最近未查询候选；候选标识处理同分。每轮投影去重，执行后移出候选集。共同随机初始化8次计入128次预算，初始化不替代变异周期。图模型可提前计算公开候选概率，其耗时单列。

原实现遇到不足3个预测阳性时重新变异，连续6次失败后结束该种子。本适配在同样6次尝试后，以当轮最高概率的其余候选补足固定预算；投影候选不足时再从剩余公开库补齐。逐查询记录尝试次数、阳性数量和阈值补齐标志，报告补齐次数。这是固定预算适配。碰撞为本任务唯一错误标志。

驾驶质量由评估端从已查询轨迹计算，选择器只接收标量。保留作者的物理指标和模糊控制规则；修正作者调用中将横向速度传给方向盘角度差的参数错误。仿真20 Hz，速度转为km/h、转向转为度，短于等于1秒的安全执行不计分，与原逻辑一致。有限库安全执行均满足时长要求。

## 作者实现核查

作者边特征流复用节点邻接及节点批次偏移，会使一个场景的预测随同批其他场景改变。本实现使用边的线图邻接和边批次索引。作者输出sigmoid后再传入`BCEWithLogits`焦点损失；本实现向损失传入logit，在推断时仅做一次sigmoid。代码审计记录了这两项修正及其数值影响。

同权重、同图且节点/边拓扑一致时，作者网络与本网络输出误差为0；作者跨批次场景顺序误差约0.00193，本实现误差为0。对应记录见[作者代码审计](../../../experiment_outputs/benchmark/collision/discovery/domain_comparison/scenario_fuzz_author_audit.json)。原生模糊驾驶分数的30组独立对照误差不超过1e-12，全部12,288条历史/开发执行的分数接口通过[核查](../../../experiment_outputs/benchmark/collision/discovery/domain_comparison/driving_quality_audit.json)。

## 构建与验证

```powershell
python -B -m testing_methods.baselines.scenario_fuzz.train
python -B -m experiments.domain_scenario_fuzz_validation
python -B -m pytest tests/experiments/test_scenario_fuzz.py tests/experiments/test_scenario_fuzz_pool.py tests/experiments/test_driving_quality.py -q -p no:cacheprovider
```

上述 `train` 入口构建开发比较的切入、切出模型，输出至 `experiment_outputs/scenario_fuzz/collision/discovery/domain_comparison/stage_a/models/`。原独立验证模型保存于 [independent_models](../../../experiment_outputs/scenario_fuzz/collision/independent_models/)，包括5类场景、5种来源范围，共25个模型。

开发比较和独立验证均已完成；独立验证包含全来源、严格留家族条件各200项 ScenarioFuzz 运行。结果见 [独立验证报告](../../../experiment_outputs/benchmark/collision/discovery/independent_validation/report.md)。
