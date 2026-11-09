# ScenarioFuzz：图风险模型与场景变异

本方法从历史记录训练图风险模型，筛选变异候选，再用已查询的碰撞和驾驶质量反馈更新种子与局部搜索中心。

依据 Wang 等的 [ScenarioFuzz 全文](https://doi.org/10.1145/3650212.3680344)第 3 节、算法1、图4及实验设置，核对[作者代码](https://github.com/AtongWang/ScenarioFuzz)提交 `49360794846786b133799a62827528bc4644bc62`。

| 论文机制 | 实现 |
|---|---|
| §3.2–3.3 场景图 | [graphs.py](graphs.py)：公开路点、自车任务和预设背景路径 |
| §3.5、图4 风险网络 | [model.py](model.py)：节点／边双 GAT、天气编码和风险输出 |
| 历史训练与预测过滤 | `model.train`、[pool.run](pool.py)：历史训练后筛选预测概率大于 0.5 的变异 |
| 算法1第16–24行 | `pool.run`：先全局、后邻域变异及种子调度 |
| §3.6 驾驶质量 | [driving_quality.py](driving_quality.py)、[评估端](../../../experiments/domain_risk.py)：从本次安全执行派生分数 |

## 图模型与反馈

图使用公开直线路网和预设运动，节点特征扩为 8 维，加入事件时间和速度；天气 8 维固定为零。默认模型为节点与边各两层 GAT，隐藏 64、4 头、BatchNorm、dropout 0.5。按坐标分组做 80%／20% 历史划分，Adam 0.001、批量 16 训练 1,000 轮，以验证焦点损失选择模型。归一化只拟合历史训练数据。

驾驶质量来自已付费安全轨迹，包括急加减速、急转、模糊过／欠转和车心距惩罚；长度不超过 1 秒的安全执行不计分。转速与角度使用作者规则对应的单位。碰撞统一作为失效标签。

实现采用边的线图邻接与独立批次索引，向焦点损失传入 logit，并在推断时做一次 sigmoid；同时修正作者驾驶质量调用中的横向速度／转向参数混用。

## 有限候选库适配

每轮生成 100 个变异，首轮全局采样，后续根据已执行安全场景的最低驾驶分数确定邻域中心，在各参数公开精度的 ±5 步内变异。提案投影到最近未查询候选并去重。每轮最多执行 3 个候选，每个种子最多 3 轮，碰撞结束该种子的本次搜索。

预测阳性不足时最多重试 6 次，再按同轮预测风险补齐预算；投影候选不足时从剩余公开库补齐。碰撞后的种子按 `1−选取次数/总搜索次数` 重入队列，队列空时随机取种子。查询记录保存补齐标志、轮次和种子信息。

初始化计入总预算。当前比较预算为 200，未达到每新增 1,000 条数据更新模型的时点，查询阶段使用固定图模型。显式参数任务通过一致的 `graph_fn`、`graph_definition` 和 `domain` 登记图与变异定义。

## 接入与验证

[model.py](model.py)提供 `train/load`，[pool.py](pool.py)提供 `run`。历史分类器由 [experiments.pilot](../../../experiments/pilot.py)构建，保存在 `artifacts/runs/joint_response/pilot/models/<场景>/scenario_fuzz/classifier.pt`。

~~~powershell
python -B -m pytest tests/experiments/test_scenario_fuzz.py tests/experiments/test_scenario_fuzz_pool.py tests/experiments/test_driving_quality.py -q
~~~

结果见[论文报告](../../../artifacts/runs/joint_response/paper/reports/results.md)。
