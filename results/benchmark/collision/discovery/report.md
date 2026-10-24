# 历史响应驱动的碰撞发现：执行结果

阶段 A：passed；核对 48 份完整历史/开发执行档案；新增物理调用 0。

阶段 B：not_passed。主矩阵 440 项，额外带宽试验 80 项，共 66,560 次等效查询。
选择带宽：0.3；最强开发基线：fsbo。

| 方法 | 平均去重碰撞数 | 平均召回率 | 发现 AUC |
|---|---:|---:|---:|
| fsbo | 36.600 | 95.47% | 22.048 |
| gp_collision | 28.675 | 85.05% | 16.031 |
| greedy | 32.725 | 91.64% | 19.922 |
| history_ranking | 21.550 | 75.39% | 12.681 |
| history_response | 32.975 | 91.96% | 20.163 |
| nearest_residual | 30.300 | 87.47% | 17.938 |
| pooled_ft | 35.575 | 94.44% | 22.502 |
| random | 9.375 | 24.73% | 5.011 |
| shared_local | 32.825 | 91.73% | 20.044 |
| source_global | 32.575 | 91.45% | 19.955 |
| source_local | 32.725 | 91.63% | 20.018 |

门槛结果：

- recall_gain_at_least_5pp：未通过。
- collision_count_not_lower：未通过。
- at_least_two_positive_lineages：未通过。
- local_recall_exceeds_global：通过。
- local_collision_count_not_lower：通过。

数量与召回率在同一候选库内成比例；此处不将其作为两份独立证据。
谱系内先汇总，再对谱系等权；零分母任务的召回率不填值。配对分层区间见 pilot/gate.json。

按预定门槛，本轮不进入阶段 C、D、E。该结果是开发证据，不是独立目标上的效果结论。
