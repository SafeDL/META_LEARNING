# Highway-env 论文复现结果

本目录保存四种论文方法的正式复现结果及同预算评价。`replications/benchmark.yaml` 指向这里的共享响应库；各方法的复现脚本和部分测试也读取这些文件。它们是独立方法对照，不是当前 FBRT 冻结银行的输入。

| 目录 | 内容 |
| --- | --- |
| `shared/` | 192 个平衡场景、六种 SUT 的 1,152 条真实仿真响应及清单 |
| `adate/` | AdaTE 的论文内实验和共享候选池实验 |
| `detour/` | DETOUR 的 D1/D2 协议、轨迹、图表和动图 |
| `fst/` | FST 相似度模型、50 次重复评价、图表和动图 |
| `scenariofuzz/` | 六目标论文对齐实验、模型与回放记录 |
| `evaluation/` | 跨方法记录、覆盖矩阵、图表、统一报告和校验结果 |
| `sut_selection/` | IDM+MOBIL、VI-TTC、MCTS-CV、PPO-ECE 的筛选证据与共同场景响应 |

跨方法可比结果见 [`evaluation/report.md`](evaluation/report.md)。失效发现和碰撞率估计分别评价，不把不同任务的指标合并排名；各论文方法的单独结论与偏差仍以各自报告为准。
