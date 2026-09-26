# 中间实验结果归档

本目录保留从 `results/method_chains/` 移出的中间尝试，以便必要时核对原始数据；它们不参与当前方法排名，也不是活动代码默认读取的结果。归档采用原方法目录层级，但文件内冻结清单仍记录迁移前的路径。若要重跑历史协议，应先将相应目录恢复到原位置。

| 路径 | 归档原因 |
| --- | --- |
| `method_chains/failure_memory_regression/interaction_v1/`、`interaction_v2/` | 两轮交互开发银行未形成有效的 age 目标回归，由独立的 age080 验证取代 |
| `method_chains/failure_memory_regression/memory_kde_v3/`、`memory_weight_v3/`、`memory_full_cov_v3/` | 三种离线原型均未优于当前保留的方法 |
| `method_chains/core_mine/studies/evidence_gate_invalidated/` | 首轮候选集不匹配，结论无效；已由修正实验取代 |
| `method_chains/core_mine/studies/corrected_*_pilot/`、`corrected_pilot/` | 后续正式实验之前的场景试跑 |

当前 FBRT 结果索引见 [`results/method_chains/failure_memory_regression/README.md`](../../results/method_chains/failure_memory_regression/README.md)。
