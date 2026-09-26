# 实验结果

本目录只保存正式实验结果、测量银行、评价报告及复核所需的协议和历史源码快照。运行时缓存和临时日志不在此维护；被正式结果取代但仍需留存的试跑已移至 [`archives/experimental_results/`](../archives/experimental_results/README.md)。

| 目录 | 内容 |
| --- | --- |
| [`highway_replications/`](highway_replications/README.md) | Highway-env 论文复现的共享响应库、各方法结果、统一评价和 SUT 筛选证据 |
| [`metadrive/`](metadrive/README.md) | MetaDrive Risk Mining 与 Formal Teacher 的源数据和评价记录；独立于当前 FBRT |
| `method_chains/` | FBRT、CoRe-Mine 及其他组合方法的结果；按方法分目录保存 |

当前 FBRT 的完整离线回放与方法比较见 [`algorithm_comparison.md`](method_chains/failure_memory_regression/repair_exploit_v3_fullbank/algorithm_comparison.md)，独立交互验证见 [`validation_report.md`](method_chains/failure_memory_regression/interaction_holdout_age080/validation_report.md)。`method_chains/` 作为历史结果路径保留，以便冻结清单中的路径和哈希仍可核对；活动源码位于 `methods/`。
