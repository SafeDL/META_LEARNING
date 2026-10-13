# 历史分支恢复归档

当前实现见 [CG-BRT](../methods/concordance_gated_bayesian_risk_transfer/README.md)。旧分支以压缩包保存，清单与核验记录保留为 JSON。

| 归档 | 内容 |
|---|---|
| [historical_branch_cleanup.zip](historical_branch_cleanup.zip) | 旧单目标流程、结果与展开历史工程 |
| [driving_benchmark_trials.zip](driving_benchmark_trials.zip) | 基准中未保留的历史轨迹 |
| [risk_method_convergence.zip](risk_method_convergence.zip) | 旧碰撞主链、混合模型探索与阶段结果 |
| [retired_method_trials.zip](retired_method_trials.zip) | 退役方法分支 |
| [method_selection.zip](method_selection.zip) | 早期方法与基线迁移快照 |
| [retired_experiments.zip](retired_experiments.zip) | 早期实验记录 |

恢复时解压到独立目录，压缩包内保留原相对路径：

```powershell
Expand-Archive -LiteralPath archives/risk_method_convergence.zip -DestinationPath archives/restored_risk_method
```

`retired_method_trials.zip`、`method_selection.zip`、`retired_experiments.zip` 均超过 100 MiB，仅在本地保留；其他归档和清单允许 Git 跟踪。

运行基线入口的来源核验需要本地 `method_selection.zip`；当前主方法的比较与核验不需要它。当前源码快照另存于 [source_inputs.zip](../results/concordance_gated_bayesian_risk_transfer/source_inputs.zip)。
