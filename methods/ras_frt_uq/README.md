# RAS-FRT-UQ：用 A 的 IDM 历史响应发现 D 中的 FVDM 碰撞

当前保留并作为论文主方法的是 **RAS-FRT-UQ**。它融合原版 RAS-FRT 的历史响应相似度、局部残差与覆盖项，以及 Transfer-UQ 的目标残差后验与不确定性。原版 **RAS-FRT** 仍作为对比方法，不能与融合方法混称。这个目录包含主方法和九方法比较所需的选择器；A、D 数据与比较结果统一存放在 `results/method_chains/ras_frt_uq/`。

历史库 A 有 2,048 个 S01 场景及同一 IDM 模型的五组参数响应。候选库 D 有另外 2,048 个场景及固定 FVDM 控制器的完整 Highway-env 物理响应。九种方法在同一 D 库上各使用最多 200 次可见查询；完整标签仅用于回放后的评价。

```powershell
conda run -n metadrive python -m methods.ras_frt_uq.experiment
```

比较方法为 Random、Farthest-First、Target-GP-UCB、HistoryRank-adapted、History-only、Residual risk-only、RAS-FRT、Transfer-UQ 和 RAS-FRT-UQ。在 10、30、50、100、150、200 次查询处统计发现数和危险参数网格覆盖。HistoryRank-adapted 运行一次，其他方法各用五个固定种子。

| 文件 | 职责 |
| --- | --- |
| `protocol.py`、`banks.py`、`data.py` | 读取 A 的五组响应及 D 的固定候选与 FVDM 标签 |
| `training.py`、`response_encoder.py` | 在 A 上训练响应模型，回放时读取已有权重 |
| `coverage_selector.py`、`transfer_uncertainty.py`、`fusion_selector.py`、`comparison_selectors.py` | 主方法、原版方法和对比选择器 |
| `target_bank.py` | 生成 D 候选场景并执行固定 FVDM |
| `experiment.py` | 200 次预算的九方法回放与汇总 |

既有 A、D 结果及早期 100 次实验的协议和源码快照见 [结果目录](../../results/method_chains/ras_frt_uq/README.md)。活动回放直接读取场景、响应和权重；原始协议中的 SHA-256 字段仍保留在既有结果文件中，但不再作为每次运行的前置校验。融合权重曾参考同一个 D 库的结果，因此 D 上的优势属于开发阶段证据。
