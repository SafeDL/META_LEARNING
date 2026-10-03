# 归档

本目录保存独立研究链的历史实现与结果。当前论文方法位于[SRD-TNP-BQD](../methods/srd_tnp_bqd/README.md)，共享A/D数据位于[原始基准库](../benchmarks/s01/README.md)。旧九基线包及对应比较结果已删除，不属于本目录的独立复现归档。

| 目录 | 内容 |
| --- | --- |
| `pearl_learning/` | 仅用于合流任务的 PEARL 历史基线、重建脚本和契约测试 |
| `sac_scenario_mining/` | 早期 SAC 场景挖掘基线 |
| [`experimental_results/`](experimental_results/README.md) | 更早的中间实验和无效尝试 |
| [`retired_research_chains/`](retired_research_chains/README.md) | 本轮退出活动目录的旧方法实现及其完整结果 |

AdaTE、DETOUR、FST、ScenarioFuzz 的独立复现及其专用仿真／控制器依赖现统一归入 `retired_research_chains/`，保持原相对目录结构。它们具有独立复现价值，但不属于当前 S01 九基线主比较。MetaDrive 旧实验结果保持已清理状态；其仿真代码与专用 SUT 已恢复到根目录，暂不使用。此前复现的 Highway-env ADS 及权重也在根目录保留；归档内的源码快照继续服务历史复现。

在仓库根目录验证 PEARL 归档代码：

```powershell
conda run -n metadrive python -m pytest archives/pearl_learning/tests -q
```

归档结果与基线源码分别放在各自子目录；当前 `results/` 中没有 PEARL 正式结果目录。
