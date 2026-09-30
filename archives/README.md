# 归档

本目录保存已冻结的历史对照实现、旧方法链及其结果。当前 RAS-FRT 方法位于 `methods/ras_frt/`，正式 A、D 结果位于 `results/method_chains/ras_frt/`。

| 目录 | 内容 |
| --- | --- |
| `pearl_learning/` | 仅用于合流任务的 PEARL 历史基线、重建脚本和契约测试 |
| `sac_scenario_mining/` | 早期 SAC 场景挖掘基线 |
| [`experimental_results/`](experimental_results/README.md) | 更早的中间实验和无效尝试 |
| [`retired_research_chains/`](retired_research_chains/README.md) | 本轮退出活动目录的旧方法实现及其完整结果 |

在仓库根目录验证 PEARL 归档代码：

```powershell
conda run -n metadrive python -m pytest archives/pearl_learning/tests -q
```

归档结果与基线源码分别放在各自子目录；当前 `results/` 中没有 PEARL 正式结果目录。
