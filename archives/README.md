# 历史基线

本目录保存已冻结的历史对照实现，不承载当前 FBRT 方法代码或正式结果。当前仿真底座分别位于 `metadrive_sim_env/` 和 `highway_sim_env/`，活动方法位于 `methods/`。

| 目录 | 内容 |
| --- | --- |
| `pearl_learning/` | 仅用于合流任务的 PEARL 历史基线、重建脚本和契约测试 |
| `sac_scenario_mining/` | 早期 SAC 场景挖掘基线 |
| [`experimental_results/`](experimental_results/README.md) | 从正式结果区移出的中间实验和无效尝试，原始文件可恢复 |

在仓库根目录验证 PEARL 归档代码：

```powershell
conda run -n metadrive python -m pytest archives/pearl_learning/tests -q
```

归档结果与基线源码分别放在各自子目录；当前 `results/` 中没有 PEARL 正式结果目录。
