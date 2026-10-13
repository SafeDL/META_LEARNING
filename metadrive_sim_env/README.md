# MetaDrive 仿真平台

本目录保留 MetaDrive 场景、车辆控制、安全评价、风险搜索和形式化测试实现。当前 [CG-BRT](../methods/concordance_gated_bayesian_risk_transfer/README.md) 实验使用 highway-env，本平台未纳入当前比较。

| 目录 | 作用 |
|---|---|
| `scenario/`、`map/`、`control/`、`safety/` | 场景、地图、车辆控制与安全测量 |
| `context/`、`failure/`、`evaluation/` | 轨迹特征、失效判定与评价 |
| `mining/`、`formal/`、`training/` | 风险搜索、形式化测试与训练 |
| `configs/`、`scripts/`、`tests/` | 配置、运行脚本与测试 |

控制器位于 `sut_algorithms/metadrive/`。单独检查平台代码：

```powershell
conda activate metadrive
python -B -m pytest metadrive_sim_env/tests -q -p no:cacheprovider
```
