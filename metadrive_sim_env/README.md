# MetaDrive 仿真平台（暂不使用）

本目录保留已有 MetaDrive 仿真、场景、安全语义、Risk Mining 和 Formal Teacher 实现。上一轮误删的文件已从当前 Git 版本恢复。当前 SRD-TNP-BQD 实验仍只使用 Highway-env 的 S01，不运行本目录的训练或实验。

MetaDrive 被测控制器保留在 `sut_algorithms/metadrive/`。

|目录|职责|
|---|---|
|`scenario/`、`map/`、`control/`、`safety/`|场景、地图、车辆控制与安全语义|
|`context/`、`failure/`、`evaluation/`|轨迹特征、失效判定与评价|
|`mining/`、`formal/`|已有 Risk Mining 和 Formal Teacher 实现|
|`configs/`、`scripts/`|原配置与数据／评价入口，当前暂不执行|
|`tests/`|仿真与方法契约测试|

保留平台的单独测试命令：

```powershell
conda activate metadrive
python -B -m pytest metadrive_sim_env/tests -q -p no:cacheprovider
```

旧 `results/metadrive/` 实验结果保持已清理状态。原脚本仍保留其输出配置；未来明确启用平台后才会重新生成结果。恢复记录见[目录修正](../results/srd_tnp_bqd/audit/layout_correction.json)。
