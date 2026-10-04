# MetaDrive 仿真平台（暂不使用）

本目录保留 MetaDrive 仿真、场景、安全语义、Risk Mining 和 Formal Teacher 实现，供驾驶规划算法研究使用。当前[历史引导风险测试](../methods/history_guided_testing/README.md)在 highway-env 的切入、急刹两类场景上运行，本目录的训练与实验尚未启用。

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

旧 `results/metadrive/` 实验结果已清理。原脚本保留其输出配置，启用平台后可重新生成结果。
