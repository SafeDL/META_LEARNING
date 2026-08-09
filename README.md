# META_LEARNING

本工程保留两条可复现的自动驾驶安全关键场景实验线：

| 模块 | 用途 | 结果 |
| --- | --- | --- |
| [`sac_scenario_mining/`](sac_scenario_mining/) | 固定 on-ramp 逻辑场景的 SAC 对抗场景挖掘 | [`results/sac_scenario_mining/`](results/sac_scenario_mining/) |
| [`pearl_learning/`](pearl_learning/) | 未见逻辑汇入场景的 PEARL 少样本适应 | [`results/pearl_learning/`](results/pearl_learning/) |

`docs/` 保存当前实验说明与后续研究目标。每个模块的配置、运行命令、指标定义和结果边界以模块 README 为准。

## 结果保留原则

- SAC 保留三组训练配置、日志、每个 validation checkpoint 的紧凑汇总，以及完整 held-out 结果和回放证据。
- PEARL 保留选定正式模型、冻结 taskbook/casebook、manifest 和最终机器可读汇总。
- 中间 validation 的重复 case 表、逐 episode 记录、top-k 动作轨迹、smoke 模型和临时 checkpoint 不纳入长期结果；它们均可由保留的输入与脚本重建。

## 验证

```powershell
conda run -n metadrive python -m unittest pearl_learning.tests.test_contract
conda run -n metadrive python -m compileall -q pearl_learning sac_scenario_mining
```
