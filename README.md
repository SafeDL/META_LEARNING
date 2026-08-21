# META_LEARNING

本工程保留一个 active MVR 路线与两条可复现的 legacy 基线：

| 模块 | 用途 | 结果 |
| --- | --- | --- |
| [`meta_testing/`](meta_testing/) | 地图感知、双时间尺度、few-shot SUT vulnerability 场景挖掘（active MVR） | 尚无性能结论；必须通过 G1–G8 |
| [`sac_scenario_mining/`](sac_scenario_mining/) | 固定 on-ramp 逻辑场景的 SAC 对抗场景挖掘 | [`results/sac_scenario_mining/`](results/sac_scenario_mining/) |
| [`pearl_learning/`](pearl_learning/) | 历史 merge-only PEARL 少样本适应（legacy baseline） | [`results/pearl_learning/`](results/pearl_learning/) |

`docs/` 保存当前实验说明与后续研究目标。每个模块的配置、运行命令、指标定义和结果边界以模块 README 为准。

## 结果保留原则

- SAC 按各模块协议保留配置、必要日志与可追溯 held-out 证据。
- PEARL 已切换到当前完整协议；所有旧 checkpoint/指标均已删除，目前只保留冻结 taskbook/casebook 输入，尚无当前实现的性能结论。
- PEARL 保留选定正式模型、冻结 taskbook/casebook、manifest 和最终机器可读汇总。
- 中间 validation 的重复 case 表、逐 episode 记录、top-k 动作轨迹、smoke 模型和临时 checkpoint 不纳入长期结果；它们均可由保留的输入与脚本重建。

`meta_testing` 的初始 SUT 仅为 IDM/rule-based controller profiles；它不构成对独立 ADS 泛化的主张。旧 PEARL 仅用于可追溯比较，禁止继续添加新方法功能。

## 验证

```powershell
conda run -n metadrive python -m unittest pearl_learning.tests.test_contract
conda run -n metadrive python -m pytest meta_testing/tests -q
conda run -n metadrive python -m compileall -q pearl_learning sac_scenario_mining
```
