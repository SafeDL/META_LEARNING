# META_LEARNING

本仓库实现面向黑盒驾驶控制器的可迁移少样本脆弱性场景挖掘：在多几何 MetaDrive 任务上训练，并以 R1–R4 分离评估 SUT 与道路几何 OOD。当前不宣称独立 ADS 或真实车辆泛化。

| 目录 | 作用 |
| --- | --- |
| `mvr/` | 当前 active MVR 方法、配置和测试 |
| `archives/pearl_learning/` | 归档的 merge-only PEARL 基线 |
| `archives/sac_scenario_mining/` | 归档的 SAC 场景挖掘基线 |
| `docs/` | 方法、实验与维护说明 |
| `results/` | 可追溯的最终 JSON/CSV 指标和 manifest |

## 复现

使用仓库根目录的 [`environment.yml`](environment.yml) 创建 `metadrive` Conda 环境，然后运行：

```powershell
conda run -n metadrive python -m pytest mvr/tests -q
conda run -n metadrive python -m pytest archives/pearl_learning/tests -q
conda run -n metadrive python -m compileall -q mvr archives/pearl_learning archives/sac_scenario_mining
```

MVR 的正式入口只有：

```powershell
python -m mvr.scripts.validate_mvr --output results/validation/g3.json
python -m mvr.scripts.build_taskbook --output mvr/configs/taskbook.json
python -m mvr.scripts.train_mvr --output results/mvr/run_001
python -m mvr.scripts.evaluate_mvr --checkpoint results/mvr/run_001/outer.pt --output results/mvr/run_001/evaluation.json
```

训练严格按 `inner_pretrain → posterior → inner_latent_calibration → outer` 执行。中间 checkpoint、rollout 和日志可由 manifest 与配置重建，不应提交。
