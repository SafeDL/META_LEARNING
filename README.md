# META_LEARNING

本仓库实现面向黑盒驾驶控制器的分层、地图条件化少样本脆弱性场景挖掘（MVR）。结论仅覆盖固定 MetaDrive 地图上的 held-out IDM controller profile；不宣称对未知道路、独立 ADS 或真实车辆的泛化。

| 目录 | 作用 |
| --- | --- |
| `meta_testing/` | 当前 active MVR 方法、配置和测试 |
| `pearl_learning/` | 冻结的 merge-only PEARL 基线 |
| `sac_scenario_mining/` | 冻结的 SAC 场景挖掘基线 |
| `docs/` | 方法、实验与维护说明 |
| `results/` | 可追溯的最终 JSON/CSV 指标和 manifest |

## 复现

使用仓库根目录的 [`environment.yml`](environment.yml) 创建 `metadrive` Conda 环境，然后运行：

```powershell
conda run -n metadrive python -m pytest meta_testing/tests -q
conda run -n metadrive python -m pytest pearl_learning/tests -q
conda run -n metadrive python -m compileall -q meta_testing pearl_learning sac_scenario_mining
```

MVR 的正式入口只有：

```powershell
python -m meta_testing.scripts.validate_mvr --output results/validation/g3.json
python -m meta_testing.scripts.train_mvr --output results/meta_testing/run_001
python -m meta_testing.scripts.evaluate_mvr --checkpoint results/meta_testing/run_001/outer.pt --output results/meta_testing/run_001/evaluation.json
```

训练严格按 `inner_pretrain → posterior → inner_latent_calibration → outer` 执行。中间 checkpoint、rollout 和日志可由 manifest 与配置重建，不应提交。
