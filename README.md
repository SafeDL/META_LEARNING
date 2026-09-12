# META_LEARNING

面向黑盒驾驶控制器的可迁移少样本脆弱场景挖掘研究代码库。项目按仿真器
隔离实现：MetaDrive 用于完整的地图感知场景实验，Highway-env 用于快速的
DIVA-Mine MVP 机制验证；两者的代码、测试和结果互不覆盖。

| 目录 | 用途 |
| --- | --- |
| `mvr/metadrive/` | MetaDrive 的 MVR、DIVA-Mine / DIVA-Former 实现、脚本和测试 |
| `mvr/highway/` | Highway-env 的 DIVA-Mine MVP 实现、脚本和测试 |
| `archives/` | 冻结的 PEARL 与 SAC 历史基线 |
| `docs/` | 方法、实验设计与执行说明 |
| `results/metadrive/` | MetaDrive 的可追溯实验工件 |
| `results/diva_highway/` | Highway-env MVP 指标、图表与 GIF 回放 |

## 复现与验证

使用根目录的 `environment.yml` 创建 `metadrive` Conda 环境后，在仓库根目录运行：

```powershell
conda run -n metadrive python -m pytest mvr/metadrive/tests mvr/highway/tests -q
conda run -n metadrive python -m pytest archives/pearl_learning/tests -q
conda run -n metadrive python -m compileall -q mvr/metadrive mvr/highway archives/pearl_learning archives/sac_scenario_mining
```

重新构建 Highway-env 的响应库、指标、图表与 GIF：

```powershell
conda run -n metadrive python -m mvr.highway.scripts.run_diva_highway_mvp --rebuild-bank
conda run -n metadrive python -m mvr.highway.scripts.render_diva_highway_mvp
```

Highway-env 的具体实验协议和已执行结果分别见
[`docs/highway_env_DIVA_Mine_MVP_experiment_design.md`](docs/highway_env_DIVA_Mine_MVP_experiment_design.md)
和 [`docs/highway_env_DIVA_Mine_MVP_execution.md`](docs/highway_env_DIVA_Mine_MVP_execution.md)。
