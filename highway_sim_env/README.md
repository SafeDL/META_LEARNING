# Highway-env 仿真与共享数据

本目录维护本文方法在 highway-env 上的轻量实现，并提供当前复现实验共用的
Cut-in 场景和响应库。SUT 统一位于根目录 `sut_algorithms/highway_env/`。
AdaTE、DETOUR 和融合链可以读取这些稳定接口，但
不得把各自的算法、配置或结果写回本目录。

主要目录：

- `envs/`：可直接执行的 highway-env 场景；
- `data/`：候选场景和响应库；
- `mining/`：低秩先验、诊断采样和后验更新；
- `tests/`：共享仿真契约。
- `build_spec.py`：构建配置和稳定指纹；
- `s01_parameters.py`、`configs/scenario_parameter_space.yaml`：S01 坐标、边界和有效响应标签。
- `envs/unified_env.py`：A、D 共用的 `UnifiedHighwayEnv`，按 20 Hz 执行控制器和脚本车辆；
- `envs/scripted_vehicle.py`、`envs/safety_metrics.py`：背景车辆事件、净间距和 TTC。

S01 直接读取 A、D 统一的四轴 `active_parameters` 字段。控制器构建指纹在同一构建对象中只计算一次；它用于响应身份。旧的 `fbrt_env.py`、`fbrt_scenarios.py`、`fbrt_training_env.py` 与 `fbrt_parameters.py` 已移至 [`archives/retired_research_chains/highway_sim_env/`](../archives/retired_research_chains/highway_sim_env/)。

```powershell
conda run -n metadrive python -m pytest highway_sim_env/tests -q -p no:cacheprovider
conda run -n metadrive python -m highway_sim_env.data.generate_anchor_bank
conda run -n metadrive python -m highway_sim_env.data.response_bank
```
