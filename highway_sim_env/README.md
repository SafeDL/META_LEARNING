# Highway-env 仿真接口

本目录提供车辆执行、场景和安全测量接口。当前实验使用切入与前车急刹两类场景，以及[六个被测系统](../sut_algorithms/README.md)。

| 文件 | 作用 |
|---|---|
| [build_spec.py](build_spec.py) | 控制器构建配置 |
| [envs/unified_env.py](envs/unified_env.py)、[envs/scripted_vehicle.py](envs/scripted_vehicle.py) | 物理执行与脚本车辆 |
| [envs/cutin_env.py](envs/cutin_env.py)、[envs/external_cutin.py](envs/external_cutin.py) | 切入环境与外部策略接口 |
| [envs/single_lane_longitudinal.py](envs/single_lane_longitudinal.py) | 单车道执行 |
| [envs/safety_metrics.py](envs/safety_metrics.py) | 间距与 TTC 测量 |

当前 CG-BRT 比较使用[锁定响应池](../results/driving_benchmark/README.md)。场景参数见[公共场景代码](../methods/history_guided_testing/scenarios.py)，比较与核验入口见[实验说明](../experiments/concordance_gated_bayesian_risk_transfer/README.md)。
