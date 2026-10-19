# highway-env 仿真环境

本目录用于当前 GCRT 和基线的实测数据采集，支持车辆切入、前车紧急制动及相应风险测量。驾驶控制器在 [controllers/highway](../../controllers/README.md)。

| 文件 | 用途 |
|---|---|
| [build_spec.py](build_spec.py) | 控制器构建配置 |
| [envs/unified_env.py](envs/unified_env.py)、[envs/scripted_vehicle.py](envs/scripted_vehicle.py) | 统一物理执行与脚本车辆 |
| [envs/cutin_env.py](envs/cutin_env.py)、[envs/external_cutin.py](envs/external_cutin.py) | 切入环境与外部策略接口 |
| [envs/single_lane_longitudinal.py](envs/single_lane_longitudinal.py) | 单车道跟驰与制动执行 |
| [envs/safety_metrics.py](envs/safety_metrics.py) | 间距与碰撞时间测量 |

场景参数见[场景定义](../../methods/baselines/hgt/scenarios.py)，采集工具见 [benchmark](../../benchmark/README.md)，已有响应见 [data](../../data/README.md)。GCRT 当前实验直接回放已有实测记录。
