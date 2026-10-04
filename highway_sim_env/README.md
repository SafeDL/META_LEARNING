# Highway-env 仿真

当前主链研究切入和前车急刹两类场景，历史库和独立目标库均为每类 1024 个。场景由 [当前方法](../methods/history_guided_testing/README.md)的 `scenarios.py` 统一定义，参数在 `config.py`。目标是安全间距 FVDM，初速 25 m/s、期望速度 23 m/s；12 s、20 Hz、双车道。

|文件|职责|
|---|---|
|build_spec.py|控制器构建配置|
|envs/unified_env.py、scripted_vehicle.py|统一物理执行、轨迹与脚本车辆|
|envs/cutin_env.py、external_cutin.py|其他切入环境与外部策略接口|
|envs/single_lane_longitudinal.py|单车道执行接口|
|envs/safety_metrics.py|间距与 TTC|

控制器见 [sut_algorithms](../sut_algorithms/README.md)。运行 `conda activate metadrive` 后使用 `python -B -m methods.history_guided_testing.run`；回归检查使用 `python -B -m pytest -q -p no:cacheprovider`。
