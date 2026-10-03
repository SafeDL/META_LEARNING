# Highway-env 仿真

当前实验使用原 S01 cut-in 的四维坐标、初始条件与 20 Hz 物理执行。A/D、五个历史 IDM 源及固定 FVDM 目标参数不变。此前复现的其他 ADS 所需通用执行器和外部策略接口同时保留，暂不参加当前比较。

|文件|职责|
|---|---|
|`build_spec.py`|原控制器构建配置与响应身份|
|`s01_parameters.py`、`configs/s01.yaml`|当前 S01 四轴坐标、参数边界和有效标签|
|`envs/unified_env.py`、`scripted_vehicle.py`|原物理步进、轨迹记录与脚本化车辆|
|`envs/cutin_env.py`、`external_cutin.py`|已有切入环境与外部 ADS 执行接口，暂不使用|
|`envs/single_lane_longitudinal.py`|保留的单车道执行接口，暂不使用|
|`envs/safety_metrics.py`|净间距和 TTC|

控制器由 [sut_algorithms](../sut_algorithms/README.md) 提供。独立旧研究的数据／挖掘接口与冻结协议仍保存在[历史归档](../archives/retired_research_chains/README.md)。恢复通用环境不改变当前 S01 的控制公式、执行次序或测量接口。

当前主链的回归测试：

```powershell
conda activate metadrive
python -B -m pytest -q -p no:cacheprovider
python -B -m methods.srd_tnp_bqd.audit
```
