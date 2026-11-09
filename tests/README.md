# 回归测试

本目录检查方法推断、反馈权限、仿真执行和结果统计。

| 目录 | 检查范围 |
|---|---|
| [history_transfer/](history_transfer/) | 联合推断、在线适配、来源权重、顺序训练及设备一致性 |
| [experiments/](experiments/) | 场景、控制器、查询隔离、付费归档、基线、调度和指标 |
| [baselines/](baselines/) | 公共初始化与预算约束 |

从项目根目录运行全部检查：

~~~powershell
python -B -m pytest tests simulator/metadrive/tests -q
~~~

也可指定单个文件或目录。`tests/` 检查 HighwayEnv 与测试方法；MetaDrive 的检查位于 [simulator/metadrive/tests/](../simulator/metadrive/tests/)。

配置统一在 [pyproject.toml](../pyproject.toml)。缓存和临时执行库写入 `artifacts/cache/`，检查后可清理；最近一次完整回归记录见 [runs_cleanup_regression.xml](../artifacts/maintenance/runs_cleanup_regression.xml)。
