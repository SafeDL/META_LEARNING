# 变维功能场景开发候选

`scenario_manifest.jsonl` 按维数对 8 个已接线功能场景族取 scrambled Sobol 点：3 维 1,024 个、4 维各 2,048 个、5 维各 4,096 个，共 19,456 个具体候选。参数维度、范围、独立性与限制见 [`FBRT_Scenario_Parameter_Space_V3.md`](../../../../docs/FBRT_Scenario_Parameter_Space_V3.md)。`protocol.json` 记录配置、目录和清单指纹及每族数量。

本目录只生成候选，**尚无 NL/PPO 物理响应**，也不是新的确认集。S07、S10–S14 仅有参数空间提案，未被生成进入本清单。既有 `single_context_grid_development/` 二维清单和 `role_gated_generalization_confirmation/` 冻结清单各自保留，不混合统计。

在仓库根目录重新生成并验证清单：

```powershell
python -m methods.failure_memory_regression.prepare_scenario_sampling
```

本目录是可再生成的开发候选，不是冻结确认集。大型清单只保留在本地，不纳入 Git；改变采样设置后重新运行上述命令即可更新。
