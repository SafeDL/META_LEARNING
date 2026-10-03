# 实验结果

算法结果直接位于 `results/` 的子目录；共享场景、原始响应和实验协议统一存放在[benchmarks/s01](../benchmarks/s01/README.md)。

|目录|内容|
|---|---|
|[srd_tnp_bqd](srd_tnp_bqd/README.md)|默认新主线、九个基线、同源 kNN、模型、风险测量及第一批参照|
|[ras_frt_uq](ras_frt_uq/README.md)|原 RAS-FRT-UQ 模型、冻结参数与原选例恢复依据|

全部比较见[完整 S01 比较](srd_tnp_bqd/comparison/README.md)。默认方法为已完成的匹配历史均值读出主线；场景、目标、原模型及已测响应不变。原事故示例保留在[第一批 GIF 目录](srd_tnp_bqd/reference/first_batch/visualizations/README.md)，不另存 PNG。

独立 Highway-env 复现结果仍在[历史归档](../archives/retired_research_chains/results/highway_replications/README.md)。MetaDrive 仿真和控制器代码已恢复保留，暂不运行；此前清理的 `results/metadrive/` 旧实验结果未恢复。
