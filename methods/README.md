# 方法

[SRD-TNP-BQD](srd_tnp_bqd/README.md) 是唯一论文主链，默认使用匹配历史源、独立均值读出、联合均值校准、冻结 h／差异核和原混合采集。执行顺序为 `audit → train → experiment → evaluate`，原 S01 A/D 各 2048 个场景、五个 IDM 源及固定 FVDM 目标不变；原 A 上已补充一个速度匹配的 IDM 源。

同一实验入口运行八个通用/文献基线及 [RAS-FRT-UQ](ras_frt_uq/README.md) 原方法，并单列同源 kNN 数据对照。RAS 使用原模型和二值反馈。第一批及旧消融仅作为历史参照保存。

[共享基准](../benchmarks/s01/README.md) · [全部比较](../results/srd_tnp_bqd/comparison/README.md) · [独立历史归档](../archives/retired_research_chains/README.md)
