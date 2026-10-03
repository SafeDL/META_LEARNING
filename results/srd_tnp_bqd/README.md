# SRD-TNP-BQD 当前结果

默认主线为匹配历史源、独立均值读出、联合均值校准、冻结 h／差异核和原混合采集；统一名称 **SRD-TNP-BQD**。[当前实验设置与结论](report.md) · [九基线比较](comparison/README.md)。

|目录|内容|
|---|---|
|comparison|当前主线和九个基线的完整轨迹、指标、真值和核验|
|history|原 A 上补充的一个速度匹配 IDM 源：2048 条真实响应、源规格与成本|
|model|原五个冻结历史模型；mean_readout 子目录为五个独立均值读出|
|training|原 A 内两阶段训练和留源选型依据|
|measurements|原 A/D 连续风险、压缩 trace 与被动测量核验|
|[reference](reference/README.md)|第一批完整结果、旧消融、旧在线复核、GIF 和未采用尝试摘要|
|audit|基准保真、历史迁移、清理与验证记录|

当前主线 F50/F100/F200 为40.2/74.2/113.6，最终召回97.93%、失效 cell 覆盖99.39%。原 A/D、目标和初始条件保持不变。本文 F50 高于九个正式基线，F100 和最终碰撞召回仍略低于 RAS-FRT-UQ。

当前复现入口为 `python -B -m methods.srd_tnp_bqd.experiment`，随后运行 `python -B -m methods.srd_tnp_bqd.evaluate`。已完成轨迹直接复用，退出主线的中间试验不再由默认入口重跑。当前方法代码与基线出处见[方法说明](../../methods/srd_tnp_bqd/README.md)，共享原始场景见[benchmarks/s01](../../benchmarks/s01/README.md)。
