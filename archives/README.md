# 归档与恢复

当前论文方法和实验结果分别位于[历史引导风险测试](../methods/history_guided_testing/README.md)与[主链结果](../results/history_guided_testing/README.md)。活动目录仅保留当前两类场景、九个基线及有效组件对照。

|内容|位置|
|---|---|
|旧试验结果、原 S01/RAS 数据与旧入口|`retired_experiments.zip`，可恢复归档|
|AdaTE、DETOUR、FST、ScenarioFuzz 独立论文复现与结果|[retired_research_chains](retired_research_chains/README.md)|
|PEARL 合流历史基线|`pearl_learning/`|
|早期 SAC 场景挖掘基线|`sac_scenario_mining/`|

旧归档共 19,253 个文件，打包后逐文件解压并与原文件进行字节比对，全部一致。展开目录已清理，原相对路径保存在压缩包中。清理范围与核验记录见 [cleanup_review.json](../results/history_guided_testing/cleanup_review.json)。压缩包仅在本地保留，不随常规 Git 提交。

需要核对旧实验时，在仓库根目录解压到独立目录：

```powershell
Expand-Archive -LiteralPath archives/retired_experiments.zip -DestinationPath archives/restored_experiments
```

该命令恢复旧文件的内容和原相对目录结构，不会修改当前主链。当前历史响应、目标响应、GP 与 RAS 模型、80 条统一比较轨迹、关键失败结论和驾驶算法平台继续保留。
