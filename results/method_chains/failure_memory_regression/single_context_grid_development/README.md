# 单上下文网格开发测试

每族使用场景目录中的基准 `fixed_context`，不额外改变速度或事件时间。选取 S01–S06、S08、S09 八个可执行场景族，每族完整遍历两个参数轴的 11×11 等距网格：每族 121 个、每条 NL/PPO 版本链 968 个具体场景。每条链的 V0/V1/V2 均执行后，计划产生 2,904 条物理结果。

本银行用于先打通配对测量、双向变化真值、预算选例和报告流程。单个上下文不能检验跨速度或事件时序的泛化。已有的 `role_gated_generalization_confirmation/` 三上下文协议及部分物理结果独立保留。

在仓库根目录生成清单和协议（不运行物理仿真）：

```powershell
conda run -n metadrive python -m methods.failure_memory_regression.prepare_single_context_grid_development
```

需要执行物理测量时，分别对 `nl/` 和 `ppo/` 目录运行 `python -m methods.failure_memory_regression.bidirectional measure --root <目录>`。完整银行结束后才运行 `evaluate`。本目录不把开发结果当作独立跨上下文确认。
