# FBRT 回归测试

这是当前方法主链：用旧版本的失败场景及其附近的通过场景估计局部失效边界，再从共同候选集中优先挑选边界附近的场景，测试新版本是否出现回归。每次查询只揭示一个已测目标版本的结果。

## 实验范围

- 仿真器：`highway-env`；参考 SUT 是 IDM，目标是三个受控 IDM 修改版本。
- 功能场景：前车切入、前车切出后出现静止车辆、前车紧急制动、停车—保持—起步。
- 数据：3 个种子；480 次参考执行、960 次目标版本执行；共同候选池 320 个通过场景。
- 选择预算：每个种子与目标版本最多测试 50 个不同场景；报告预算点为 @5、@10、@20、@50。
- 比较方法：Random、ART-Maximin、HistoryMargin、FailureDistance、HistoryRank-UCB，以及 FBRT-Static、FBRT-Adaptive、FBRT-RegionBandit。

当前结果是同一开发结果库上的离线比较，不是独立确认。报告显示，@5 时多个简单历史排序基线与 FBRT 都能较快发现回归；FBRT 原型已跑通，但结果尚不能说明 FBRT 普遍优于基线。

## 入口与结果

`boundary_memory.py` 从参考执行构建失败—通过局部配对；`selectors.py` 实现预算内的场景选择；`experiment.py` 运行或重放选择实验；`report.py` 校验结果并生成图表和回放。场景与执行器位于 `highway_env_benchmark/envs/fbrt_scenarios.py`、`fbrt_env.py` 和 `fbrt_scripted_vehicle.py`。IDM 参考配置和局部修改复用自 [`../core_mine/README.md`](../core_mine/README.md) 所列实现。

正式结果位于 [`results/method_chains/failure_memory_regression/standard_aligned/core/`](../../results/method_chains/failure_memory_regression/standard_aligned/core/)，方法设计与详细实验设置见 [`docs/FBRT_STANDARD_ALIGNED_CODEX_PLAN.md`](../../docs/FBRT_STANDARD_ALIGNED_CODEX_PLAN.md)。

离线重放已保存的实测库（不新增仿真）：

```powershell
conda run -n metadrive python -m method_chains.failure_memory_regression.experiment --replay-measured-bank
conda run -n metadrive python -m method_chains.failure_memory_regression.report --reuse-replay
```

不带 `--replay-measured-bank` 运行 `experiment` 会执行缺失的物理 episode，并写入同一正式结果目录。
