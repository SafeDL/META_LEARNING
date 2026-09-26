# META_LEARNING

面向黑盒驾驶控制器测试的研究代码库。当前方法主线是基于历史失效边界的
功能场景回归测试（FBRT），运行在 Highway-env 上；MetaDrive 的 Risk Mining /
Formal Teacher 实验、其他方法链和论文复现作为独立研究记录保留，不混入 FBRT 结果。

| 目录 | 用途 |
| --- | --- |
| `metadrive_benchmark/` | MetaDrive 仿真底座与 Risk Mining / Formal Teacher 历史实现 |
| `highway_env_benchmark/` | Highway-env 共享仿真底座及 FBRT 功能场景实现 |
| `sut_algorithms/` | 两套仿真器共用的被测驾驶算法与控制器 registry |
| `archives/` | 冻结的 PEARL 与 SAC 历史基线 |
| `docs/` | 方法、实验设计与执行说明 |
| `results/metadrive/` | MetaDrive Risk Mining / Formal Teacher 实验工件 |
| `replications/` | AdaTE、DETOUR、FST、ScenarioFuzz 的独立 highway-env 复现与统一评测 |
| `results/highway_replications/` | 共享响应库、各方法唯一正式结果与跨方法评价 |
| `results/method_chains/` | 组合方法的正式结果根目录，按方法分开保存 |
| `method_chains/detour_fusion/` | Risk Mining 与 DETOUR 的独立融合研究链 |
| `method_chains/function_conditioned_routing/` | 功能条件化历史迁移方法及危险场景回放 |
| `method_chains/function_posterior_search/` | 经独立物理确认的自适应功能后验搜索 |
| `method_chains/failure_memory_regression/` | 当前 FBRT 回归测试主链：历史失效边界、预算选例与结果分析 |
| `method_chains/core_mine/` | CoRe-Mine 历史实验；FBRT 复用其中的 IDM 基线和受控修改实现 |

## 复现与验证

使用根目录的 `environment.yml` 创建 `metadrive` Conda 环境后，在仓库根目录运行：

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
conda run -n metadrive python -m pytest -q -p no:cacheprovider
```

重新构建 Highway-env 的基础候选库与响应库：

```powershell
conda run -n metadrive python -m highway_env_benchmark.data.generate_anchor_bank
conda run -n metadrive python -m highway_env_benchmark.data.response_bank
```

两套实现的维护边界分别见 [`metadrive_benchmark/README.md`](metadrive_benchmark/README.md)
和 [`highway_env_benchmark/README.md`](highway_env_benchmark/README.md)。

四套代表性工作、共享数据契约、同预算评测口径和完整运行命令见
[`replications/README.md`](replications/README.md)。各方法的论文对齐范围和偏差
分别记录在其包内 README 与 `results/highway_replications/` 的最终报告中。

方法链的当前主线、历史分支和依赖关系见
[`method_chains/README.md`](method_chains/README.md)。

Highway-env 驾驶算法的接入、筛选和风险差异审计位于
`replications/highway_sut_selection/`，正式结果位于
`results/highway_replications/sut_selection/`。

其他方法链的范围和依赖边界见
[`method_chains/detour_fusion/README.md`](method_chains/detour_fusion/README.md)。

功能条件化历史迁移方法及五类危险场景 GIF 见
[`method_chains/function_conditioned_routing/README.md`](method_chains/function_conditioned_routing/README.md)。

独立确认的功能后验搜索见
[`method_chains/function_posterior_search/README.md`](method_chains/function_posterior_search/README.md)。

当前 FBRT 的代码修复与零新增仿真回放方案见
[`docs/FBRT_CODE_FIXES_ZERO_SIM_CODEX_PLAN.md`](docs/FBRT_CODE_FIXES_ZERO_SIM_CODEX_PLAN.md)，最近一次回放结果见
[`修复回放报告`](results/method_chains/failure_memory_regression/repair_20260926/repair_report.md)。
此前的标准对齐实验报告仍保留为基准结果：
[`standard_aligned/core/report.md`](results/method_chains/failure_memory_regression/standard_aligned/core/report.md)。
重放现有缓存且不新增仿真的命令为
`conda run -n metadrive python -m method_chains.failure_memory_regression.repair_replay --offline-only --output results/method_chains/failure_memory_regression/repair_20260926`。
方法包的文件职责、实验范围和入口见
[`method_chains/failure_memory_regression/README.md`](method_chains/failure_memory_regression/README.md)。

CoRe-Mine 的历史实验及其保留原因见
[`method_chains/core_mine/README.md`](method_chains/core_mine/README.md)。
