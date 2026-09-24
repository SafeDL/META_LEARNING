# META_LEARNING

面向黑盒驾驶控制器的可迁移少样本脆弱场景挖掘研究代码库。项目按仿真器
隔离实现：MetaDrive 用于完整的地图感知场景实验，Highway-env 用于快速的
Risk Mining MVP 机制验证；两者的代码、测试和结果互不覆盖。

| 目录 | 用途 |
| --- | --- |
| `metadrive_benchmark/` | 本文 Risk Mining / Formal Teacher 的 MetaDrive 实现 |
| `highway_env_benchmark/` | 本文 Mining 的 highway-env 实现与共享仿真底座 |
| `sut_algorithms/` | 两套仿真器共用的被测驾驶算法与控制器 registry |
| `archives/` | 冻结的 PEARL 与 SAC 历史基线 |
| `docs/` | 方法、实验设计与执行说明 |
| `results/metadrive/` | MetaDrive 的可追溯实验工件 |
| `replications/` | AdaTE、DETOUR、FST、ScenarioFuzz 的独立 highway-env 复现与统一评测 |
| `results/highway_replications/` | 共享响应库、各方法唯一正式结果与跨方法评价 |
| `results/method_chains/` | 本文组合方法的唯一正式结果根目录 |
| `method_chains/detour_fusion/` | 本文 Risk Mining 与 DETOUR 的独立融合链及专属结果 |
| `method_chains/function_conditioned_routing/` | 本文功能条件化历史迁移方法、双基准与危险场景回放 |
| `method_chains/function_posterior_search/` | 经独立物理确认的自适应功能后验搜索 |
| `method_chains/core_mine/` | 组合残差脆弱区域挖掘及其冻结负结果 |

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

Highway-env 驾驶算法的接入、筛选和风险差异审计位于
`replications/highway_sut_selection/`，正式结果位于
`results/highway_replications/sut_selection/`。

融合方法的目录所有权和修改边界见
[`method_chains/detour_fusion/README.md`](method_chains/detour_fusion/README.md)。

本文功能条件化方法及五类危险场景 GIF 见
[`method_chains/function_conditioned_routing/README.md`](method_chains/function_conditioned_routing/README.md)。

独立确认的功能后验搜索见
[`method_chains/function_posterior_search/README.md`](method_chains/function_posterior_search/README.md)。

CoRe-Mine 的冻结协议、运行入口和负结果边界见
[`method_chains/core_mine/README.md`](method_chains/core_mine/README.md)。
