# META_LEARNING

面向黑盒驾驶控制器的可迁移少样本脆弱场景挖掘研究代码库。项目按仿真器
隔离实现：MetaDrive 用于完整的地图感知场景实验，Highway-env 用于快速的
DIVA-Mine MVP 机制验证；两者的代码、测试和结果互不覆盖。

| 目录 | 用途 |
| --- | --- |
| `diva_metadrive/` | 本文 DIVA-Mine / DIVA-Former 的 MetaDrive 实现 |
| `diva_highway_env/` | 本文 DIVA 的 highway-env 实现与共享仿真底座 |
| `archives/` | 冻结的 PEARL 与 SAC 历史基线 |
| `docs/` | 方法、实验设计与执行说明 |
| `results/metadrive/` | MetaDrive 的可追溯实验工件 |
| `replications/` | AdaTE、DETOUR、FST、ScenarioFuzz 的独立 highway-env 复现与统一评测 |
| `results/highway_replications/` | 共享响应库、各方法唯一正式结果与跨方法评价 |
| `results/method_chains/` | 本文组合方法的唯一正式结果根目录 |
| `method_chains/diva_detour_fusion/` | 本文 DIVA-Mine 与 DETOUR 的独立融合链及专属结果 |
| `method_chains/diva_function_conditioned_routing/` | 本文功能条件化历史迁移方法、双基准与危险场景回放 |
| `method_chains/function_posterior_search/` | 经独立物理确认的自适应功能后验搜索 |

## 复现与验证

使用根目录的 `environment.yml` 创建 `metadrive` Conda 环境后，在仓库根目录运行：

```powershell
conda run -n metadrive python -m pytest diva_metadrive/tests diva_highway_env/tests -q
conda run -n metadrive python -m pytest archives/pearl_learning/tests -q
conda run -n metadrive python -m compileall -q diva_metadrive diva_highway_env archives/pearl_learning archives/sac_scenario_mining
```

重新构建 Highway-env 的基础候选库与响应库：

```powershell
conda run -n metadrive python -m diva_highway_env.data.generate_anchor_bank
conda run -n metadrive python -m diva_highway_env.data.response_bank
```

两套实现的维护边界分别见 [`diva_metadrive/README.md`](diva_metadrive/README.md)
和 [`diva_highway_env/README.md`](diva_highway_env/README.md)。

四套代表性工作、共享数据契约、同预算评测口径和完整运行命令见
[`replications/README.md`](replications/README.md)。各方法的论文对齐范围和偏差
分别记录在其包内 README 与 `results/highway_replications/` 的最终报告中。

融合方法的目录所有权和修改边界见
[`method_chains/diva_detour_fusion/README.md`](method_chains/diva_detour_fusion/README.md)。

本文功能条件化方法及五类危险场景 GIF 见
[`method_chains/diva_function_conditioned_routing/README.md`](method_chains/diva_function_conditioned_routing/README.md)。

独立确认的功能后验搜索见
[`method_chains/function_posterior_search/README.md`](method_chains/function_posterior_search/README.md)。
