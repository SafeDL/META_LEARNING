# META_LEARNING

面向黑盒驾驶控制器测试的研究代码库。当前论文主线是 RAS-FRT：使用历史库 A 中五组
同模型不同参数的 IDM 响应，在 Highway-env 的新场景库 D 上寻找固定 FVDM 的碰撞场景。
九种选例方法共享 D 的候选场景和 200 次查询预算。

| 目录 | 用途 |
| --- | --- |
| `metadrive_sim_env/` | MetaDrive 仿真底座与 Risk Mining / Formal Teacher 历史实现 |
| [`highway_sim_env/`](highway_sim_env/README.md) | Highway-env 共享仿真底座、构建标识与 S01 场景参数规则 |
| [`sut_algorithms/`](sut_algorithms/README.md) | 两套仿真器共用的被测驾驶算法与控制器 registry |
| [`archives/`](archives/README.md) | 历史基线、退出活动目录的研究链及其结果 |
| `docs/` | 方法、实验设计与执行说明 |
| `results/metadrive/` | MetaDrive Risk Mining / Formal Teacher 实验工件 |
| `replications/` | AdaTE、DETOUR、FST、ScenarioFuzz 的独立 highway-env 复现与统一评测 |
| `results/highway_replications/` | 共享响应库、各方法唯一正式结果与跨方法评价 |
| [`results/method_chains/`](results/method_chains/README.md) | 当前 RAS-FRT 的 A、D 数据和九方法结果 |
| [`methods/ras_frt/`](methods/ras_frt/README.md) | 当前 RAS-FRT 方法、九种选择器及 A→D 实验入口 |

## 当前实验与验证

使用根目录的 `environment.yml` 创建 `metadrive` Conda 环境后，在仓库根目录重放 D 上的九方法比较。此命令读取已有的 A、D 数据；D 的 2,048 次物理执行已提前完成，200 次是每种方法可见的目标标签预算。

```powershell
conda run -n metadrive python -m methods.ras_frt.d_budget_200_experiment
```

RAS-FRT 的模块职责与结果索引见[方法说明](methods/ras_frt/README.md)和[A、D 数据说明](results/method_chains/ras_frt/README.md)。

运行活动代码测试：

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
conda run -n metadrive python -m pytest -q -p no:cacheprovider
```

默认测试由 `pytest.ini` 排除归档目录；归档测试可显式指定路径单独运行。

## 独立复现链

以下命令重新构建 Highway-env 论文复现链的基础候选库与响应库；它们不是 RAS-FRT 的 A、D 数据生成入口。

```powershell
conda run -n metadrive python -m highway_sim_env.data.generate_anchor_bank
conda run -n metadrive python -m highway_sim_env.data.response_bank
```

两套实现的维护边界分别见 [`metadrive_sim_env/README.md`](metadrive_sim_env/README.md)
和 [`highway_sim_env/README.md`](highway_sim_env/README.md)。

四套代表性工作、共享数据契约、同预算评测口径和完整运行命令见
[`replications/README.md`](replications/README.md)。各方法的论文对齐范围和偏差
分别记录在其包内 README 与 `results/highway_replications/` 的最终报告中。

Highway-env 驾驶算法的接入、筛选和风险差异审计位于
`replications/highway_sut_selection/`，正式结果位于
`results/highway_replications/sut_selection/`。

旧方法实现与结果的归档位置和原路径映射见
[`archives/retired_research_chains/README.md`](archives/retired_research_chains/README.md)。
