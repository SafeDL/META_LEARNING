# Highway-env 论文复现

本目录保存代表性测试方法的独立复现。各方法共用 `highway_sim_env/` 的仿真环境、`sut_algorithms/` 的被测控制器及统一响应数据契约；方法实现留在各自目录，正式结果统一写入 `results/highway_replications/`。

| 路径 | 职责 |
| --- | --- |
| `benchmark.py`、`benchmark.yaml` | 构建共享候选与响应库，规定跨方法评价的数据和预算 |
| `adate_highway_env/` | AdaTE 方法复现 |
| `detour_highway_env/` | DETOUR 方法复现 |
| `fst_highway_env/` | FST 相似度方法复现 |
| `scenariofuzz_highway_env/` | ScenarioFuzz 方法复现 |
| `highway_sut_selection/` | 异构驾驶策略筛选、PPO-ECE 权重获取与风险结构审计 |
| `tests/` | 共享协议和复现契约测试 |

共享基准包含三种交互模式，每种模式 64 个 Sobol 候选，共 192 个场景；六种 SUT 产生 1,152 条真实仿真响应。失效发现使用预算 5/10/20 下的碰撞精确率与召回率，性能估计使用相同预算下的碰撞率误差。论文特有指标保留在各方法报告中，不将不同任务混为同一排名。

复现时保持原论文的信息边界、更新规则、基线和消融；Highway-env 无法支持的部分在方法报告中标明偏差。共享结果入口见 [`results/highway_replications/README.md`](../results/highway_replications/README.md)，各方法目录的 README 给出具体重建命令。

```powershell
conda run -n metadrive python -m replications.benchmark build
conda run -n metadrive python -m replications.benchmark evaluate
conda run -n metadrive python -m pytest replications -q -p no:cacheprovider
```
