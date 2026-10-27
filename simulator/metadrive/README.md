# MetaDrive 仿真平台

本目录提供切入、汇入和环岛场景，包含地图处理、交通参与者控制、安全评价、场景搜索与回归测试。当前论文的 HighwayEnv 独立验证使用另一套场景执行与数据流程。

| 位置 | 用途 |
|---|---|
| [scenario](scenario/)、[map](map/) | 将场景参数转换为道路、路线、车道与车辆配置 |
| [control](control/)、[safety](safety/) | 执行交通动作，计算运动与安全约束 |
| [context](context/)、[failure](failure/)、[evaluation](evaluation/) | 提取轨迹特征、判断失效并汇总指标 |
| [mining](mining/) | 利用历史测试响应和目标反馈选择后续场景 |
| [formal](formal/) | 构造反事实参考行为并校准约束判定 |
| [training](training/) | 组织场景执行和训练过程 |
| [configs](configs/)、[scripts](scripts/)、[tests](tests/) | 配置、运行脚本和回归测试 |

驾驶控制器来自 [controllers/metadrive](../../controllers/metadrive/)。场景执行需要安装 MetaDrive，并准备相应地图和配置资源。

## 常用脚本

| 脚本 | 用途 |
|---|---|
| [build_taskbook.py](scripts/build_taskbook.py) | 构建任务配置 |
| [build_cutin_casebook.py](scripts/build_cutin_casebook.py) | 生成切入候选场景 |
| [collect_source_bank.py](scripts/collect_source_bank.py) | 采集历史系统的场景响应 |
| [fit_prior.py](scripts/fit_prior.py)、[evaluate_source_loso.py](scripts/evaluate_source_loso.py) | 构建历史先验并进行来源留出检查 |
| [evaluate_cutin.py](scripts/evaluate_cutin.py) | 在统一预算接口下运行切入评价 |
| [validate_formal_teacher.py](scripts/validate_formal_teacher.py) | 构建和验证反事实参考行为 |

场景选择器见 [miner.py](mining/miner.py)，只使用目标观测的高斯过程对照见 [baselines.py](mining/baselines.py)。具体输入和输出参数可通过相应模块的 `--help` 查看。

从项目根目录运行本平台的测试：

```powershell
python -B -m pytest simulator/metadrive/tests -q -p no:cacheprovider
```
