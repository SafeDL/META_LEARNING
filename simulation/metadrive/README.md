# MetaDrive 仿真平台

本目录是工程内保留的独立仿真平台，包含切入、汇入和环岛场景，以及地图处理、车辆控制、安全评价和场景搜索。当前 HighwayEnv 碰撞发现主实验的结果不代表 MetaDrive 的测试结果。

| 目录 | 作用 |
|---|---|
| [scenario](scenario/)、[map](map/) | 把场景参数放到道路上，处理路线、车道和车辆位置 |
| [control](control/)、[safety](safety/) | 执行交通参与者动作，检查车辆运动和安全约束 |
| [context](context/)、[failure](failure/)、[evaluation](evaluation/) | 从轨迹提取特征，判断失效并汇总评价 |
| [mining](mining/) | 学习已有测试结果，选择或搜索后续场景 |
| [formal](formal/) | 构造反事实参考行为，并校准相应的约束判定 |
| [training](training/) | 组织训练过程 |
| [configs](configs/)、[scripts](scripts/)、[tests](tests/) | 场景配置、构建与运行脚本、回归测试 |

驾驶控制器位于 [controllers/metadrive](../../controllers/metadrive/)，包括 IDM 控制器和六种驾驶配置。脚本入口集中在 `scripts/`，例如 `build_taskbook.py` 构建任务配置，`collect_source_bank.py` 采集历史测试记录，`evaluate_cutin.py` 运行切入场景评价。运行场景仿真需要可用的 MetaDrive 环境与地图资源。

从项目根目录运行本平台的回归测试：

```powershell
conda activate metadrive
python -B -m pytest simulation/metadrive/tests -q -p no:cacheprovider
```

MetaDrive 的代码、配置、脚本和有效测试均保留在本目录；清理缓存和其他旧实验文件时保留该平台。
