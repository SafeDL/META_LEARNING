# MetaDrive 场景执行与搜索

本目录实现切入、汇入和环岛场景，包含地图与路线、交通控制、安全评价、历史响应建模和预算内场景搜索。其配置与评价流程独立于 HighwayEnv 主实验。

| 目录 | 职责 |
|---|---|
| [scenario/](scenario/)、[map/](map/) | 将参数转为道路、路线、车道及车辆配置 |
| [control/](control/)、[safety/](safety/) | 执行交通动作，检查运动和安全约束 |
| [context/](context/)、[failure/](failure/)、[evaluation/](evaluation/) | 提取轨迹特征、判定失效并统计指标 |
| [mining/](mining/) | 构建历史先验、适配目标响应并选择场景 |
| [formal/](formal/) | 构造反事实参考行为，校准约束判定 |
| [training/](training/) | 组织执行和训练过程 |
| [configs/](configs/)、[scripts/](scripts/)、[tests/](tests/) | 保存配置、操作入口和回归检查 |

驾驶控制器位于 [controllers/metadrive/](../../controllers/metadrive/)。执行需要安装 `metadrive-simulator==0.4.3`；场景搜索脚本默认读取 `configs/cutin.yaml`，反事实参考入口读取 `configs/formal_teacher.yaml`。

## 操作入口

| 脚本 | 用途 |
|---|---|
| [build_taskbook.py](scripts/build_taskbook.py)、[build_cutin_casebook.py](scripts/build_cutin_casebook.py) | 构建地图任务和切入候选清单 |
| [collect_source_bank.py](scripts/collect_source_bank.py)、[fit_prior.py](scripts/fit_prior.py) | 执行历史场景并拟合先验 |
| [analyze_prior.py](scripts/analyze_prior.py)、[evaluate_source_loso.py](scripts/evaluate_source_loso.py) | 分析先验结构，逐个留出来源进行核验 |
| [evaluate_cutin.py](scripts/evaluate_cutin.py) | 按指定目标、方法和预算协议运行评价 |
| [diagnose_sut.py](scripts/diagnose_sut.py)、[validate_formal_teacher.py](scripts/validate_formal_teacher.py) | 检查驾驶响应和反事实参考行为 |
| [render_stage_a.py](scripts/render_stage_a.py) | 导出已有来源记录和真实轨迹图 |

从项目根目录查看参数或运行测试：

~~~powershell
python -B -m simulator.metadrive.scripts.evaluate_cutin --help
python -B -m pytest simulator/metadrive/tests -q
~~~

场景选择见 [miner.py](mining/miner.py)，仅使用目标反馈的 GP 对照见 [baselines.py](mining/baselines.py)。输入清单、先验与输出位置由各脚本参数指定。
