# 危险用例发现效果与测试效率

本目录比较收益校正风险测试（GCRT）与均匀随机测试的发现数量、覆盖率和执行开销。先看[实验报告](report.md)、[结果表](table.md)或[主图](figures/retention_efficiency.png)。

## 运行

在项目根目录、`metadrive` 环境中重新生成图表：

```powershell
python -B -m methods.gcrt.efficiency_report
```

该命令复用已有预测和计时记录，重新计算 MCTS 补充评价的校准参数、停止位置及停止前的累计成本，不新增仿真、训练或网络预测。停止判断仅使用当前检查点及此前反馈，成本计入最后一次停止检查。

需要先生成预测及计时记录时，在模型、完整轨迹和特征缓存就绪后运行：

```powershell
python -B -m methods.gcrt.retention_efficiency
```

该入口会运行预测网络，并在完成停止回放后自动生成效率报告。模型与缓存的准备步骤见[方法说明](../../../methods/gcrt/README.md)。

## 实验配置

一组配置指一个控制器在一个功能场景、一个测试集上的实验。每个控制器使用三个测试集，种子固定为 11，最多测试 200 次，包含 10 次共同初始化。

| 范围 | 控制器与场景 | 配置数 |
|---|---|---:|
| 原预设独立评价 | IDM、IDM＋MOBIL，分别测试车辆切入和前车紧急制动 | 12 |
| 探索性补充评价 | MCTS，测试前车紧急制动 | 3 |
| 当前主图与主表 | 车辆切入的 IDM、IDM＋MOBIL，以及紧急制动的 MCTS | 9 |

MCTS 为恒速预测蒙特卡洛树搜索控制器，是查看原评价结果后增加的探索性补充评价，报告和图注须注明。网络权重保持固定；停止参数使用 PPO、IDM、IDM＋MOBIL 的全部三个制动测试集重新校准，排除被评价的 MCTS，不根据其评价结果调参。三个 MCTS 测试集全部进入主图，原 12 组结果单独完整保留。

IDM 紧急制动的三个测试集分别有 662、662、664 个碰撞用例，200 次测试最多覆盖约 30%。IDM＋MOBIL 的三个制动测试集共 3072 个用例均完成换道且无碰撞，单列适用性核查。这些双车道记录不代表受限纵向跟驰条件下的制动表现。

## 指标与成本

| 指标 | 定义 |
|---|---|
| 危险用例发现数量 | 已测用例中实测风险 `R > 0.5` 的数量 |
| 碰撞用例发现数量 | 已测用例中实测 `collision` 标签为真的数量 |
| 危险／碰撞用例覆盖率 | 已发现同类用例数 ÷ 对应测试集全部同类用例数 |
| 实际仿真调用次数 | 实际执行的场景数，包含初始化调用 |
| 执行总耗时 | 按实测记录重建的仿真执行与算法计算时间，不含离线训练、校准和维护 |

危险与碰撞结果分别报告。测试集中没有同类用例时，覆盖率未定义，不记为 0% 或 100%，也不进入该覆盖率的汇总。`included_in_collision_study` 表示测试集存在碰撞用例，`included_in_main_figure` 表示配置进入主图，两者含义不同。

## 图表与文字要求

- 使用简明中文和常用术语，明确控制器、功能场景和测试集；图例统一写“本发明方法（GCRT）”。
- 主要结论围绕发现数量、覆盖率、调用次数和耗时，只对应均匀随机测试这一对照。更换对照后重新计算差异和节省比例。
- 主图四个面板和主表使用相同的 9 组配置，两种方法的配置一致。每个柱表示三个测试集的均值，标注数值，不绘制误差棒；坐标轴、单位和图注应能独立说明图意。
- 汇总采用配置等权平均，合并计数结果另列。统计文件保留样本标准差，表示配置之间的差异；固定种子 11 的结果不作为跨种子重复实验。
- 图片仅保存为 `figures/` 下的 PNG。主图、主表不放调试线、停止阈值、校准过程或“取消收益校正”消融；相关参数、消融和其他基线记录继续保留。

## 文件索引

| 文件 | 内容 |
|---|---|
| [report.md](report.md)、[table.md](table.md)、[disclosure_text.md](disclosure_text.md) | 实验报告、结果表和交底书填写内容 |
| [figures/retention_efficiency.png](figures/retention_efficiency.png) | 主图 E1 |
| [comparison_metrics.csv](comparison_metrics.csv) | 原 12 组及补充 3 组配置的结果，含评价来源与主图标记 |
| [comparison_summary.csv](comparison_summary.csv) | 主图 9 组配置的汇总；同名 JSON 保存完整数值 |
| [original_evaluation_table.md](original_evaluation_table.md)、`original_evaluation_metrics.*`、`original_evaluation_summary.*` | 原预设独立评价的完整结果 |
| [supplementary_protocol.json](supplementary_protocol.json)、[supplementary_stopping.json](supplementary_stopping.json) | MCTS 补充评价的数据划分、校准参数、权重摘要、停止位置与成本 |
| [scenario_audit.json](scenario_audit.json) | 原评价配置的碰撞数、车道数、换道情况与测量记录路径 |
| `config.json`、`calibration.json`、`replays.json`、`task_metrics.*`、`summary.*` | 停止回放的设置、校准记录和原始统计，当前主图使用 `comparison_*` |
| [validation.json](validation.json) | 数据划分、停止位置和成本汇总的核查结果 |

分布校正前后的预测误差与停止表现见[校正有效性实验](../calibration_effectiveness/README.md)。
