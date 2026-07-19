# SAC 对抗场景生成：正式实验结果

本目录只保留当前实现的正式训练与 held-out 测试工件。已通过的一次性 MetaDrive 集成冒烟测试没有保留结果，也不再保留相应脚本；请勿将其他历史目录或日志与这里的正式结论混用。

## 实验设置

任务采用 MetaDrive 内置 `SrS` merge 地图模板。当前 `SrS` 的道路块序列固定为 `I-S-r-S`，场景 seed 不改变道路拓扑。项目传入场景 seed、交通密度等配置，由 MetaDrive 的内部程序化生成器确定固定道路上的车辆位置、车型和 IDM 策略随机初相等；不是录制场景，也不是项目手写的随机场景生成器。

| 场景划分 | 场景 seed | 用途 |
| --- | --- | --- |
| train | 0–39 | SAC 可交互、可写入 replay buffer 的 40 个 merge 实例 |
| validation | 1000–1009 | 每 5,000 训练步选取最佳 checkpoint，不用于最终结论 |
| held-out test | 2000–2019 | 训练和选模均未使用的 20 个 merge 实例 |

这些场景共享同一个固定 merge 模板和任务定义，但因场景 seed 不同而具有不同的初始交通与交互实例。因此 held-out test 测量的是同一任务族中对未见实例的泛化，而不是跨道路模板泛化。SAC 训练 seed 0、1、2 则是三次独立的训练随机过程，不能与上表的场景 seed 混为一谈。

## 工件来源

| 目录 | 产生入口 | 内容与用途 |
| --- | --- | --- |
| `merge_sac_seed0/` | `train_sac --seed 0` | 100k steps 的模型、配置、训练回报与 validation 记录 |
| `merge_sac_seed1/` | `train_sac --seed 1` | 第二次独立训练，用于检查训练稳定性 |
| `merge_sac_seed2/` | `train_sac --seed 2` | 第三次独立训练；其最佳模型给出当前最强测试结果 |
| `final_eval/random/` | `evaluate --policy random` | 随机动作基线的 100 个 test episode |
| `final_eval/sac_seed0/` 至 `sac_seed2/` | `evaluate --policy-path .../best_model.zip` | 三个最佳 SAC 模型各自的 100 个 held-out test episode |
| `final_eval/comparison.csv`、`plots/` | `report` | 汇总比较表与图表 |
| `final_eval/acceptance_audit.json` | `audit_results` | 依据验收阈值的自动判定 |
| `final_eval/sac_seed2/replay_audit.json` | `audit_replays` | seed 2 top-10 关键场景的回放一致性审计 |

`best_model.zip` 是 validation 指标最优的 checkpoint；`final_model.zip` 只是 100k steps 的最后一个 checkpoint，不一定是正式评估模型。模型二进制不纳入 Git；CSV、JSON、图表、manifest 和动作轨迹保留以支持复核。

## 正式结果与结论

| 策略 | 有效关键场景率 | 目标碰撞率 | 无效率 | 验收 |
| --- | ---: | ---: | ---: | --- |
| Random | 3% | 3% | 1% | 基线 |
| SAC（训练 seed 0） | 52% | 44% | 4% | 通过 |
| SAC（训练 seed 1） | 71% | 58% | 29% | 未通过无效率阈值 |
| SAC（训练 seed 2） | 80% | 77% | 20% | 通过，最佳 |

验收阈值要求性能提升且 `invalid_rate <= 25%`。seed 0 与 seed 2 通过，因此三个独立 SAC 训练运行中有两个通过。seed 2 的 top-10 关键场景回放审计为 10/10 通过，TTC 误差均在 0.1 s 容差内。

该实验支持 SAC 相对于随机动作基线显著提高了当前 `SrS` merge 任务族中的目标碰撞率与有效关键场景率。所有策略共享同一个 held-out 场景池，但既有评估没有固定完全相同的 episode 抽样顺序；若要报告严格的逐 episode 配对统计，应固定同一 episode 场景 seed 序列后重新评估。

## 复核与可视化

在仓库根目录、激活 `metadrive` 环境后运行：

```powershell
python -m sac_scenario_mining.scripts.report
python -m sac_scenario_mining.scripts.audit_results
python -m sac_scenario_mining.scripts.audit_replays --scenarios-root results/sac_scenario_mining/final_eval/sac_seed2/critical_scenarios
python -m sac_scenario_mining.scripts.visualize
```

`visualize` 默认加载 `merge_sac_seed2/best_model.zip` 并播放 held-out 场景 seed 2016；`replay.py --manifest <manifest.json> --render topdown` 可精确重放保存的动作轨迹。实现、指标定义和训练入口见 [`sac_scenario_mining/README.md`](../../sac_scenario_mining/README.md)。
