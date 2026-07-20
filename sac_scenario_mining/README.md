# MetaDrive on-ramp merge 的 SAC 对抗场景生成

本模块在 MetaDrive 的固定 `SrS` 地图模板上实现 `on_ramp_merge` 逻辑场景中的 SAC 对抗场景生成。它用于验证“学习到的匝道对抗车辆策略能否更有效地产生有效目标碰撞”，不是 AEB 测试，也不代表对任意真实 ADS 的安全结论。

## 问题定义

`SrS` 的地图拓扑和车道几何由 MetaDrive 模板提供；实验代码不从一组现成的具体场景中抽取样本，而是固定模板后，按配置随机采样并显式实例化逻辑参数。每个 case 由 `case_id + theta + background_seed` 定义：

- `sut_speed_mps`：主线 SUT 初速度；
- `adversary_speed_mps`：匝道对抗车初速度；
- `longitudinal_gap_m`：两车初始纵向间隔；
- `adversary_ramp_position_m`：对抗车在入口匝道上的起始位置；
- `background_density`：显式生成相邻主线背景车的概率参数。

地图引擎 seed 固定为 `0`，以保持 `SrS` 的 `I-S-r-S` 几何不变。`background_seed` 只决定显式背景车的位置、速度和 IDM 随机性。因而，seed 改变的不只是地图拓扑：本项目把地图固定，并把交通参与者的初始条件和背景随机性写入 case，保证其可检查、可复现。

角色固定如下：

- SUT：主线汇入前车道上的 `TrafficDefaultVehicle`，使用 IDM 基准控制器；
- adversary：入口匝道上的 MetaDrive 默认车；SAC 在每个仿真步输出连续动作 `[steering, throttle/brake]`；
- background：可选的相邻主线 IDM 车，不改变 SUT/adversary 的语义角色。

SAC 不改变 `theta`，不控制 SUT，也不直接把“碰撞”作为动作；它只通过连续驾驶动作影响汇入时序和相对运动。

## 训练、验证与 held-out 测试

配置文件为 [configs/merge_sac.yaml](configs/merge_sac.yaml)。训练、验证、测试分别使用由 case seed `101`、`202`、`303` 确定生成的 96、24、60 个 case。三者是互不重叠的 case 集，但共享同一 `on_ramp_merge` 逻辑任务族、地图模板、参数空间和角色定义。

`seed 0/1/2` 是三次独立 SAC 优化的随机种子，影响网络初始化、经验采样与优化过程；它们不是三种地图或三类逻辑场景。每个 seed 训练 300,000 个环境步，并以 24 个 validation case 上的目标碰撞率、有效危险率、无效率和最小 TTC 选择最佳 checkpoint。Random 与三组 SAC 均在同一张 60-case held-out 表上评估，才可以公平比较。

观测为固定 38 维的 `on_ramp_merge_obs_v2`，包含对抗车状态、SUT 相对状态、至多三辆邻车的相对状态以及道路特征。其维度和顺序是已训练模型的接口，不能随意删改。

奖励由低 TTC、近距离、目标碰撞奖励构成，并惩罚非目标碰撞、出界/逆行和过激或不平滑动作。目标碰撞会终止 episode；非目标碰撞、任一角色出界、对象碰撞及 horizon 也会结束 episode。

## 指标含义

- `target_collision`：adversary 与固定 SUT 的碰撞；
- `critical`：发生目标碰撞，或 `min_ttc <= 1.5 s`；
- `invalid_before_critical`：达到危险事件前发生非目标碰撞、任一角色出界或对抗车逆行；
- `valid_critical`：`critical` 且未在此前无效，比例即“有效场景率”；
- `invalid_rate`：出现无效事件的 case 比例；
- `target_collision_rate`：目标碰撞 case 比例；`median_min_ttc` 越小表示最接近风险更高，但不能单独代替有效性判断。

## 已归档正式结果

所有策略在相同 60 个 held-out case 上运行：

| 策略 | 有效场景率 | 目标碰撞率 | 无效率 | 中位最小 TTC |
| --- | ---: | ---: | ---: | ---: |
| Random | 6.7% | 3.3% | 65.0% | 3.278 s |
| SAC seed 0 | 96.7% | 91.7% | 3.3% | 1.035 s |
| SAC seed 1 | 98.3% | 100.0% | 1.7% | 0.929 s |
| SAC seed 2 | 100.0% | 100.0% | 0.0% | 0.861 s |

自动验收为 3/3 seed 通过。seed 2 导出的 top-10 关键场景可由 `manifest.json + actions.npy` 回放，10/10 的目标碰撞与危险判定一致，最大 TTC 误差约 `8.4e-05 s`，小于 `0.1 s` 容差。

这些结果支持：在这个固定 MetaDrive `SrS`、IDM SUT 的 `on_ramp_merge` 任务族内，三次 SAC 训练都显著高于随机连续动作基线。它们不支持对其他地图、真实道路、其他交通分布或特定 ADS 的泛化主张。

> 历史归档 case 中仍可见 `adversary_target_speed_mps`。审计确认该字段从未参与环境初始化或 SAC 控制，当前配置和新生成 case 已移除它。回放加载旧 manifest 时会忽略该历史字段，因此既不改变已有实验行为，也不影响结果复现。

## 目录与入口

```text
configs/merge_sac.yaml        场景参数、case 集与 SAC 超参数
src/casebook.py               确定性 case 生成和边界校验
src/env.py                    固定角色的 Gymnasium 环境
src/metadrive_compat.py       MetaDrive 0.4.3 的地图、车辆和 IDM 适配
src/{observation,reward,metrics,scenario_manifest}.py
scripts/train_sac.py          训练与 checkpoint validation
scripts/evaluate.py           Random/SAC 的固定 case 评估和关键场景导出
scripts/replay.py             已导出动作轨迹回放
scripts/visualize.py          加载策略的交互可视化
scripts/{report,audit_results,audit_replays}.py
```

在仓库根目录执行：

```powershell
conda activate metadrive
python -m sac_scenario_mining.scripts.train_sac --seed 0 --run-name merge_sac_seed0
python -m sac_scenario_mining.scripts.evaluate --policy random --split test --seed 123
# 默认从固定 held-out case 表抽取 5 个互不重复的场景
python -m sac_scenario_mining.scripts.visualize --selection-seed 42
# 如需锁定单个场景：
python -m sac_scenario_mining.scripts.visualize --case-id test_000
python -m sac_scenario_mining.scripts.replay --manifest results/sac_scenario_mining/final_eval/sac_seed2/critical_scenarios/rank_001/manifest.json
```

也可以在 PyCharm 中直接运行 `scripts/visualize.py`。默认行为是依据 `--selection-seed 0` 从固定 held-out test case 表中**无放回抽取 5 个不同 case 并顺序展示**；同一选择种子总会得到同一组 case。它不重新生成地图、交通流或逻辑场景参数，因而这五个展示仍可对应正式测试表。`--num-cases` 可调整数量，`--case-id` 则锁定一个具体 case。脚本会打开**一个并排窗口**：左侧为跟随主车（IDM SUT）的第三人称追踪画面，右侧为全局鸟瞰图，并叠加 case、最小 TTC、车间距与 SAC 动作。两图均以蓝色表示 SUT、以红色表示 SAC 对抗车；两者的物理车型、控制器和碰撞几何仍与训练/测试完全一致，颜色仅用于角色辨识。按 `Q` 或 `Esc` 可提前关闭。不需要手动启动 `F:\PyCharm 2024.3.2\work\metadrive`：它是 Python 包源码目录，不是独立服务。
