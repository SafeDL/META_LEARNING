# SAC 驱动的 MetaDrive 对抗场景生成

本模块在 MetaDrive 的 merge 场景中训练 SAC（Soft Actor-Critic）策略。SAC 控制对抗车辆（adversary），目标是在不依赖出界、逆行或碰撞无关车辆等无效行为的前提下，为被测车辆（SUT）生成高风险交互场景。

## 场景、角色与指标

MetaDrive 提供内置 `SrS` merge 地图模板。本项目向它传入地图模板、场景 seed 和交通参数；并非录制数据集或本项目手写的随机道路/车辆生成器。当前 `SrS` 的道路块序列固定为 `I-S-r-S`，**不会随场景 seed 改变**。场景 seed 由 MetaDrive 的内部生成器用于确定该固定道路上的初始交通实例（车辆位置、车型、IDM 策略的随机初相等），并可能影响默认车的初始生成位置。`random_traffic=False` 使相同配置和相同场景 seed 的实例可复现。

- adversary：MetaDrive 默认受控车，由 SAC 输出连续动作 `[steering, throttle/brake]`。
- SUT：每次 reset 时离 adversary 最近的 IDM 交通车；一个 episode 内保持不变。
- `target_collision`：adversary 与该固定 SUT 的碰撞，不包括与其他交通车的碰撞。

```text
critical       = target_collision OR min_ttc <= 1.5 s
valid          = critical 发生前未出现非目标碰撞、adversary 出界或逆行
valid_critical = critical AND valid
invalid_rate   = 1 - valid_rate
```

奖励以接近 SUT 和降低 TTC 为稠密部分，以目标碰撞为奖励；非目标碰撞、出界、逆行、过大动作及动作突变会被惩罚。训练时每 5,000 步在 validation split 上选择 checkpoint，优先比较 `valid_critical_rate`，再比较更低的中位 `min_ttc`。

## seed 的两层含义

这里有两类互不相同的随机数种子：

| 名称 | 取值 | 控制对象 |
| --- | --- | --- |
| 场景 seed | train: 0–39；validation: 1000–1009；test: 2000–2019 | MetaDrive 在固定 `SrS` 道路上的具体初始交通/交互实例；不改变道路拓扑 |
| SAC 训练 seed | 0、1、2 | 网络初始化、采样和训练过程的随机性；分别得到三个独立策略 |

“互不重叠的 seed 集合，但属于同一 merge 任务族”仅指第一行的**场景 seed**：训练从 40 个 merge 初始交通实例学习；验证和测试使用未在训练或选模阶段出现的实例。它们共享固定的 `SrS` merge 道路拓扑、车辆类型、交通规则、观测、动作和奖励定义，因此是在同一任务分布内检验对未见初始交通实例的泛化；它不证明对全新道路拓扑或不同驾驶任务的泛化。

## 当前正式结果

每个策略在 held-out test split 上运行 100 个 episode。Random 与三个 SAC 策略共享 test 场景池，但当前保存的评估并未把每一个 episode 的抽样顺序严格配对；下表支持总体比较，不应据此作严格的逐 episode 显著性检验。

| 策略 | 有效关键场景率 | 目标碰撞率 | 无效率 | 结论 |
| --- | ---: | ---: | ---: | --- |
| Random | 3% | 3% | 1% | 随机基线 |
| SAC（训练 seed 0） | 52% | 44% | 4% | 通过验收 |
| SAC（训练 seed 1） | 71% | 58% | 29% | 风险提升明显，但无效率超过 25% |
| SAC（训练 seed 2） | 80% | 77% | 20% | 通过验收，当前最佳 |

SAC seed 0 和 2 同时满足性能提升与 `invalid_rate <= 25%`；因此 3 个独立训练运行中有 2 个满足验收条件。结果支持“该实现能够显著提升同一 merge 任务族内的有效风险场景率”，但不等同于已经证明跨道路模板或真实道路的泛化能力。完整工件和审计记录见 [`results/sac_scenario_mining/README.md`](../results/sac_scenario_mining/README.md)。

## 代码结构

```text
sac_scenario_mining/
├── configs/merge_sac.yaml  # 场景划分、环境、奖励、SAC 与评估参数
├── src/                    # 可复用环境和领域逻辑
│   ├── env.py              # Gymnasium 环境、角色绑定、事件与终止
│   ├── metadrive_compat.py # 唯一的 MetaDrive 0.4.x 访问层
│   ├── observation.py      # 固定的 38 维观测
│   ├── reward.py           # TTC 和奖励计算
│   ├── metrics.py          # episode 指标与聚合
│   └── scenario_manifest.py# 可回放场景工件
└── scripts/                # 命令行入口
```

## 使用

在仓库根目录激活已安装 MetaDrive 的环境：

```powershell
conda activate metadrive
```

训练一个独立 SAC 运行：

```powershell
python -m sac_scenario_mining.scripts.train_sac --seed 0 --run-name merge_sac_seed0
```

评估随机基线或已训练策略：

```powershell
python -m sac_scenario_mining.scripts.evaluate --policy random --split test
python -m sac_scenario_mining.scripts.evaluate --policy-path results/sac_scenario_mining/merge_sac_seed2/best_model.zip --split test --deterministic
```

生成汇总和审计正式工件：

```powershell
python -m sac_scenario_mining.scripts.report
python -m sac_scenario_mining.scripts.audit_results
python -m sac_scenario_mining.scripts.audit_replays --scenarios-root results/sac_scenario_mining/final_eval/sac_seed2/critical_scenarios
```

不需要手动启动 `F:\PyCharm 2024.3.2\work\metadrive` 源码目录或独立服务。要在本地桌面窗口观看已训练策略，运行：

```powershell
python -m sac_scenario_mining.scripts.visualize
```

默认播放 SAC 训练 seed 2 在 held-out 场景 seed 2016 上的 rollout。精确回放已保存动作时可使用 `replay.py --manifest <manifest.json> --render topdown`。
