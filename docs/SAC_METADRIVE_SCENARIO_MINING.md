# MetaDrive `on_ramp_merge` 的 SAC 对抗场景生成目标

## 已实现的目标

在 MetaDrive 固定 `SrS` 路网的入口匝道并入段上，使用 SAC 逐步控制匝道对抗车的连续动作 `[steering, throttle/brake]`；主线 SUT 固定在真正的汇入车道并由 IDM 基准控制。该工作不是 AEB 或任何具体 ADS 测试。

测试用例不是一个“地图 seed”。每个 case 为：

```text
case = (case_id, theta, background_seed, policy_seed, action_trace)
```

其中 `theta` 显式包含主车初速、对抗车初速、纵向间隙、匝道起点和背景密度。地图引擎 seed 固定为 0，保证 `SrS` 的 `I-S-r-S` 拓扑和匝道车道几何不变；`background_seed` 仅决定显式的背景车。SAC 不修改 `theta`，也不控制 SUT。

## 正确的场景行为

1. reset 时验证 adversary 在入口匝道、SUT 在汇入主线车道，初始无重叠且存在可行汇入时间窗。
2. SUT 由 IDM 沿主线行驶；adversary 每个物理步仅接收 SAC 的连续转向与油门/制动动作。
3. 背景车（若 case 启用）独立使用 IDM，永不决定 SUT/adversary 身份。
4. 目标事件只能是 adversary 与固定 SUT 的碰撞；出界、逆行、非目标碰撞不能换取成功。
5. 每个评估 case 保存完整参数、角色、动作轨迹和结果，可精确回放。

```text
critical       = target_collision OR min_ttc <= 1.5 s
valid          = critical 前没有非目标碰撞、角色出界或 adversary 逆行
valid_critical = critical AND valid
```

## 配置与划分

配置唯一入口为 `sac_scenario_mining/configs/merge_sac.yaml`：

| 划分 | case 数 | 用途 |
| --- | ---: | --- |
| train | 96 | SAC 交互与 replay buffer |
| validation | 24 | checkpoint 选择，不参与最终结论 |
| test | 60 | Random/SAC 的共同 held-out case 表 |

训练 seed `0/1/2` 只改变网络初始化、策略采样与优化随机性。所有 seed 使用同一 case 表；这检验同一逻辑场景任务族内的训练稳定性，而非新的地图拓扑泛化。

## 奖励与预期学习曲线

奖励由低 TTC、接近度、目标碰撞奖励、非目标碰撞/出界/逆行惩罚以及动作平滑项组成。目标碰撞奖励设为 200，必须高于仅在低 TTC 状态拖延到 horizon 的累计收益，避免把近失误优化为最终目标。

正式报告必须分别产生：

- `train_episode_return.png`：原始 return 与滚动均值；滚动均值探索后提升并趋稳即可，不要求单调。
- `sac_losses.png`：actor loss、critic loss 和熵系数；应有界，无 NaN/Inf 或持续爆炸。
- `validation_metrics.png`：固定 validation case 上的有效危险率与无效率；前者上升、后者受控。
- `held_out_comparison.png`：最终 Random 与三组 SAC 的 held-out 对比，绝不当作学习曲线。

checkpoint 选择优先级是目标碰撞率、有效危险率、低无效率和低 TTC；因此场景生成目标与模型选择一致。

## 2026-07-19 完整实验结果

每个 SAC seed 完整训练 300,000 环境步（GPU 批量更新），在同一张 held-out 60-case 表上评估：

| 策略 | 有效危险率 | 目标碰撞率 | 无效率 |
| --- | ---: | ---: | ---: |
| Random | 6.7% | 3.3% | 65.0% |
| SAC seed 0 | 96.7% | 91.7% | 3.3% |
| SAC seed 1 | 98.3% | 100.0% | 1.7% |
| SAC seed 2 | 100.0% | 100.0% | 0.0% |

三组均通过自动验收；seed 2 导出的 top-10 关键 case 回放为 10/10 一致，TTC 误差小于 0.1 s。这支持“在本固定 `on_ramp_merge` 逻辑场景族中，SAC 显著提升有效目标事故发现率”的结论，但不支持对真实道路、其他地图模板或具体 ADS 的泛化主张。

## 保持的目录结构与入口

```text
sac_scenario_mining/
├── configs/merge_sac.yaml
├── src/          # casebook、环境、MetaDrive 适配、观测、奖励、指标与 manifest
└── scripts/      # train_sac、evaluate、replay、visualize、report、审计
```

```powershell
conda activate metadrive
python -m sac_scenario_mining.scripts.visualize --case-id test_000
python -m sac_scenario_mining.scripts.report
python -m sac_scenario_mining.scripts.audit_results
```

MetaDrive 位于 `F:\PyCharm 2024.3.2\work\metadrive`，作为 Python 包使用，无需手动启动服务。
