# on-ramp merge 正式实验结果

本目录只保留当前参数化 `on_ramp_merge` 的正式训练、held-out 测试和回放证据。已清理的早期原型、冒烟测试和历史命名结果不应与这里的结论混用。

## 结果如何产生

| 目录或文件 | 生成入口 | 内容与用途 |
| --- | --- | --- |
| `merge_sac_seed0/`、`seed1/`、`seed2/` | `train_sac --seed N` | 三次各 300,000 环境步的独立 SAC 训练；保存解析后配置、版本、训练/验证 case 表、Monitor、优化日志和各 checkpoint 的验证汇总。 |
| `final_eval/random/` | `evaluate --policy random --seed 123` | 同一张 60-case held-out 表上的随机连续动作基线。 |
| `final_eval/sac_seed{0,1,2}/` | `evaluate --policy-path .../best_model.zip` | 三个按 validation 选择的 SAC 模型的 held-out 结果、episode 明细、case 表和 top-k 可回放关键场景。 |
| `final_eval/comparison.csv`、`plots/` | `report` | 策略对比、训练回报、actor/critic/熵系数损失、validation 指标和 TTC 分布图。 |
| `final_eval/acceptance_audit.json` | `audit_results` | 根据有效场景提升、无效率和目标驱动性进行自动验收。 |
| `final_eval/sac_seed2/replay_audit.json` | `audit_replays` | seed 2 top-10 关键场景的动作轨迹回放审计。 |

训练时，根目录的 `validation_case_table.json` 是各 validation checkpoint 的唯一固定 case 表；当前代码不会在以后重复写入相同 case 表。正式 held-out 评估目录仍各自保存 `case_table.json`，以保证结果自包含。

## 实验口径

MetaDrive 的 `SrS` 地图模板固定，地图引擎 seed 为 0。每个 case 显式指定两车初速度、纵向间隔、匝道起点、背景密度和背景随机 seed。SUT 是主线 IDM 车，SAC 每一步仅控制匝道 adversary 的 `[steering, throttle/brake]`。

训练、验证、测试分别为 96、24、60 个确定性且互不重叠的 case；Random 和三组 SAC 在完全相同的 60 个测试 case 上比较。`seed0/1/2` 是优化随机种子，而不是不同地图。

有效场景率为 `valid_critical_rate`：发生目标碰撞或最小 TTC 不高于 1.5 s，且此前未发生非目标碰撞、角色出界或对抗车逆行。无效率为 `invalid_rate`；目标碰撞率只统计 adversary 与固定 SUT 的碰撞。

## 正式 held-out 结果

| 策略 | 有效场景率 | 目标碰撞率 | 无效率 | 中位最小 TTC |
| --- | ---: | ---: | ---: | ---: |
| Random | 6.7% | 3.3% | 65.0% | 3.278 s |
| SAC seed 0 | 96.7% | 91.7% | 3.3% | 1.035 s |
| SAC seed 1 | 98.3% | 100.0% | 1.7% | 0.929 s |
| SAC seed 2 | 100.0% | 100.0% | 0.0% | 0.861 s |

`acceptance_audit.json` 显示三组 SAC 均通过验收：相对 Random 的有效场景率提升均超过 0.10、无效率不高于 0.25，且目标碰撞率高于非目标碰撞率和对抗车出界率。seed 2 的 10 个导出关键场景均已回放成功；最大 TTC 误差约 `8.4e-05 s`，低于 0.1 s 容差。

因此，结果说明 SAC 在本固定 `on_ramp_merge` 任务族内能稳定提高有效目标事故发现率，且相对于随机连续动作基线提升显著。结论不外推至真实道路、其他 MetaDrive 地图模板、其他交通分布或任意 ADS。

## 版本管理与兼容性

CSV、JSON、PNG、case 表、manifest 和 `actions.npy` 是可审计证据，应纳入版本管理。训练模型 `.zip`、TensorBoard 原始事件和日志是可再生大文件，按仓库 `.gitignore` 不提交。

较早归档的 case 记录中可能带有 `adversary_target_speed_mps`。该字段已被确认未参与运行时控制，当前代码不再生成它；回放旧 manifest 时会忽略它，原始训练和测试证据则保留不改写。
