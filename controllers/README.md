# 被测驾驶控制器

当前实验使用以下六种驾驶控制器，代码中的固定标识用于对应实测数据和实验结果。

| 驾驶控制器 | 记录标识 | 实现 |
|---|---|---|
| IDM 跟驰 | `idm_ref` | [reference_profiles.py](highway/reference_profiles.py)、[idm_profiles.py](highway/idm_profiles.py) |
| FVDM 跟驰 | `fvdm_target` | [perception.py](highway/perception.py)，感知延迟 0.15 秒 |
| IDM＋MOBIL | `mobil_ref_v2` | [registry.py](highway/registry.py) |
| 碰撞时间值迭代 | `vi_ttc_ref_audit_v4` | [value_iteration.py](highway/value_iteration.py) |
| 恒速预测蒙特卡洛树搜索 | `mcts_cv_ref_audit_v4` | [mcts_cv.py](highway/mcts_cv.py) |
| PPO | `ppo_ref_v2` | [ppo_ece.py](highway/ppo_ece.py) |

PPO 使用已有权重 `highway/checkpoints/ppo_ece/vd_1_5_trial_1.zip`。标识中的 `v2`、`v4` 沿用已有数据记录。

历史数据来自另外六种 IDM／FVDM 配置，包括普通、延迟、制动受限和预测型配置，参数见[历史配置](../methods/baselines/hgt/config.py)。GCRT 的周期更新使用 TTC 值迭代和 FVDM 记录，停止校准使用恒速预测蒙特卡洛树搜索和 PPO，后续效果在 IDM＋MOBIL、IDM 上评价，分组见[方法配置](../methods/gcrt/config.py)。

`highway/` 用于当前论文实验，`metadrive/` 为保留的 MetaDrive 控制器。实测记录见 [data](../data/README.md)。
