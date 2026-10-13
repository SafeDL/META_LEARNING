# 被测驾驶算法

当前 highway-env 实验使用六个目标系统，固定标识见[基准配置](../experiments/driving_benchmark/config.py)。

| 目标系统 | 实现 |
|---|---|
| `idm_ref` | [reference_profiles.py](highway_env/reference_profiles.py)、[idm_profiles.py](highway_env/idm_profiles.py)：IDM 跟驰 |
| `fvdm_target` | [perception.py](highway_env/perception.py)：安全间距 FVDM，感知延迟 0.15 秒 |
| `mobil_ref_v2` | highway-env 原生 IDM + MOBIL，见 [registry.py](highway_env/registry.py) |
| `vi_ttc_ref_audit_v4` | [value_iteration.py](highway_env/value_iteration.py)：基于 TTC 的值迭代 |
| `mcts_cv_ref_audit_v4` | [mcts_cv.py](highway_env/mcts_cv.py)：恒速预测 MCTS |
| `ppo_ref_v2` | [ppo_ece.py](highway_env/ppo_ece.py)：冻结 PPO 策略 |

PPO 权重位于 `highway_env/checkpoints/ppo_ece/vd_1_5_trial_1.zip`。`v2/v4` 是冻结系统标识的一部分。

六个历史来源包括普通、延迟、制动受限与预测型 IDM/FVDM，参数见[历史配置](../methods/history_guided_testing/config.py)。它们用于提供历史记录，与上表目标系统分开组织。

`metadrive/` 保留 MetaDrive 控制器，当前比较使用 highway-env。结果见[统一比较](../results/concordance_gated_bayesian_risk_transfer/comparison.md)。
