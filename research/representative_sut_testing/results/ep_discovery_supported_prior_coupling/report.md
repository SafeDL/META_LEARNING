# 联合 EP 推断下的发现损失支持

新执行 12 条 EP 策略曲线，复用 24 条冻结参照／ADF 控制；同 SUT、场景、先验、观测似然、选择规则与 200 次预算。
本轮只将 ADF 换为联合 EP 推断。选择规则保持不变：当联合工作后验中的候选无碰撞／参照碰撞概率不超过 1/sqrt(200)，预测发现更高且实际损失额度允许时，可跳过原定参照步骤。
不要求严格辨明潜在倾向排序。固定前缀至少 105 项的保证已移除，实际全程计数界保持不变。

| 方法 | Area % | Recall % | 平均 F200 |
|---|---:|---:|---:|
| frozen_reference | 52.837 | 76.885 | 118.92 |
| discovery_supported_prior | 53.052 | 76.915 | 119.08 |
| ep_discovery_supported_prior | 53.040 | 77.067 | 119.75 |

| 池 | SUT | 方法 | Area % | Recall % | F200 |
|---|---|---|---:|---:|---:|
| 0 | idm_ref | frozen_reference | 12.402 | 24.968 | 197 |
| 0 | idm_ref | discovery_supported_prior | 12.086 | 24.588 | 194 |
| 0 | idm_ref | ep_discovery_supported_prior | 12.077 | 24.588 | 194 |
| 0 | fvdm_target | frozen_reference | 69.812 | 100.000 | 88 |
| 0 | fvdm_target | discovery_supported_prior | 70.665 | 98.864 | 87 |
| 0 | fvdm_target | ep_discovery_supported_prior | 70.597 | 98.864 | 87 |
| 0 | mobil_ref_v2 | frozen_reference | 79.913 | 100.000 | 69 |
| 0 | mobil_ref_v2 | discovery_supported_prior | 80.254 | 100.000 | 69 |
| 0 | mobil_ref_v2 | ep_discovery_supported_prior | 80.181 | 100.000 | 69 |
| 0 | vi_ttc_ref_audit_v4 | frozen_reference | 26.585 | 53.145 | 169 |
| 0 | vi_ttc_ref_audit_v4 | discovery_supported_prior | 27.395 | 55.346 | 176 |
| 0 | vi_ttc_ref_audit_v4 | ep_discovery_supported_prior | 27.664 | 55.346 | 176 |
| 0 | mcts_cv_ref_audit_v4 | frozen_reference | 54.899 | 87.097 | 108 |
| 0 | mcts_cv_ref_audit_v4 | discovery_supported_prior | 55.391 | 87.903 | 109 |
| 0 | mcts_cv_ref_audit_v4 | ep_discovery_supported_prior | 55.242 | 87.903 | 109 |
| 0 | ppo_ref_v2 | frozen_reference | 76.675 | 100.000 | 83 |
| 0 | ppo_ref_v2 | discovery_supported_prior | 76.663 | 100.000 | 83 |
| 0 | ppo_ref_v2 | ep_discovery_supported_prior | 76.759 | 100.000 | 83 |
| 1 | idm_ref | frozen_reference | 12.519 | 25.031 | 199 |
| 1 | idm_ref | discovery_supported_prior | 12.519 | 25.031 | 199 |
| 1 | idm_ref | ep_discovery_supported_prior | 12.519 | 25.031 | 199 |
| 1 | fvdm_target | frozen_reference | 65.723 | 97.872 | 92 |
| 1 | fvdm_target | discovery_supported_prior | 65.660 | 97.872 | 92 |
| 1 | fvdm_target | ep_discovery_supported_prior | 65.106 | 96.809 | 91 |
| 1 | mobil_ref_v2 | frozen_reference | 78.021 | 100.000 | 73 |
| 1 | mobil_ref_v2 | discovery_supported_prior | 78.137 | 100.000 | 73 |
| 1 | mobil_ref_v2 | ep_discovery_supported_prior | 78.041 | 100.000 | 73 |
| 1 | vi_ttc_ref_audit_v4 | frozen_reference | 27.249 | 54.662 | 170 |
| 1 | vi_ttc_ref_audit_v4 | discovery_supported_prior | 27.788 | 54.341 | 169 |
| 1 | vi_ttc_ref_audit_v4 | ep_discovery_supported_prior | 28.423 | 57.235 | 178 |
| 1 | mcts_cv_ref_audit_v4 | frozen_reference | 53.016 | 79.839 | 99 |
| 1 | mcts_cv_ref_audit_v4 | discovery_supported_prior | 52.827 | 79.032 | 98 |
| 1 | mcts_cv_ref_audit_v4 | ep_discovery_supported_prior | 52.681 | 79.032 | 98 |
| 1 | ppo_ref_v2 | frozen_reference | 77.225 | 100.000 | 80 |
| 1 | ppo_ref_v2 | discovery_supported_prior | 77.244 | 100.000 | 80 |
| 1 | ppo_ref_v2 | ep_discovery_supported_prior | 77.194 | 100.000 | 80 |

查询前证据、联合概率、条件参照观测、实际反馈及原参照全程界核对通过。
EP 相对 ADF 的 Recall 提高 0.153、Area 下降 0.012 个百分点；相对既有 Student-t 受保护方法，两项均略低。本轮不晋升、不补种子，尚未建立显著优势。

全部 EP 更新最多 47 次迭代达到收敛阈值，每条曲线选择进程平均耗时约 145.6 s。453 次跳过的预测局部损失总和约 13.565，已付费标签支持实际损失区间 16–20；未披露参照结果不补读。支持量仍来自近似工作后验，数值收敛不等于概率校准，也不提供未知目标的频率保证。见 [推断与损失核查](inference_calibration_audit.json)、[较强方法对比](stronger_method_comparison.json)。
