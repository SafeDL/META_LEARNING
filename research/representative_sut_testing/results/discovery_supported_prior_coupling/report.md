# 发现损失支持下的参照观测

新执行 12 条策略曲线，复用 24 条既有控制；同 SUT、场景、模型与 200 次预算。
只修改参照时段：当联合工作后验中的候选无碰撞／参照碰撞概率不超过 1/sqrt(200)，且实际损失额度允许时，继续候选。
不要求严格辨明潜在倾向排序。固定前缀至少 105 项的保证已移除，实际全程计数界保持不变。

| 方法 | Area % | Recall % | 平均 F200 |
|---|---:|---:|---:|
| frozen_reference | 52.837 | 76.885 | 118.92 |
| online_prior_average | 53.005 | 76.902 | 119.08 |
| discovery_supported_prior | 53.052 | 76.915 | 119.08 |

| 池 | SUT | 方法 | Area % | Recall % | F200 |
|---|---|---|---:|---:|---:|
| 0 | idm_ref | frozen_reference | 12.402 | 24.968 | 197 |
| 0 | idm_ref | online_prior_average | 12.190 | 24.715 | 195 |
| 0 | idm_ref | discovery_supported_prior | 12.086 | 24.588 | 194 |
| 0 | fvdm_target | frozen_reference | 69.812 | 100.000 | 88 |
| 0 | fvdm_target | online_prior_average | 70.085 | 98.864 | 87 |
| 0 | fvdm_target | discovery_supported_prior | 70.665 | 98.864 | 87 |
| 0 | mobil_ref_v2 | frozen_reference | 79.913 | 100.000 | 69 |
| 0 | mobil_ref_v2 | online_prior_average | 80.254 | 100.000 | 69 |
| 0 | mobil_ref_v2 | discovery_supported_prior | 80.254 | 100.000 | 69 |
| 0 | vi_ttc_ref_audit_v4 | frozen_reference | 26.585 | 53.145 | 169 |
| 0 | vi_ttc_ref_audit_v4 | online_prior_average | 26.939 | 53.774 | 171 |
| 0 | vi_ttc_ref_audit_v4 | discovery_supported_prior | 27.395 | 55.346 | 176 |
| 0 | mcts_cv_ref_audit_v4 | frozen_reference | 54.899 | 87.097 | 108 |
| 0 | mcts_cv_ref_audit_v4 | online_prior_average | 55.298 | 87.903 | 109 |
| 0 | mcts_cv_ref_audit_v4 | discovery_supported_prior | 55.391 | 87.903 | 109 |
| 0 | ppo_ref_v2 | frozen_reference | 76.675 | 100.000 | 83 |
| 0 | ppo_ref_v2 | online_prior_average | 76.548 | 100.000 | 83 |
| 0 | ppo_ref_v2 | discovery_supported_prior | 76.663 | 100.000 | 83 |
| 1 | idm_ref | frozen_reference | 12.519 | 25.031 | 199 |
| 1 | idm_ref | online_prior_average | 12.519 | 25.031 | 199 |
| 1 | idm_ref | discovery_supported_prior | 12.519 | 25.031 | 199 |
| 1 | fvdm_target | frozen_reference | 65.723 | 97.872 | 92 |
| 1 | fvdm_target | online_prior_average | 66.170 | 97.872 | 92 |
| 1 | fvdm_target | discovery_supported_prior | 65.660 | 97.872 | 92 |
| 1 | mobil_ref_v2 | frozen_reference | 78.021 | 100.000 | 73 |
| 1 | mobil_ref_v2 | online_prior_average | 78.096 | 100.000 | 73 |
| 1 | mobil_ref_v2 | discovery_supported_prior | 78.137 | 100.000 | 73 |
| 1 | vi_ttc_ref_audit_v4 | frozen_reference | 27.249 | 54.662 | 170 |
| 1 | vi_ttc_ref_audit_v4 | online_prior_average | 27.777 | 55.627 | 173 |
| 1 | vi_ttc_ref_audit_v4 | discovery_supported_prior | 27.788 | 54.341 | 169 |
| 1 | mcts_cv_ref_audit_v4 | frozen_reference | 53.016 | 79.839 | 99 |
| 1 | mcts_cv_ref_audit_v4 | online_prior_average | 52.952 | 79.032 | 98 |
| 1 | mcts_cv_ref_audit_v4 | discovery_supported_prior | 52.827 | 79.032 | 98 |
| 1 | ppo_ref_v2 | frozen_reference | 77.225 | 100.000 | 80 |
| 1 | ppo_ref_v2 | online_prior_average | 77.237 | 100.000 | 80 |
| 1 | ppo_ref_v2 | discovery_supported_prior | 77.244 | 100.000 | 80 |

查询前证据、联合概率、条件参照观测、实际反馈及原参照全程界核对通过。
支持量来自 ADF 工作后验，不是未知目标的频率保证；本轮不自动补种子或宣称显著优势。
