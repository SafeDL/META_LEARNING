# 反馈解释与发现损失支持的实际对照

新执行 12 条策略曲线，复用 24 条既有控制；同 SUT、场景与 200 次预算。
候选为 report_binary_prior，证据评分为 event，直接机制对照为 event_evidence_prior。
两组件接收同次执行的真实反馈；权重按查询前评分更新，第二组件的观测解释由协议锁定。
不要求严格辨明潜在倾向排序。固定前缀至少 105 项的保证已移除，实际全程计数界保持不变。

| 方法 | Area % | Recall % | 平均 F200 |
|---|---:|---:|---:|
| frozen_reference | 52.837 | 76.885 | 118.92 |
| event_evidence_prior | 53.051 | 77.067 | 119.75 |
| report_binary_prior | 53.088 | 77.011 | 119.50 |

| 池 | SUT | 方法 | Area % | Recall % | F200 |
|---|---|---|---:|---:|---:|
| 0 | idm_ref | frozen_reference | 12.402 | 24.968 | 197 |
| 0 | idm_ref | event_evidence_prior | 12.082 | 24.588 | 194 |
| 0 | idm_ref | report_binary_prior | 12.196 | 24.715 | 195 |
| 0 | fvdm_target | frozen_reference | 69.812 | 100.000 | 88 |
| 0 | fvdm_target | event_evidence_prior | 70.534 | 98.864 | 87 |
| 0 | fvdm_target | report_binary_prior | 70.534 | 98.864 | 87 |
| 0 | mobil_ref_v2 | frozen_reference | 79.913 | 100.000 | 69 |
| 0 | mobil_ref_v2 | event_evidence_prior | 80.181 | 100.000 | 69 |
| 0 | mobil_ref_v2 | report_binary_prior | 80.130 | 100.000 | 69 |
| 0 | vi_ttc_ref_audit_v4 | frozen_reference | 26.585 | 53.145 | 169 |
| 0 | vi_ttc_ref_audit_v4 | event_evidence_prior | 27.664 | 55.346 | 176 |
| 0 | vi_ttc_ref_audit_v4 | report_binary_prior | 27.575 | 55.346 | 176 |
| 0 | mcts_cv_ref_audit_v4 | frozen_reference | 54.899 | 87.097 | 108 |
| 0 | mcts_cv_ref_audit_v4 | event_evidence_prior | 55.242 | 87.903 | 109 |
| 0 | mcts_cv_ref_audit_v4 | report_binary_prior | 55.508 | 87.903 | 109 |
| 0 | ppo_ref_v2 | frozen_reference | 76.675 | 100.000 | 83 |
| 0 | ppo_ref_v2 | event_evidence_prior | 76.819 | 100.000 | 83 |
| 0 | ppo_ref_v2 | report_binary_prior | 76.693 | 100.000 | 83 |
| 1 | idm_ref | frozen_reference | 12.519 | 25.031 | 199 |
| 1 | idm_ref | event_evidence_prior | 12.519 | 25.031 | 199 |
| 1 | idm_ref | report_binary_prior | 12.519 | 25.031 | 199 |
| 1 | fvdm_target | frozen_reference | 65.723 | 97.872 | 92 |
| 1 | fvdm_target | event_evidence_prior | 65.112 | 96.809 | 91 |
| 1 | fvdm_target | report_binary_prior | 65.271 | 96.809 | 91 |
| 1 | mobil_ref_v2 | frozen_reference | 78.021 | 100.000 | 73 |
| 1 | mobil_ref_v2 | event_evidence_prior | 78.151 | 100.000 | 73 |
| 1 | mobil_ref_v2 | report_binary_prior | 78.151 | 100.000 | 73 |
| 1 | vi_ttc_ref_audit_v4 | frozen_reference | 27.249 | 54.662 | 170 |
| 1 | vi_ttc_ref_audit_v4 | event_evidence_prior | 28.423 | 57.235 | 178 |
| 1 | vi_ttc_ref_audit_v4 | report_binary_prior | 28.275 | 55.627 | 173 |
| 1 | mcts_cv_ref_audit_v4 | frozen_reference | 53.016 | 79.839 | 99 |
| 1 | mcts_cv_ref_audit_v4 | event_evidence_prior | 52.681 | 79.032 | 98 |
| 1 | mcts_cv_ref_audit_v4 | report_binary_prior | 52.976 | 79.839 | 99 |
| 1 | ppo_ref_v2 | frozen_reference | 77.225 | 100.000 | 80 |
| 1 | ppo_ref_v2 | event_evidence_prior | 77.200 | 100.000 | 80 |
| 1 | ppo_ref_v2 | report_binary_prior | 77.225 | 100.000 | 80 |

查询前证据、联合概率、条件参照观测、实际反馈及原参照全程界核对通过。
支持量来自 EP 工作后验，不是未知目标的频率保证；本轮不自动补种子或宣称显著优势。
