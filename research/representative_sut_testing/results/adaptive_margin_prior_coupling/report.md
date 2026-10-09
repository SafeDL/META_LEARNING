# 初始反馈先验选择与持续先验平均

六固定 SUT、两共享池、200 次唯一查询、共同十次初始化、相同两步排程及发现额度。
新增 24 条真实选择器曲线，36 条固定控制从已执行阶段复用，不增加独立证据。
权重使用查询前的事件质量或正代理值密度，实际反馈后更新；初始 MAP 在第十次反馈后锁定，持续平均则逐次更新。
ADF 为工作近似，模型平均与先验选择本身不是新的通用理论。

| 方法 | Area % | Recall % | 平均 F200 |
|---|---:|---:|---:|
| frozen_reference | 52.837 | 76.885 | 118.92 |
| history_margin_response | 52.443 | 76.435 | 117.75 |
| neutral_margin_response | 52.638 | 76.813 | 119.00 |
| initial_prior_selection | 52.966 | 76.902 | 119.08 |
| online_prior_average | 53.005 | 76.902 | 119.08 |

| 池 | SUT | 方法 | Area % | Recall % | F200 |
|---|---|---|---:|---:|---:|
| 0 | idm_ref | frozen_reference | 12.402 | 24.968 | 197 |
| 0 | idm_ref | history_margin_response | 12.174 | 24.715 | 195 |
| 0 | idm_ref | neutral_margin_response | 12.190 | 24.715 | 195 |
| 0 | idm_ref | initial_prior_selection | 12.174 | 24.715 | 195 |
| 0 | idm_ref | online_prior_average | 12.190 | 24.715 | 195 |
| 0 | fvdm_target | frozen_reference | 69.812 | 100.000 | 88 |
| 0 | fvdm_target | history_margin_response | 69.909 | 98.864 | 87 |
| 0 | fvdm_target | neutral_margin_response | 68.653 | 98.864 | 87 |
| 0 | fvdm_target | initial_prior_selection | 69.909 | 98.864 | 87 |
| 0 | fvdm_target | online_prior_average | 70.085 | 98.864 | 87 |
| 0 | mobil_ref_v2 | frozen_reference | 79.913 | 100.000 | 69 |
| 0 | mobil_ref_v2 | history_margin_response | 80.210 | 100.000 | 69 |
| 0 | mobil_ref_v2 | neutral_margin_response | 79.659 | 100.000 | 69 |
| 0 | mobil_ref_v2 | initial_prior_selection | 80.210 | 100.000 | 69 |
| 0 | mobil_ref_v2 | online_prior_average | 80.254 | 100.000 | 69 |
| 0 | vi_ttc_ref_audit_v4 | frozen_reference | 26.585 | 53.145 | 169 |
| 0 | vi_ttc_ref_audit_v4 | history_margin_response | 25.687 | 52.201 | 166 |
| 0 | vi_ttc_ref_audit_v4 | neutral_margin_response | 26.939 | 53.774 | 171 |
| 0 | vi_ttc_ref_audit_v4 | initial_prior_selection | 26.939 | 53.774 | 171 |
| 0 | vi_ttc_ref_audit_v4 | online_prior_average | 26.939 | 53.774 | 171 |
| 0 | mcts_cv_ref_audit_v4 | frozen_reference | 54.899 | 87.097 | 108 |
| 0 | mcts_cv_ref_audit_v4 | history_margin_response | 53.315 | 87.097 | 108 |
| 0 | mcts_cv_ref_audit_v4 | neutral_margin_response | 55.298 | 87.903 | 109 |
| 0 | mcts_cv_ref_audit_v4 | initial_prior_selection | 55.298 | 87.903 | 109 |
| 0 | mcts_cv_ref_audit_v4 | online_prior_average | 55.298 | 87.903 | 109 |
| 0 | ppo_ref_v2 | frozen_reference | 76.675 | 100.000 | 83 |
| 0 | ppo_ref_v2 | history_margin_response | 76.645 | 100.000 | 83 |
| 0 | ppo_ref_v2 | neutral_margin_response | 76.157 | 100.000 | 83 |
| 0 | ppo_ref_v2 | initial_prior_selection | 76.645 | 100.000 | 83 |
| 0 | ppo_ref_v2 | online_prior_average | 76.548 | 100.000 | 83 |
| 1 | idm_ref | frozen_reference | 12.519 | 25.031 | 199 |
| 1 | idm_ref | history_margin_response | 12.519 | 25.031 | 199 |
| 1 | idm_ref | neutral_margin_response | 12.519 | 25.031 | 199 |
| 1 | idm_ref | initial_prior_selection | 12.519 | 25.031 | 199 |
| 1 | idm_ref | online_prior_average | 12.519 | 25.031 | 199 |
| 1 | fvdm_target | frozen_reference | 65.723 | 97.872 | 92 |
| 1 | fvdm_target | history_margin_response | 65.617 | 97.872 | 92 |
| 1 | fvdm_target | neutral_margin_response | 64.718 | 96.809 | 91 |
| 1 | fvdm_target | initial_prior_selection | 65.617 | 97.872 | 92 |
| 1 | fvdm_target | online_prior_average | 66.170 | 97.872 | 92 |
| 1 | mobil_ref_v2 | frozen_reference | 78.021 | 100.000 | 73 |
| 1 | mobil_ref_v2 | history_margin_response | 78.342 | 100.000 | 73 |
| 1 | mobil_ref_v2 | neutral_margin_response | 78.068 | 100.000 | 73 |
| 1 | mobil_ref_v2 | initial_prior_selection | 78.342 | 100.000 | 73 |
| 1 | mobil_ref_v2 | online_prior_average | 78.096 | 100.000 | 73 |
| 1 | vi_ttc_ref_audit_v4 | frozen_reference | 27.249 | 54.662 | 170 |
| 1 | vi_ttc_ref_audit_v4 | history_margin_response | 26.122 | 52.412 | 163 |
| 1 | vi_ttc_ref_audit_v4 | neutral_margin_response | 27.777 | 55.627 | 173 |
| 1 | vi_ttc_ref_audit_v4 | initial_prior_selection | 27.777 | 55.627 | 173 |
| 1 | vi_ttc_ref_audit_v4 | online_prior_average | 27.777 | 55.627 | 173 |
| 1 | mcts_cv_ref_audit_v4 | frozen_reference | 53.016 | 79.839 | 99 |
| 1 | mcts_cv_ref_audit_v4 | history_margin_response | 51.560 | 79.032 | 98 |
| 1 | mcts_cv_ref_audit_v4 | neutral_margin_response | 52.952 | 79.032 | 98 |
| 1 | mcts_cv_ref_audit_v4 | initial_prior_selection | 52.952 | 79.032 | 98 |
| 1 | mcts_cv_ref_audit_v4 | online_prior_average | 52.952 | 79.032 | 98 |
| 1 | ppo_ref_v2 | frozen_reference | 77.225 | 100.000 | 80 |
| 1 | ppo_ref_v2 | history_margin_response | 77.212 | 100.000 | 80 |
| 1 | ppo_ref_v2 | neutral_margin_response | 76.731 | 100.000 | 80 |
| 1 | ppo_ref_v2 | initial_prior_selection | 77.212 | 100.000 | 80 |
| 1 | ppo_ref_v2 | online_prior_average | 77.237 | 100.000 | 80 |

初始选择的真实轨迹与对应纯策略逐项一致，持续平均的预测权重、反馈更新、独立参照与计数界均核对通过。
本轮为有限机制验证，不自动追加种子，未进行独立显著性确认。
