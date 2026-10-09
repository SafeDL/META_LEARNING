# 历史与中性事件先验的有限筛查

六固定 SUT、两共享池、200 次唯一查询及共同十次初始化；两步排程和发现额度不变。
两个余量模型共享混合似然、历史安全代理均值及相关性结构；历史事件先验与固定 p=0.5 中性先验分别匹配边缘均值和方差。
为保持正报告条件均值，事件先验改变时边缘尺度也随唯一矩匹配解改变；绝对协方差并非完全相同。
无碰撞代理值为 1−Risk，碰撞只提供非正潜在余量约束；潜变量是工作模型，不是实际车距或原 Risk 的物理截断。
Gaussian ADF 为近似推断，不自动提供未知 SUT 上的校准保证。

| 方法 | Area % | Recall % | 平均 F200 |
|---|---:|---:|---:|
| frozen_reference | 52.837 | 76.885 | 118.92 |
| history_margin_response | 52.443 | 76.435 | 117.75 |
| neutral_margin_response | 52.638 | 76.813 | 119.00 |

| 池 | SUT | 方法 | Area % | Recall % | F200 |
|---|---|---|---:|---:|---:|
| 0 | idm_ref | frozen_reference | 12.402 | 24.968 | 197.00 |
| 0 | idm_ref | history_margin_response | 12.174 | 24.715 | 195.00 |
| 0 | idm_ref | neutral_margin_response | 12.190 | 24.715 | 195.00 |
| 0 | fvdm_target | frozen_reference | 69.812 | 100.000 | 88.00 |
| 0 | fvdm_target | history_margin_response | 69.909 | 98.864 | 87.00 |
| 0 | fvdm_target | neutral_margin_response | 68.653 | 98.864 | 87.00 |
| 0 | mobil_ref_v2 | frozen_reference | 79.913 | 100.000 | 69.00 |
| 0 | mobil_ref_v2 | history_margin_response | 80.210 | 100.000 | 69.00 |
| 0 | mobil_ref_v2 | neutral_margin_response | 79.659 | 100.000 | 69.00 |
| 0 | vi_ttc_ref_audit_v4 | frozen_reference | 26.585 | 53.145 | 169.00 |
| 0 | vi_ttc_ref_audit_v4 | history_margin_response | 25.687 | 52.201 | 166.00 |
| 0 | vi_ttc_ref_audit_v4 | neutral_margin_response | 26.939 | 53.774 | 171.00 |
| 0 | mcts_cv_ref_audit_v4 | frozen_reference | 54.899 | 87.097 | 108.00 |
| 0 | mcts_cv_ref_audit_v4 | history_margin_response | 53.315 | 87.097 | 108.00 |
| 0 | mcts_cv_ref_audit_v4 | neutral_margin_response | 55.298 | 87.903 | 109.00 |
| 0 | ppo_ref_v2 | frozen_reference | 76.675 | 100.000 | 83.00 |
| 0 | ppo_ref_v2 | history_margin_response | 76.645 | 100.000 | 83.00 |
| 0 | ppo_ref_v2 | neutral_margin_response | 76.157 | 100.000 | 83.00 |
| 1 | idm_ref | frozen_reference | 12.519 | 25.031 | 199.00 |
| 1 | idm_ref | history_margin_response | 12.519 | 25.031 | 199.00 |
| 1 | idm_ref | neutral_margin_response | 12.519 | 25.031 | 199.00 |
| 1 | fvdm_target | frozen_reference | 65.723 | 97.872 | 92.00 |
| 1 | fvdm_target | history_margin_response | 65.617 | 97.872 | 92.00 |
| 1 | fvdm_target | neutral_margin_response | 64.718 | 96.809 | 91.00 |
| 1 | mobil_ref_v2 | frozen_reference | 78.021 | 100.000 | 73.00 |
| 1 | mobil_ref_v2 | history_margin_response | 78.342 | 100.000 | 73.00 |
| 1 | mobil_ref_v2 | neutral_margin_response | 78.068 | 100.000 | 73.00 |
| 1 | vi_ttc_ref_audit_v4 | frozen_reference | 27.249 | 54.662 | 170.00 |
| 1 | vi_ttc_ref_audit_v4 | history_margin_response | 26.122 | 52.412 | 163.00 |
| 1 | vi_ttc_ref_audit_v4 | neutral_margin_response | 27.777 | 55.627 | 173.00 |
| 1 | mcts_cv_ref_audit_v4 | frozen_reference | 53.016 | 79.839 | 99.00 |
| 1 | mcts_cv_ref_audit_v4 | history_margin_response | 51.560 | 79.032 | 98.00 |
| 1 | mcts_cv_ref_audit_v4 | neutral_margin_response | 52.952 | 79.032 | 98.00 |
| 1 | ppo_ref_v2 | frozen_reference | 77.225 | 100.000 | 80.00 |
| 1 | ppo_ref_v2 | history_margin_response | 77.212 | 100.000 | 80.00 |
| 1 | ppo_ref_v2 | neutral_margin_response | 76.731 | 100.000 | 80.00 |

36 条曲线、7200 次披露与混合似然、原参照及全程计数界核对通过。
未通过筛查，不补种子。
