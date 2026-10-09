# EP 发现损失支持与反馈证据评分

新执行 12 条策略曲线，复用 24 条既有控制；同 SUT、场景与 200 次预算。
候选为 event_evidence_prior，证据评分为 event，直接机制对照为 ep_discovery_supported_prior。
两个组件继续接收全部实际 R/C；事件评分只移除连续安全代理密度对模型权重的直接贡献。
不要求严格辨明潜在倾向排序。固定前缀至少 105 项的保证已移除，实际全程计数界保持不变。

| 方法 | Area % | Recall % | 平均 F200 |
|---|---:|---:|---:|
| frozen_reference | 52.837 | 76.885 | 118.92 |
| ep_discovery_supported_prior | 53.040 | 77.067 | 119.75 |
| event_evidence_prior | 53.051 | 77.067 | 119.75 |

| 池 | SUT | 方法 | Area % | Recall % | F200 |
|---|---|---|---:|---:|---:|
| 0 | idm_ref | frozen_reference | 12.402 | 24.968 | 197 |
| 0 | idm_ref | ep_discovery_supported_prior | 12.077 | 24.588 | 194 |
| 0 | idm_ref | event_evidence_prior | 12.082 | 24.588 | 194 |
| 0 | fvdm_target | frozen_reference | 69.812 | 100.000 | 88 |
| 0 | fvdm_target | ep_discovery_supported_prior | 70.597 | 98.864 | 87 |
| 0 | fvdm_target | event_evidence_prior | 70.534 | 98.864 | 87 |
| 0 | mobil_ref_v2 | frozen_reference | 79.913 | 100.000 | 69 |
| 0 | mobil_ref_v2 | ep_discovery_supported_prior | 80.181 | 100.000 | 69 |
| 0 | mobil_ref_v2 | event_evidence_prior | 80.181 | 100.000 | 69 |
| 0 | vi_ttc_ref_audit_v4 | frozen_reference | 26.585 | 53.145 | 169 |
| 0 | vi_ttc_ref_audit_v4 | ep_discovery_supported_prior | 27.664 | 55.346 | 176 |
| 0 | vi_ttc_ref_audit_v4 | event_evidence_prior | 27.664 | 55.346 | 176 |
| 0 | mcts_cv_ref_audit_v4 | frozen_reference | 54.899 | 87.097 | 108 |
| 0 | mcts_cv_ref_audit_v4 | ep_discovery_supported_prior | 55.242 | 87.903 | 109 |
| 0 | mcts_cv_ref_audit_v4 | event_evidence_prior | 55.242 | 87.903 | 109 |
| 0 | ppo_ref_v2 | frozen_reference | 76.675 | 100.000 | 83 |
| 0 | ppo_ref_v2 | ep_discovery_supported_prior | 76.759 | 100.000 | 83 |
| 0 | ppo_ref_v2 | event_evidence_prior | 76.819 | 100.000 | 83 |
| 1 | idm_ref | frozen_reference | 12.519 | 25.031 | 199 |
| 1 | idm_ref | ep_discovery_supported_prior | 12.519 | 25.031 | 199 |
| 1 | idm_ref | event_evidence_prior | 12.519 | 25.031 | 199 |
| 1 | fvdm_target | frozen_reference | 65.723 | 97.872 | 92 |
| 1 | fvdm_target | ep_discovery_supported_prior | 65.106 | 96.809 | 91 |
| 1 | fvdm_target | event_evidence_prior | 65.112 | 96.809 | 91 |
| 1 | mobil_ref_v2 | frozen_reference | 78.021 | 100.000 | 73 |
| 1 | mobil_ref_v2 | ep_discovery_supported_prior | 78.041 | 100.000 | 73 |
| 1 | mobil_ref_v2 | event_evidence_prior | 78.151 | 100.000 | 73 |
| 1 | vi_ttc_ref_audit_v4 | frozen_reference | 27.249 | 54.662 | 170 |
| 1 | vi_ttc_ref_audit_v4 | ep_discovery_supported_prior | 28.423 | 57.235 | 178 |
| 1 | vi_ttc_ref_audit_v4 | event_evidence_prior | 28.423 | 57.235 | 178 |
| 1 | mcts_cv_ref_audit_v4 | frozen_reference | 53.016 | 79.839 | 99 |
| 1 | mcts_cv_ref_audit_v4 | ep_discovery_supported_prior | 52.681 | 79.032 | 98 |
| 1 | mcts_cv_ref_audit_v4 | event_evidence_prior | 52.681 | 79.032 | 98 |
| 1 | ppo_ref_v2 | frozen_reference | 77.225 | 100.000 | 80 |
| 1 | ppo_ref_v2 | ep_discovery_supported_prior | 77.194 | 100.000 | 80 |
| 1 | ppo_ref_v2 | event_evidence_prior | 77.200 | 100.000 | 80 |

查询前证据、联合概率、条件参照观测、实际反馈及原参照全程界核对通过。
支持量来自 EP 工作后验，不是未知目标的频率保证；本轮不自动补种子或宣称显著优势。

本轮相对原 EP 的 Area +0.010 个百分点、Recall 不变，12 条曲线的终点发现数全部相同，4 条查询轨迹完全一致。相对 Student-t 受保护方法，Area +0.002、Recall −0.089；相对无保护方法，Recall −0.405。未通过双指标门槛，不晋升、不补种子。

27 项相关检查通过；实际权重时序、动作、联合概率、独立参照与全程计数界均核对。事件评分对数损失恒等式在所有前缀成立，只比较实际共同查询轨迹上的组件预测，不保证未查询场景的排序。434 次参照跳过的预测局部损失总和为 12.801，已付费标签支持区间 18–22；未披露参照结果不补读。全部 EP 更新最多 47 次迭代收敛，平均选择进程耗时约 142.1 s。[结果核查](outcome_audit.json)。

两个已有解释的评分变化没有增加终点发现。下一步检查独立事件响应与几何结构，继续沿用原实验对象与预算；具体设计只维护在 [唯一设计文档](../../../../docs/Failure_Discovery_Design.md)。
