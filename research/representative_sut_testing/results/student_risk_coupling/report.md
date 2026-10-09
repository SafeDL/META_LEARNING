# 冻结原方法参照上的受限纠正

六固定 SUT、两共享池、选择种子 [11]、200 次唯一查询和共同原参照前十次初始化。
候选及去耦合对照使用同六来源、同历史训练划分、同核和实际风险／事件反馈；冻结参照保持原风险权限。

风险 GP 只由实际风险更新；历史风险到事件的平均映射经目标场景事件残差修正，允许同风险有不同碰撞响应。
事件残差采用 probit 拉普拉斯近似；风险映射的常量段精确积分，线性段在概率域作十六节点积分，不读取未查询标签。

事件残差同时包含场景类内共享校准偏差和局部变化，两项归一化后保持与上一轮相同的初始边缘方差。

候选风险 GP 加入同一六来源的经验响应差异协方差；初始风险均值、噪声、学习核与事件推断保持不变。

每类已观测场景的整体风险偏移由目标风险反馈作广义最小二乘估计，预测方差包含均值估计不确定性。

风险尺度不再固定：逆伽马尺度先验的自由度预先固定为 5，已查询风险的残差二次型更新尺度后验。
积分使用 Student-t 预测分布；扣除已观测类型均值参数的自由度，均值保持原广义最小二乘形式。
信号和数值噪声共享尺度；这是工作响应模型，不是未知目标上的频率校准保证。

候选保留参照下一项的功能场景类型，只在同类中预测事件概率严格更高时改序；模型无区别时执行原参照。

本轮只修改探测额度：L_t≤1+F_actual(t)/sqrt(200)，执行候选前预留一个最坏情况非碰撞。额度仅由固定预算和实际已发现数决定。
全程界为 F_actual(t)≥(F_reference(t)−1)/(1+1/sqrt(200))；这是比原最多损失一个发现更弱的保证。

| 方法 | Area % | Recall % | F200 | 平均偏离参照查询数 |
|---|---:|---:|---:|---:|
| frozen_reference | 52.837 | 76.885 | 118.92 | 0.00 |
| uncoupled_response | 53.025 | 77.473 | 118.50 | 155.42 |
| coupled_response | 53.048 | 77.157 | 119.25 | 103.75 |

| 池 | SUT | 方法 | Area % | Recall % | F200 | 最大未接纳非碰撞数 | 不同序列数 |
|---|---|---|---:|---:|---:|---:|---:|
| 0 | idm_ref | frozen_reference | 12.402 | 24.968 | 197.00 | 0 | 1 |
| 0 | idm_ref | uncoupled_response | 11.176 | 22.053 | 174.00 | 24 | 1 |
| 0 | idm_ref | coupled_response | 12.010 | 24.208 | 191.00 | 7 | 1 |
| 0 | fvdm_target | frozen_reference | 69.812 | 100.000 | 88.00 | 0 | 1 |
| 0 | fvdm_target | uncoupled_response | 69.119 | 97.727 | 86.00 | 71 | 1 |
| 0 | fvdm_target | coupled_response | 69.534 | 98.864 | 87.00 | 7 | 1 |
| 0 | mobil_ref_v2 | frozen_reference | 79.913 | 100.000 | 69.00 | 0 | 1 |
| 0 | mobil_ref_v2 | uncoupled_response | 80.000 | 100.000 | 69.00 | 39 | 1 |
| 0 | mobil_ref_v2 | coupled_response | 79.949 | 100.000 | 69.00 | 5 | 1 |
| 0 | vi_ttc_ref_audit_v4 | frozen_reference | 26.585 | 53.145 | 169.00 | 0 | 1 |
| 0 | vi_ttc_ref_audit_v4 | uncoupled_response | 27.535 | 54.717 | 174.00 | 20 | 1 |
| 0 | vi_ttc_ref_audit_v4 | coupled_response | 27.621 | 55.660 | 177.00 | 11 | 1 |
| 0 | mcts_cv_ref_audit_v4 | frozen_reference | 54.899 | 87.097 | 108.00 | 0 | 1 |
| 0 | mcts_cv_ref_audit_v4 | uncoupled_response | 56.105 | 88.710 | 110.00 | 69 | 1 |
| 0 | mcts_cv_ref_audit_v4 | coupled_response | 55.440 | 87.903 | 109.00 | 8 | 1 |
| 0 | ppo_ref_v2 | frozen_reference | 76.675 | 100.000 | 83.00 | 0 | 1 |
| 0 | ppo_ref_v2 | uncoupled_response | 75.355 | 100.000 | 83.00 | 35 | 1 |
| 0 | ppo_ref_v2 | coupled_response | 76.693 | 100.000 | 83.00 | 6 | 1 |
| 1 | idm_ref | frozen_reference | 12.519 | 25.031 | 199.00 | 0 | 1 |
| 1 | idm_ref | uncoupled_response | 12.519 | 25.031 | 199.00 | 0 | 1 |
| 1 | idm_ref | coupled_response | 12.519 | 25.031 | 199.00 | 0 | 1 |
| 1 | fvdm_target | frozen_reference | 65.723 | 97.872 | 92.00 | 0 | 1 |
| 1 | fvdm_target | uncoupled_response | 66.101 | 100.000 | 94.00 | 63 | 1 |
| 1 | fvdm_target | coupled_response | 65.463 | 96.809 | 91.00 | 7 | 1 |
| 1 | mobil_ref_v2 | frozen_reference | 78.021 | 100.000 | 73.00 | 0 | 1 |
| 1 | mobil_ref_v2 | uncoupled_response | 77.568 | 100.000 | 73.00 | 64 | 1 |
| 1 | mobil_ref_v2 | coupled_response | 78.068 | 100.000 | 73.00 | 6 | 1 |
| 1 | vi_ttc_ref_audit_v4 | frozen_reference | 27.249 | 54.662 | 170.00 | 0 | 1 |
| 1 | vi_ttc_ref_audit_v4 | uncoupled_response | 27.998 | 55.949 | 174.00 | 22 | 1 |
| 1 | vi_ttc_ref_audit_v4 | coupled_response | 27.743 | 54.341 | 169.00 | 12 | 1 |
| 1 | mcts_cv_ref_audit_v4 | frozen_reference | 53.016 | 79.839 | 99.00 | 0 | 1 |
| 1 | mcts_cv_ref_audit_v4 | uncoupled_response | 55.427 | 85.484 | 106.00 | 56 | 1 |
| 1 | mcts_cv_ref_audit_v4 | coupled_response | 54.133 | 83.065 | 103.00 | 8 | 1 |
| 1 | ppo_ref_v2 | frozen_reference | 77.225 | 100.000 | 80.00 | 0 | 1 |
| 1 | ppo_ref_v2 | uncoupled_response | 77.394 | 100.000 | 80.00 | 73 | 1 |
| 1 | ppo_ref_v2 | coupled_response | 77.406 | 100.000 | 80.00 | 6 | 1 |

36 条曲线、7200 次披露核对通过。
独立参照与既有冻结路径完全一致；回放只使用已测反馈。每个实际前缀的全程计数界、终点与累计面积界通过核对。
相对 frozen_reference：Area +0.212、Recall +0.272 个百分点。
相对 uncoupled_response：Area +0.023、Recall -0.316 个百分点。

单种子筛查未通过，不扩大重复。
计数界、额度支出和相对发现数界均按实际前缀核对；额度规则不保证严格增益，新增物理仿真为 0。
