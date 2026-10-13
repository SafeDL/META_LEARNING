# CG-BRT统一基准结果

高风险标签严格定义为实测R>0.5。CG-BRT只接收已请求的R，C不进入选择器。全表使用同一六SUT、两个2048点场景池和预算200；十次初始化计入预算。历史信息与反馈通道按方法实际记录，各方法初始化规则并不全部相同。

百分比为逐轨迹归一化后宏平均，H50/H100/H200为高风险发现数。新方法与冻结碰撞方法分开标记，后者只作为已有对照。

## 五种子完整比较

| 方法 | 历史信息 | 推断反馈 | 高风险Area % | 高风险Recall % | H50 | H100 | H200 |
|---|---|---|---:|---:|---:|---:|---:|
| CG-BRT（当前） | six historical SUTs | R only | 46.170 | 72.695 | 45.650 | 89.317 | 141.250 |
| CG-CBM-FD（冻结碰撞方法） | six historical SUTs | paired risk R + collision C | 46.076 | 72.257 | 46.183 | 89.217 | 141.550 |
| frozen_original | six historical SUTs | risk R | 46.003 | 72.428 | 45.817 | 89.100 | 141.833 |
| previous_best | six historical SUTs | risk R | 45.266 | 71.854 | 46.983 | 83.150 | 137.933 |
| RAS-FRT-UQ | six historical SUTs | collision C | 44.439 | 68.008 | 44.883 | 85.600 | 133.067 |
| matched_collision_gp | six sources + 48 public IDM/FVDM configs | risk R | 44.052 | 68.967 | 47.283 | 81.600 | 133.600 |
| risk_conditioned | six sources + 48 public IDM/FVDM configs | risk R | 42.798 | 62.802 | 43.900 | 82.083 | 115.617 |
| behavior_posterior | six sources + 48 public IDM/FVDM configs | risk R | 42.778 | 63.403 | 42.150 | 81.933 | 115.250 |
| RF-BO | none | risk R | 36.313 | 64.662 | 30.083 | 69.883 | 126.067 |
| GP-EI | none | risk R | 35.796 | 60.652 | 35.417 | 74.867 | 125.350 |
| knn_history | six historical SUTs | none used (fixed ranking) | 30.026 | 43.942 | 35.583 | 58.167 | 86.083 |
| BOP-Elites | none | risk R | 27.018 | 42.443 | 33.767 | 62.450 | 98.150 |
| GP-UCB | none | risk R | 13.391 | 23.901 | 20.617 | 34.283 | 58.933 |
| farthest_first | none | none used | 6.363 | 12.198 | 8.383 | 16.367 | 32.517 |
| uniform_random | none | none used | 5.378 | 10.383 | 7.933 | 15.167 | 30.250 |
| BAS | none | risk R | 4.758 | 10.157 | 7.167 | 14.333 | 28.950 |

## 种子11补充比较

| 方法 | 历史信息 | 推断反馈 | 高风险Area % | 高风险Recall % | H50 | H100 | H200 |
|---|---|---|---:|---:|---:|---:|---:|
| CG-BRT（当前） | six historical SUTs | R only | 46.162 | 72.773 | 45.917 | 89.333 | 141.250 |
| student_protected | six historical SUTs | R+C | 46.130 | 72.242 | 46.250 | 89.750 | 141.417 |
| Student-t来源混合 | six historical SUTs | R | 46.115 | 72.205 | 46.000 | 90.333 | 142.333 |
| student_unprotected | six historical SUTs | R+C | 46.069 | 72.303 | 46.500 | 89.667 | 140.917 |
| Student-t固定平均 | six historical SUTs | R | 46.062 | 72.520 | 45.917 | 89.333 | 142.000 |
| 固定平均GP | six historical SUTs | R | 46.037 | 72.485 | 45.833 | 89.083 | 141.917 |
| 高斯来源混合 | six historical SUTs | R | 46.027 | 72.137 | 46.000 | 90.500 | 142.167 |
| gaussian_unprotected | six historical SUTs | R+C | 45.981 | 71.540 | 46.500 | 89.667 | 139.750 |
| event_signs_protected | six historical SUTs | R+C | 45.957 | 72.022 | 45.667 | 89.250 | 140.833 |

额外48个IDM/FVDM配置的工程参照会在历史信息列明确披露。补充表仅种子11；风险混合控制有的将C写入评价记录，但推断只用R。各实现并非只改变一个组件，因此不能把补充表当作单因素消融。

## 当前方法的预算覆盖上限

| SUT | 两池高风险总数 | H200 | 实际覆盖率 % | 预算理论上限 % | 固定初始化后上限 % |
|---|---|---:|---:|---:|---:|
| idm_ref | 910/915 | 191.200 | 20.951 | 21.918 | 21.863 |
| fvdm_target | 144/140 | 133.900 | 94.282 | 100.000 | 100.000 |
| mobil_ref_v2 | 93/92 | 92.500 | 100.000 | 100.000 | 100.000 |
| vi_ttc_ref_audit_v4 | 400/399 | 189.200 | 47.359 | 50.063 | 49.062 |
| mcts_cv_ref_audit_v4 | 180/178 | 131.700 | 73.577 | 100.000 | 100.000 |
| ppo_ref_v2 | 110/108 | 109.000 | 100.000 | 100.000 | 100.000 |

覆盖率为H200/N_H，理论上限为min(200,N_H)/N_H；固定初始化后为min(N_H,190+H10)/N_H。这些是理想计数上限，未附加决策规则限制。

## 辅助碰撞评价（同一五种子轨迹）

| 方法 | 碰撞Area % | 碰撞Recall % | F50 | F100 | F200 |
|---|---:|---:|---:|---:|---:|
| CG-CBM-FD（冻结碰撞方法） | 53.173 | 77.509 | 44.033 | 80.200 | 120.583 |
| previous_best | 52.984 | 76.661 | 45.833 | 76.183 | 116.900 |
| frozen_original | 52.858 | 77.062 | 43.833 | 79.733 | 119.200 |
| CG-BRT（当前） | 52.518 | 76.661 | 43.017 | 78.767 | 116.733 |
| matched_collision_gp | 52.424 | 75.686 | 46.383 | 73.533 | 115.400 |
| RAS-FRT-UQ | 52.203 | 74.062 | 43.400 | 76.800 | 113.650 |
| risk_conditioned | 50.367 | 67.804 | 43.150 | 73.500 | 98.083 |
| behavior_posterior | 49.735 | 66.920 | 41.367 | 72.683 | 96.100 |
| GP-EI | 41.822 | 68.795 | 31.883 | 67.383 | 110.717 |
| RF-BO | 40.695 | 71.275 | 25.483 | 59.367 | 104.850 |
| knn_history | 36.580 | 50.192 | 33.333 | 52.833 | 75.583 |
| BOP-Elites | 29.371 | 43.589 | 30.050 | 53.733 | 79.783 |
| GP-UCB | 14.479 | 25.422 | 17.750 | 28.683 | 49.050 |
| farthest_first | 6.167 | 11.774 | 6.500 | 13.100 | 25.550 |
| uniform_random | 5.331 | 10.247 | 6.250 | 12.067 | 23.917 |
| BAS | 4.385 | 9.278 | 5.200 | 10.367 | 20.550 |

[comparison.json](comparison.json)包含全部轨迹指标及六SUT块的配对检验。只报告现有基准证据，排名与显著性分别判断；没有新增基线轨迹或物理测试。

```powershell
conda run -n metadrive python -B -m experiments.concordance_gated_bayesian_risk_transfer.compare
```
