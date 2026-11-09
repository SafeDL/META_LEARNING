# 候选规模实验完整结果

前车制动场景；八个目标 ADS、八种方法，候选数 1,000／2,000／4,000；每任务共同初始化 20 次、总预算 200 次，算法种子 11、环境种子 81001，候选 replicate=30。三种规模严格嵌套，各任务独立使用目标反馈。

真实失效来自实际碰撞事件，按唯一候选 ID 计数。覆盖率分母取对应完整候选库的真实失效数。零失效库的覆盖率、少于两个发现的参数距离以“—”表示；平均值只使用可定义任务并登记任务数。参数距离使用声明范围归一化的平均 L1 距离及类别不一致项。

下表的发现总数、等任务平均发现 AUC、覆盖率及距离分别报告；相同任务内比较方法，跨规模结合预算比例和真实失效规模解释。单种子结果不计算跨种子置信区间。

## 1,000 个候选

每任务 B/N=20.0%；八个候选库累计 50 个真实失效，整体失效率 0.6250%；5/8 个库无失效，召回率在3/8个任务中可定义。

| 方法 | 发现总数 | 初始化后新增总数 | 平均发现 AUC | 平均召回 AUC | 平均覆盖率 | 汇总召回 |
|---|---:|---:|---:|---:|---:|---:|
| History transfer + joint | 49 | 48 | 4.3256 | 0.3260 (3/8) | 66.67% (3/8) | 98.00% |
| Random | 8 | 7 | 0.2612 | 0.0148 (3/8) | 5.67% (3/8) | 16.00% |
| RF-EI | 48 | 47 | 4.1131 | 0.4568 (3/8) | 65.96% (3/8) | 96.00% |
| SAMOTA | 49 | 48 | 3.2875 | 0.4012 (3/8) | 66.67% (3/8) | 98.00% |
| ScenarioFuzz | 47 | 46 | 3.9475 | 0.3545 (3/8) | 65.96% (3/8) | 94.00% |
| Paired-system | 17 | 16 | 1.3587 | 0.0771 (3/8) | 12.06% (3/8) | 34.00% |
| Indago (history) | 10 | 9 | 0.8481 | 0.0481 (3/8) | 7.09% (3/8) | 20.00% |
| BehAVExplor | 3 | 2 | 0.1550 | 0.0088 (3/8) | 2.13% (3/8) | 6.00% |

| 方法 | 同预算两两距离（任务数） | 同预算最近邻距离（任务数） | 同发现数两两距离（任务数） | 同发现数最近邻距离（任务数） |
|---|---:|---:|---:|---:|
| History transfer + joint | 0.1897 (2/8) | 0.1209 (2/8) | 0.1517 (1/8) | 0.1303 (1/8) |
| Random | 0.2733 (1/8) | 0.2114 (1/8) | 0.2952 (1/8) | 0.2705 (1/8) |
| RF-EI | 0.1892 (2/8) | 0.1209 (2/8) | 0.2318 (1/8) | 0.1942 (1/8) |
| SAMOTA | 0.1897 (2/8) | 0.1209 (2/8) | 0.1582 (1/8) | 0.1470 (1/8) |
| ScenarioFuzz | 0.2510 (1/8) | 0.1155 (1/8) | 0.1839 (1/8) | 0.1585 (1/8) |
| Paired-system | 0.2964 (1/8) | 0.1597 (1/8) | 0.3633 (1/8) | 0.2633 (1/8) |
| Indago (history) | 0.2313 (1/8) | 0.1437 (1/8) | 0.2088 (1/8) | 0.1822 (1/8) |
| BehAVExplor | 0.2489 (1/8) | 0.2191 (1/8) | 0.2489 (1/8) | 0.2191 (1/8) |

共同失效数按每个 ADS 内全部八种方法的最小发现数确定，按发现顺序截取；k<2时不计算距离。

| 目标 ADS | 真值失效数 | 共同 k | History transfer + joint | Random | RF-EI | SAMOTA | ScenarioFuzz | Paired-system | Indago (history) | BehAVExplor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| independent_idm_mobil | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| independent_fvdm_lane | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| independent_linear_mpc | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| independent_nonlinear_trajectory | 2 | 0 | 2 | 0 | 2 | 2 | 0 | 0 | 0 | 0 |
| independent_uct | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| independent_beam | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| independent_ppo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| independent_dqn | 47 | 3 | 47 | 8 | 46 | 47 | 46 | 17 | 10 | 3 |

| 目标 ADS | 方法 | 发现数 | 初始后新增 | 覆盖率 | 发现 AUC | 首次失效位置 | 两两距离 | 最近邻距离 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| independent_idm_mobil | History transfer + joint | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | Random | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | RF-EI | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | SAMOTA | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | ScenarioFuzz | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | Paired-system | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | Indago (history) | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | BehAVExplor | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | History transfer + joint | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | Random | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | RF-EI | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | SAMOTA | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | ScenarioFuzz | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | Paired-system | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | Indago (history) | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | BehAVExplor | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | History transfer + joint | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | Random | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | RF-EI | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | SAMOTA | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | ScenarioFuzz | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | Paired-system | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | Indago (history) | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | BehAVExplor | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_nonlinear_trajectory | History transfer + joint | 2 | 2 | 100.00% | 0.5050 | 149 | 0.1264 | 0.1264 |
| independent_nonlinear_trajectory | Random | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_nonlinear_trajectory | RF-EI | 2 | 2 | 100.00% | 1.4000 | 58 | 0.1264 | 0.1264 |
| independent_nonlinear_trajectory | SAMOTA | 2 | 2 | 100.00% | 1.3450 | 66 | 0.1264 | 0.1264 |
| independent_nonlinear_trajectory | ScenarioFuzz | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_nonlinear_trajectory | Paired-system | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_nonlinear_trajectory | Indago (history) | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_nonlinear_trajectory | BehAVExplor | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | History transfer + joint | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | Random | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | RF-EI | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | SAMOTA | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | ScenarioFuzz | 1 | 1 | 100.00% | 0.4000 | 121 | — | — |
| independent_uct | Paired-system | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | Indago (history) | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | BehAVExplor | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_beam | History transfer + joint | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | Random | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | RF-EI | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | SAMOTA | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | ScenarioFuzz | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | Paired-system | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | Indago (history) | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | BehAVExplor | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | History transfer + joint | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | Random | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | RF-EI | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | SAMOTA | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | ScenarioFuzz | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | Paired-system | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | Indago (history) | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | BehAVExplor | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_dqn | History transfer + joint | 47 | 46 | 100.00% | 34.1000 | 5 | 0.2530 | 0.1154 |
| independent_dqn | Random | 8 | 7 | 17.02% | 2.0900 | 5 | 0.2733 | 0.2114 |
| independent_dqn | RF-EI | 46 | 45 | 97.87% | 31.5050 | 5 | 0.2520 | 0.1154 |
| independent_dqn | SAMOTA | 47 | 46 | 100.00% | 24.9550 | 5 | 0.2530 | 0.1154 |
| independent_dqn | ScenarioFuzz | 46 | 45 | 97.87% | 31.1800 | 5 | 0.2510 | 0.1155 |
| independent_dqn | Paired-system | 17 | 16 | 36.17% | 10.8700 | 5 | 0.2964 | 0.1597 |
| independent_dqn | Indago (history) | 10 | 9 | 21.28% | 6.7850 | 5 | 0.2313 | 0.1437 |
| independent_dqn | BehAVExplor | 3 | 2 | 6.38% | 1.2400 | 5 | 0.2489 | 0.2191 |

| 方法 | 相对 Random 发现数之差 | 平均发现 AUC 之差 | 数量胜／负／平（8 ADS） |
|---|---:|---:|---:|
| History transfer + joint | +41 | +4.0644 | 2/0/6 |
| Random | +0 | +0.0000 | 0/0/8 |
| RF-EI | +40 | +3.8519 | 2/0/6 |
| SAMOTA | +41 | +3.0262 | 2/0/6 |
| ScenarioFuzz | +39 | +3.6862 | 2/0/6 |
| Paired-system | +9 | +1.0975 | 1/0/7 |
| Indago (history) | +2 | +0.5869 | 1/0/7 |
| BehAVExplor | -5 | -0.1062 | 0/1/7 |

## 2,000 个候选

每任务 B/N=10.0%；八个候选库累计 99 个真实失效，整体失效率 0.6188%；5/8 个库无失效，召回率在3/8个任务中可定义。

| 方法 | 发现总数 | 初始化后新增总数 | 平均发现 AUC | 平均召回 AUC | 平均覆盖率 | 汇总召回 |
|---|---:|---:|---:|---:|---:|---:|
| History transfer + joint | 93 | 92 | 6.7181 | 0.1866 (3/8) | 32.29% (3/8) | 93.94% |
| Random | 8 | 7 | 0.4419 | 0.0588 (3/8) | 19.10% (3/8) | 8.08% |
| RF-EI | 89 | 88 | 5.8087 | 0.3017 (3/8) | 47.22% (3/8) | 89.90% |
| SAMOTA | 71 | 70 | 3.9100 | 0.1788 (3/8) | 57.29% (3/8) | 71.72% |
| ScenarioFuzz | 85 | 84 | 5.2175 | 0.1928 (3/8) | 62.50% (3/8) | 85.86% |
| Paired-system | 21 | 20 | 1.4475 | 0.0402 (3/8) | 7.29% (3/8) | 21.21% |
| Indago (history) | 9 | 8 | 0.5544 | 0.0154 (3/8) | 3.12% (3/8) | 9.09% |
| BehAVExplor | 5 | 4 | 0.2731 | 0.0076 (3/8) | 1.74% (3/8) | 5.05% |

| 方法 | 同预算两两距离（任务数） | 同预算最近邻距离（任务数） | 同发现数两两距离（任务数） | 同发现数最近邻距离（任务数） |
|---|---:|---:|---:|---:|
| History transfer + joint | 0.2452 (1/8) | 0.0983 (1/8) | 0.1492 (1/8) | 0.1134 (1/8) |
| Random | 0.2117 (1/8) | 0.1495 (1/8) | 0.2180 (1/8) | 0.1591 (1/8) |
| RF-EI | 0.2497 (1/8) | 0.1009 (1/8) | 0.1922 (1/8) | 0.1321 (1/8) |
| SAMOTA | 0.1808 (2/8) | 0.1139 (2/8) | 0.1368 (1/8) | 0.0999 (1/8) |
| ScenarioFuzz | 0.2451 (1/8) | 0.0991 (1/8) | 0.2164 (1/8) | 0.1448 (1/8) |
| Paired-system | 0.2793 (1/8) | 0.1434 (1/8) | 0.2562 (1/8) | 0.1669 (1/8) |
| Indago (history) | 0.2287 (1/8) | 0.1391 (1/8) | 0.2642 (1/8) | 0.1651 (1/8) |
| BehAVExplor | 0.2023 (1/8) | 0.1245 (1/8) | 0.2023 (1/8) | 0.1245 (1/8) |

共同失效数按每个 ADS 内全部八种方法的最小发现数确定，按发现顺序截取；k<2时不计算距离。

| 目标 ADS | 真值失效数 | 共同 k | History transfer + joint | Random | RF-EI | SAMOTA | ScenarioFuzz | Paired-system | Indago (history) | BehAVExplor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| independent_idm_mobil | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| independent_fvdm_lane | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| independent_linear_mpc | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| independent_nonlinear_trajectory | 2 | 0 | 0 | 1 | 1 | 2 | 0 | 0 | 0 | 0 |
| independent_uct | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| independent_beam | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| independent_ppo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| independent_dqn | 96 | 5 | 93 | 7 | 88 | 69 | 84 | 21 | 9 | 5 |

| 目标 ADS | 方法 | 发现数 | 初始后新增 | 覆盖率 | 发现 AUC | 首次失效位置 | 两两距离 | 最近邻距离 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| independent_idm_mobil | History transfer + joint | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | Random | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | RF-EI | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | SAMOTA | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | ScenarioFuzz | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | Paired-system | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | Indago (history) | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | BehAVExplor | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | History transfer + joint | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | Random | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | RF-EI | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | SAMOTA | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | ScenarioFuzz | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | Paired-system | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | Indago (history) | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | BehAVExplor | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | History transfer + joint | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | Random | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | RF-EI | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | SAMOTA | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | ScenarioFuzz | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | Paired-system | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | Indago (history) | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | BehAVExplor | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_nonlinear_trajectory | History transfer + joint | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_nonlinear_trajectory | Random | 1 | 1 | 50.00% | 0.2850 | 144 | — | — |
| independent_nonlinear_trajectory | RF-EI | 1 | 1 | 50.00% | 0.8600 | 29 | — | — |
| independent_nonlinear_trajectory | SAMOTA | 2 | 2 | 100.00% | 0.4300 | 157 | 0.1264 | 0.1264 |
| independent_nonlinear_trajectory | ScenarioFuzz | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_nonlinear_trajectory | Paired-system | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_nonlinear_trajectory | Indago (history) | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_nonlinear_trajectory | BehAVExplor | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | History transfer + joint | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | Random | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | RF-EI | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | SAMOTA | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | ScenarioFuzz | 1 | 1 | 100.00% | 0.1450 | 172 | — | — |
| independent_uct | Paired-system | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | Indago (history) | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | BehAVExplor | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_beam | History transfer + joint | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | Random | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | RF-EI | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | SAMOTA | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | ScenarioFuzz | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | Paired-system | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | Indago (history) | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | BehAVExplor | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | History transfer + joint | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | Random | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | RF-EI | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | SAMOTA | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | ScenarioFuzz | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | Paired-system | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | Indago (history) | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | BehAVExplor | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_dqn | History transfer + joint | 93 | 92 | 96.88% | 53.7450 | 5 | 0.2452 | 0.0983 |
| independent_dqn | Random | 7 | 6 | 7.29% | 3.2500 | 5 | 0.2117 | 0.1495 |
| independent_dqn | RF-EI | 88 | 87 | 91.67% | 45.6100 | 5 | 0.2497 | 0.1009 |
| independent_dqn | SAMOTA | 69 | 68 | 71.88% | 30.8500 | 5 | 0.2352 | 0.1014 |
| independent_dqn | ScenarioFuzz | 84 | 83 | 87.50% | 41.5950 | 5 | 0.2451 | 0.0991 |
| independent_dqn | Paired-system | 21 | 20 | 21.88% | 11.5800 | 5 | 0.2793 | 0.1434 |
| independent_dqn | Indago (history) | 9 | 8 | 9.38% | 4.4350 | 5 | 0.2287 | 0.1391 |
| independent_dqn | BehAVExplor | 5 | 4 | 5.21% | 2.1850 | 5 | 0.2023 | 0.1245 |

| 方法 | 相对 Random 发现数之差 | 平均发现 AUC 之差 | 数量胜／负／平（8 ADS） |
|---|---:|---:|---:|
| History transfer + joint | +85 | +6.2763 | 1/1/6 |
| Random | +0 | +0.0000 | 0/0/8 |
| RF-EI | +81 | +5.3669 | 1/0/7 |
| SAMOTA | +63 | +3.4681 | 2/0/6 |
| ScenarioFuzz | +77 | +4.7756 | 2/1/5 |
| Paired-system | +13 | +1.0056 | 1/1/6 |
| Indago (history) | +1 | +0.1125 | 1/1/6 |
| BehAVExplor | -3 | -0.1687 | 0/2/6 |

## 4,000 个候选

每任务 B/N=5.0%；八个候选库累计 200 个真实失效，整体失效率 0.6250%；5/8 个库无失效，召回率在3/8个任务中可定义。

| 方法 | 发现总数 | 初始化后新增总数 | 平均发现 AUC | 平均召回 AUC | 平均覆盖率 | 汇总召回 |
|---|---:|---:|---:|---:|---:|---:|
| History transfer + joint | 136 | 135 | 8.3438 | 0.2449 (3/8) | 46.48% (3/8) | 68.00% |
| Random | 9 | 8 | 0.5837 | 0.0396 (3/8) | 8.05% (3/8) | 4.50% |
| RF-EI | 108 | 107 | 6.4831 | 0.1935 (3/8) | 31.64% (3/8) | 54.00% |
| SAMOTA | 72 | 71 | 3.9044 | 0.2572 (3/8) | 44.91% (3/8) | 36.00% |
| ScenarioFuzz | 100 | 99 | 5.5200 | 0.0763 (3/8) | 17.27% (3/8) | 50.00% |
| Paired-system | 19 | 18 | 1.2969 | 0.0179 (3/8) | 3.28% (3/8) | 9.50% |
| Indago (history) | 11 | 10 | 0.5863 | 0.0081 (3/8) | 1.90% (3/8) | 5.50% |
| BehAVExplor | 6 | 5 | 0.3419 | 0.0740 (3/8) | 17.53% (3/8) | 3.00% |

| 方法 | 同预算两两距离（任务数） | 同预算最近邻距离（任务数） | 同发现数两两距离（任务数） | 同发现数最近邻距离（任务数） |
|---|---:|---:|---:|---:|
| History transfer + joint | 0.2376 (1/8) | 0.0829 (1/8) | 0.1974 (1/8) | 0.1067 (1/8) |
| Random | 0.2443 (1/8) | 0.1458 (1/8) | 0.2234 (1/8) | 0.1561 (1/8) |
| RF-EI | 0.2174 (2/8) | 0.1375 (2/8) | 0.1798 (1/8) | 0.1258 (1/8) |
| SAMOTA | 0.1809 (2/8) | 0.0984 (2/8) | 0.1137 (1/8) | 0.0859 (1/8) |
| ScenarioFuzz | 0.2428 (1/8) | 0.0874 (1/8) | 0.1863 (1/8) | 0.1475 (1/8) |
| Paired-system | 0.3096 (1/8) | 0.1447 (1/8) | 0.3281 (1/8) | 0.2578 (1/8) |
| Indago (history) | 0.2443 (1/8) | 0.1450 (1/8) | 0.2463 (1/8) | 0.1785 (1/8) |
| BehAVExplor | 0.2021 (1/8) | 0.1472 (1/8) | 0.2021 (1/8) | 0.1472 (1/8) |

共同失效数按每个 ADS 内全部八种方法的最小发现数确定，按发现顺序截取；k<2时不计算距离。

| 目标 ADS | 真值失效数 | 共同 k | History transfer + joint | Random | RF-EI | SAMOTA | ScenarioFuzz | Paired-system | Indago (history) | BehAVExplor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| independent_idm_mobil | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| independent_fvdm_lane | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| independent_linear_mpc | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| independent_nonlinear_trajectory | 5 | 0 | 1 | 1 | 2 | 5 | 0 | 0 | 0 | 0 |
| independent_uct | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| independent_beam | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| independent_ppo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| independent_dqn | 193 | 5 | 134 | 8 | 106 | 67 | 100 | 19 | 11 | 5 |

| 目标 ADS | 方法 | 发现数 | 初始后新增 | 覆盖率 | 发现 AUC | 首次失效位置 | 两两距离 | 最近邻距离 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| independent_idm_mobil | History transfer + joint | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | Random | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | RF-EI | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | SAMOTA | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | ScenarioFuzz | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | Paired-system | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | Indago (history) | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_idm_mobil | BehAVExplor | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | History transfer + joint | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | Random | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | RF-EI | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | SAMOTA | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | ScenarioFuzz | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | Paired-system | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | Indago (history) | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_fvdm_lane | BehAVExplor | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | History transfer + joint | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | Random | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | RF-EI | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | SAMOTA | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | ScenarioFuzz | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | Paired-system | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | Indago (history) | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_linear_mpc | BehAVExplor | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_nonlinear_trajectory | History transfer + joint | 1 | 1 | 20.00% | 0.2050 | 160 | — | — |
| independent_nonlinear_trajectory | Random | 1 | 1 | 20.00% | 0.4850 | 104 | — | — |
| independent_nonlinear_trajectory | RF-EI | 2 | 2 | 40.00% | 1.6000 | 35 | 0.1912 | 0.1912 |
| independent_nonlinear_trajectory | SAMOTA | 5 | 5 | 100.00% | 3.1300 | 65 | 0.1479 | 0.1117 |
| independent_nonlinear_trajectory | ScenarioFuzz | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_nonlinear_trajectory | Paired-system | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_nonlinear_trajectory | Indago (history) | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_nonlinear_trajectory | BehAVExplor | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | History transfer + joint | 1 | 1 | 50.00% | 0.7050 | 60 | — | — |
| independent_uct | Random | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | RF-EI | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | SAMOTA | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | ScenarioFuzz | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | Paired-system | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | Indago (history) | 0 | 0 | 0.00% | 0.0000 | >200 | — | — |
| independent_uct | BehAVExplor | 1 | 1 | 50.00% | 0.4200 | 117 | — | — |
| independent_beam | History transfer + joint | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | Random | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | RF-EI | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | SAMOTA | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | ScenarioFuzz | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | Paired-system | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | Indago (history) | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_beam | BehAVExplor | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | History transfer + joint | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | Random | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | RF-EI | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | SAMOTA | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | ScenarioFuzz | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | Paired-system | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | Indago (history) | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_ppo | BehAVExplor | 0 | 0 | — | 0.0000 | >200 | — | — |
| independent_dqn | History transfer + joint | 134 | 133 | 69.43% | 65.8400 | 5 | 0.2376 | 0.0829 |
| independent_dqn | Random | 8 | 7 | 4.15% | 4.1850 | 5 | 0.2443 | 0.1458 |
| independent_dqn | RF-EI | 106 | 105 | 54.92% | 50.2650 | 5 | 0.2436 | 0.0837 |
| independent_dqn | SAMOTA | 67 | 66 | 34.72% | 28.1050 | 5 | 0.2140 | 0.0850 |
| independent_dqn | ScenarioFuzz | 100 | 99 | 51.81% | 44.1600 | 5 | 0.2428 | 0.0874 |
| independent_dqn | Paired-system | 19 | 18 | 9.84% | 10.3750 | 5 | 0.3096 | 0.1447 |
| independent_dqn | Indago (history) | 11 | 10 | 5.70% | 4.6900 | 5 | 0.2443 | 0.1450 |
| independent_dqn | BehAVExplor | 5 | 4 | 2.59% | 2.3150 | 5 | 0.2021 | 0.1472 |

| 方法 | 相对 Random 发现数之差 | 平均发现 AUC 之差 | 数量胜／负／平（8 ADS） |
|---|---:|---:|---:|
| History transfer + joint | +127 | +7.7600 | 2/0/6 |
| Random | +0 | +0.0000 | 0/0/8 |
| RF-EI | +99 | +5.8994 | 2/0/6 |
| SAMOTA | +63 | +3.3206 | 2/0/6 |
| ScenarioFuzz | +91 | +4.9362 | 1/1/6 |
| Paired-system | +10 | +0.7131 | 1/1/6 |
| Indago (history) | +2 | +0.0025 | 1/1/6 |
| BehAVExplor | -3 | -0.2419 | 1/2/5 |

## 数据与图片

全部192次运行、38,400次逻辑查询使用同一32,000条最大库执行真值，不再重复计作物理仿真。付费、来源和模型核验分别见 [completed_run_check.json](completed_run_check.json)、[source_admission_check.json](source_admission_check.json)；逐规模报告位于 ../1000/reports/、../2000/reports/、../4000/reports/。

保留的四张跨规模图见 [figure_data.json](figure_data.json)，逐任务时间和内存读取各规模 runs.csv；累计等效测试时间与实际选择器墙钟耗时分开报告。逐规模重复分图和案例 GIF 已清理，零失效任务和不利结果的完整数值记录保留。
