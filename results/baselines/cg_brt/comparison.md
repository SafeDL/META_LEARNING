# CG-BRT统一基准结果

高风险标签严格定义为实测R>0.5。CG-BRT只接收已请求的R，C不进入选择器。全表使用同一六SUT、两个2048点场景池和预算200；十次初始化计入预算。历史信息与反馈通道按方法实际记录，各方法初始化规则并不全部相同。

评价比较200次测试的高风险发现数H200和覆盖率Recall；覆盖率先逐轨迹计算，再宏平均。方法全称与反馈条件见[方法索引](../../../methods/README.md)。

## 五种子完整比较

| 方法 | 历史信息 | 推断反馈 | H200 | 覆盖率 % |
|---|---|---|---:|---:|
| HGT | six historical SUTs | risk R | 141.833 | 72.428 |
| CG-CBM-FD | six historical SUTs | paired risk R + collision C | 141.550 | 72.257 |
| CG-BRT | six historical SUTs | R only | 141.250 | 72.695 |
| HR-GP | six historical SUTs | risk R | 137.933 | 71.854 |
| Matched Collision GP | six sources + 48 public IDM/FVDM configs | risk R | 133.600 | 68.967 |
| RAS-FRT-UQ | six historical SUTs | collision C | 133.067 | 68.008 |
| RF-BO | none | risk R | 126.067 | 64.662 |
| GP-EI | none | risk R | 125.350 | 60.652 |
| Risk-Conditioned Response GP | six sources + 48 public IDM/FVDM configs | risk R | 115.617 | 62.802 |
| Behavior Posterior | six sources + 48 public IDM/FVDM configs | risk R | 115.250 | 63.403 |
| BOP-Elites | none | risk R | 98.150 | 42.443 |
| Historical kNN Ranking | six historical SUTs | none used (fixed ranking) | 86.083 | 43.942 |
| GP-UCB | none | risk R | 58.933 | 23.901 |
| Farthest-First | none | none used | 32.517 | 12.198 |
| Uniform Random | none | none used | 30.250 | 10.383 |
| BAS | none | risk R | 28.950 | 10.157 |

## 种子11补充比较

| 方法 | 历史信息 | 推断反馈 | H200 | 覆盖率 % |
|---|---|---|---:|---:|
| Student-t Source Mixture | six historical SUTs | R | 142.333 | 72.205 |
| Gaussian Source Mixture | six historical SUTs | R | 142.167 | 72.137 |
| Student-t Fixed Mean | six historical SUTs | R | 142.000 | 72.520 |
| Fixed-Mean GP | six historical SUTs | R | 141.917 | 72.485 |
| Protected Student-t Mixture | six historical SUTs | R+C | 141.417 | 72.242 |
| CG-BRT | six historical SUTs | R only | 141.250 | 72.773 |
| Student-t Mixture without Protection | six historical SUTs | R+C | 140.917 | 72.303 |
| Protected Event-Sign Mixture | six historical SUTs | R+C | 140.833 | 72.022 |
| Gaussian Mixture without Protection | six historical SUTs | R+C | 139.750 | 71.540 |

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

| 方法 | 碰撞发现数 F200 | 碰撞覆盖率 % |
|---|---:|---:|
| CG-CBM-FD | 120.583 | 77.509 |
| HGT | 119.200 | 77.062 |
| HR-GP | 116.900 | 76.661 |
| CG-BRT | 116.733 | 76.661 |
| Matched Collision GP | 115.400 | 75.686 |
| RAS-FRT-UQ | 113.650 | 74.062 |
| GP-EI | 110.717 | 68.795 |
| RF-BO | 104.850 | 71.275 |
| Risk-Conditioned Response GP | 98.083 | 67.804 |
| Behavior Posterior | 96.100 | 66.920 |
| BOP-Elites | 79.783 | 43.589 |
| Historical kNN Ranking | 75.583 | 50.192 |
| GP-UCB | 49.050 | 25.422 |
| Farthest-First | 25.550 | 11.774 |
| Uniform Random | 23.917 | 10.247 |
| BAS | 20.550 | 9.278 |

[comparison.json](comparison.json)包含全部轨迹指标及六SUT块的配对检验。只报告现有基准证据，排名与显著性分别判断；没有新增基线轨迹或物理测试。

```powershell
conda run -n metadrive python -B -m methods.lrt.cg_brt.evaluation.compare
```
