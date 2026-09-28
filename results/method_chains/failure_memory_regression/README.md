# FBRT 结果索引

当前方法统一称为“FBRT 失效记忆后验利用”；冻结协议中的方法 ID 为 `FBRT-Memory-Exploit-v3`。下表按实验用途命名，路径列保留已冻结的数据标识：

| 实验名称 | 冻结路径 | 内容 |
| --- | --- | --- |
| 旧场景响应库 | `standard_aligned/core/` | 参考、候选与目标响应银行，供离线回放读取 |
| 历史失效记忆基线 | `memory_v2/` | 失效档案、MOBIL/PPO 测量银行、原始 Memory 基线及初始状态审计 |
| 覆盖槽留一物理种子簇评测 | `memory_exploit_v3/` | 当前策略的覆盖次数分析与 compact 检查 |
| 六方法冻结银行离线比较 | `repair_exploit_v3_fullbank/` | 完整银行回放、逐任务结果、清单和算法比较 |
| 后车状态滞后 800 ms 交互留出验证 | `interaction_holdout_age080/` | 事前冻结的八种子独立物理验证；目标回归池为零 |
| 六方法统计比较 | `current_method_statistical_comparison.md` | 当前方法与五种对照的统计解释和适用边界 |
| NL-IDM 双向 release chain | [`nl_release/`](nl_release/) | V6.1 的三版本配对库、双向回放与独立开发/确认记录 |
| PPO 权重继承 release chain | [`ppo_release/`](ppo_release/) | 三个真实继承权重检查点、训练账本、验证及首次确认的负面结果 |
| NL/PPO 共同新场景确认 | [`cross_sut_confirmation/`](cross_sut_confirmation/) | 角色门控修订的预冻结协议、完整配对银行与双向对照 |
| 四族 NL/PPO 补充确认 | [`family_confirmation_v1/`](family_confirmation_v1/) | 多区域前沿修订的独立物理上下文确认及同采集规则的坐标消融 |
| 五族开发调查 | [`family_survey/`](family_survey/) | 新模板能力审计与仅供方法开发的配对结果 |
| 有向特征确认 | [`family_confirmation2/`](family_confirmation2/) | 角色门控区保留/关闭有向边界特征的配对确认，未发现优势 |
| 本文核心协议确认 | [`core_confirmation/`](core_confirmation/) | 两条链的完整冻结确认及负面结果；当前主方法未体现一致优越性，见 [`findings.md`](core_confirmation/findings.md) |
| 上下文局部后验 UCB 确认 | [`contextual_confirmation/`](contextual_confirmation/) | NL/PPO 完整配对 bank 已完成；未支持跨任务优越性，见 [`findings.md`](contextual_confirmation/findings.md) |
| 上下文父风险校准确认 | [`offset_calibration_confirmation/`](offset_calibration_confirmation/) | NL/PPO 完整配对 bank 已完成；局部斜率校准未显示一致优势，见 [`findings.md`](offset_calibration_confirmation/findings.md) |
| 上下文启动覆盖与校准 | [`context_bootstrap_confirmation/`](context_bootstrap_confirmation/) | 阶段四完整 bank 显示回归发现提升、改善发现下降；不支持一致双向优势，见 [`findings.md`](context_bootstrap_confirmation/findings.md) |
| Regression-only context bootstrap | [regression_bootstrap_confirmation/](regression_bootstrap_confirmation/) | Stage five completed in NL/PPO; direction-specific gains were not consistent across transitions. See [findings](regression_bootstrap_confirmation/findings.md). |
| Independent context replication | [superiority_replication_confirmation/](superiority_replication_confirmation/) | Complete 3,267-episode NL/PPO banks and frozen replay; primary found 45 changes at D@20 versus 54 for coordinate role-gated search, without statistical evidence of superiority. See [findings](superiority_replication_confirmation/findings.md). |
| Role-gated fresh-family confirmation | [role_gated_generalization_confirmation/](role_gated_generalization_confirmation/) | Candidate frozen across eight executable families and 24 new contexts; measurement paused before evaluation at NL 4,153/8,712 and PPO 3,951/8,712 valid cached episodes. |

当前方法在旧银行预算 20 的发现总数为 1106，原 Memory 为 1056；三个独立旧物理种子簇的精确双侧检验为 `p=0.25`，不能宣称显著提升。后车状态滞后 800 ms 的交互留出验证中，目标回归池为零，也不能作为方法优势证据。详见 [`统计对比`](current_method_statistical_comparison.md) 和 [`独立验证`](interaction_holdout_age080/validation_report.md)。

新增 NL-IDM 双向确认集包含 441 个场景、1323 次完整配对物理执行；两次版本更新均未出现回退。改善数分别为 3 和 8，预算 20 下有向残差方法找到 3 和 6，静态父风险基线找到 3 和 7。它不支持双向优越性主张，详细数据见 [`NL release 确认报告`](nl_release/confirmation/report.md)。

后续三组独立确认记录见 [`NL release 索引`](nl_release/README.md)。第四组冻结银行中，父版本 TTC 余量覆盖与回退前沿搜索结合定向模型，在回退/改善方向分别找到 3/3、13/17，静态风险为 0/3、6/17；但所有变化集中于 S08 一个场景族，故只能称有限银行上的描述性双向增益，不能宣称跨族显著优势或 PPO 权重更新上的效果。前三组负面确认及消融全部保留。

已归档的中间尝试包括：KDE 局部核原型（留出发现 542，低于原 Memory 的 943）、历史权重调节（942）、完整协方差（941）；三者均未优于当前方案。早期交互开发银行中，后车状态滞后 300 ms 的目标没有形成有效回归；第二轮调整后的五个构建也全部没有有效 ego 碰撞，因此它们不承担当前结论。原始文件及旧源码快照位于 [`archives/experimental_results/`](../../../archives/experimental_results/README.md)，不在当前结果中重复展示。`memory_v2/validation_audit_v3/` 是独立物理诊断，仍保留以支持修正审计的对照。
