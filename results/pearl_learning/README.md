# PEARL 最终实验结果

本目录仅保留可复核的最终实验材料，不保留日期或版本后缀：选定的 PEARL checkpoint、冻结 taskbook/casebook 和报告性汇总指标。逐 episode 轨迹、重复 checkpoint、历史测试输出和单独 SAC 策略文件已移除。

## 保留内容

- `key_results.json`：本文档列出的全部报告性数值及比较协议。
- `models/selected_pearl/`：选定 checkpoint `best_model.pt`、其 manifest 与训练时解析配置。
- `taskbooks/` 与 `casebooks/`：复现 support/query 划分所需的冻结任务与场景；taskbook 哈希为 `d28933ade2878c82bcd228883565394f8f8f7ae86147929c119fee1ca1be966a`。

checkpoint 的 manifest、任务输入和配置共同约束一次评估。原始训练配置即使包含当前已不再使用的字段，也作为历史 checkpoint 的精确 provenance 保留，不应被修改。

## 评估协议

测试对象为 4 个未见的 `meta_test_logical` 任务；每个任务以 20 个冻结且独立的 query case 评估。K 是用于后验推断的 support episode 数。适应阶段只更新潜变量后验，不执行梯度计算、网络参数更新或重新训练。

主指标为四任务平均 `valid_critical_strict_rate`。一个 query episode 必须出现目标对 adversary–SUT 的 critical 事件（物理目标接触或低 TTC），且没有非目标碰撞、出界或错误路线，才计入成功。因此该指标并非“发生碰撞的场景比例”。

## PEARL few-shot 结果

| Support episodes K | 严格有效关键率 |
| ---: | ---: |
| 0 | 0.588 |
| 1 | 0.575 |
| 2 | 0.588 |
| 5 | **0.600** |
| 10 | 0.575 |

在 K=5 时，平均需要 57 个新任务环境步，PEARL 在不更新模型参数的条件下达到 0.600。去除拓扑输入的 K=5 消融为 0.575，低于选定模型的 0.600。

## 相同新任务交互预算比较

下表将 online scratch SAC 与 pooled fine-tune SAC 的每任务训练步数匹配为 PEARL 前 K 个 support episode 的实际累计环境步数。两个 SAC 在线基线各使用 3 个独立随机种子，报告均值 ± 标准差；它们每个环境步都执行一次梯度更新，计算上对在线 SAC 更有利。

| K | 平均交互步数 | PEARL（无梯度） | scratch SAC | pooled fine-tune SAC |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11.0 | **0.575** | 0.317 ± 0.204 | 0.267 ± 0.208 |
| 2 | 19.0 | **0.588** | 0.079 ± 0.044 | 0.275 ± 0.099 |
| 5 | 57.0 | **0.600** | 0.171 ± 0.115 | 0.325 ± 0.094 |
| 10 | 119.5 | **0.575** | 0.196 ± 0.059 | 0.217 ± 0.072 |

这些结果支持的结论是：在这个冻结任务集和相同的新任务交互预算下，PEARL 的无梯度少样本适应表现更好。它们不表示 PEARL 在任意训练预算下都优于专门重训的 SAC。

作为非等预算参考，给每个新任务 5,000 个环境步后，scratch SAC、pooled fine-tune SAC、topology-conditioned pooled SAC 分别为 0.587、0.562、0.525。该组结果只描述较长在线训练后的参考水平，不能用于证明 PEARL 的样本效率优势。

完整机器可读数值和协议以 [`key_results.json`](key_results.json) 为准；代码和重新运行方式见 [`../../pearl_learning/README.md`](../../pearl_learning/README.md)。
