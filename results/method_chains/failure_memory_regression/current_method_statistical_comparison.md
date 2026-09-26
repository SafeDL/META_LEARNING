# 当前方法统计对比：FBRT-Memory-Exploit-v3

统计对象是当前算法候选 `FBRT-Memory-Exploit-v3`，对照 `FBRT-Memory`、`FBRT-NoMemory`、`HistoryRank-UCB-v2`、`FailureDistance-v2` 和 `Random`。主结果取完整冻结库 `repair_exploit_v3_fullbank/summary_by_task.csv`，仅计回归任务、同一预算和十次配对选择器重复。这里的“发现数”是查询到的有效回归次数；同一物理候选可在不同重复中被重复计数，不能解释为独立故障数。

## 查询预算结果

以下为 11 个回归任务、每任务十次选择器重复的累计发现数。十次重复只改变选择器随机序列，不增加物理样本。

| 预算 | Exploit-v3 | Memory | NoMemory | HistoryRank-UCB | FailureDistance | Random |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 31 | 31 | 21 | 48 | 51 | 31 |
| 5 | 272 | 272 | 184 | 280 | 266 | 131 |
| 10 | 586 | 543 | 406 | 590 | 488 | 258 |
| 20 | 1106 | 1056 | 900 | 1022 | 942 | 510 |

预算 20 时，Exploit-v3 比 Memory 多 50 次（+4.7%），比 NoMemory 多 206 次（+22.9%），比 HistoryRank-UCB 多 84 次（+8.2%），比 FailureDistance 多 164 次（+17.4%），比 Random 多 596 次（+116.9%）。早期预算优势不明显：预算 1 和 5 时 v3 与 Memory 持平，预算 10 时低于 HistoryRank-UCB 4 次；优势主要体现在预算 20。

这 11 个任务中有 10 个目标故障池非空，共 256 个不同故障候选；另一个 PPO 目标池为空。汇总累计数受任务故障率、候选池大小及重复次数影响，不能直接作为显著性检验样本量。

## 按独立物理种子簇比较

三个 legacy 物理种子簇各含三个相关回归任务。先对每个簇内三个任务的十次配对选择器结果求平均，再比较 v3 与各基线在预算 20 的差值。精确双侧符号翻转检验以**物理种子簇**为单位；不把 90 个任务×重复单元当成独立样本。

| 对照基线 | 种子 4179801 | 种子 4179802 | 种子 4179803 | 三簇平均差 | 精确双侧 p |
|---|---:|---:|---:|---:|---:|
| Memory | +0.9 | +2.1 | +1.6 | +1.53 | 0.25 |
| NoMemory | +6.3 | +5.4 | +8.5 | +6.73 | 0.25 |
| HistoryRank-UCB | +1.7 | 0.0 | +2.2 | +1.30 | 0.50 |
| FailureDistance | +4.8 | +4.0 | +5.9 | +4.90 | 0.25 |
| Random | +15.5 | +18.2 | +17.0 | +16.90 | 0.25 |

v3 对 Memory、NoMemory、FailureDistance 和 Random 在三个簇中均为正，但三个簇的最小双侧精确 p 值是 0.25；对 HistoryRank-UCB 有一个簇持平，p=0.50。样本簇太少，不能据此宣称统计显著。

在留一物理种子评估中，每折只用另外两个簇选覆盖查询数，三个训练折都选择 0 个固定覆盖查询。留出结果合计为 v3 989、Memory 943、NoMemory 787；v3 对 Memory 多 46 次（+4.9%），对 NoMemory 多 202 次（+25.7%）。三簇差值分别为 `+9/+21/+16` 和 `+63/+54/+85`，两组精确双侧检验均为 `p=0.25`。这是有用的交叉验证方向性证据，但仍只有三个独立簇。

若把一个 compact MOBIL 上下文作为第四个簇加入，v3 对 Memory 的平均增量为 `+0.4`；即使把这四个上下文簇都视为独立单位，双侧精确检验也只有 `p=0.125`。跨算法迁移中唯一非空目标池为 `17/19`，v3 与 Memory、NoMemory 持平。

## age080 交互目标验证

事前冻结的 age080 验证覆盖八个物理种子、1024 个目标回合。参考构建均通过，age080 目标也没有 ego 碰撞，因此八个目标回归池全为零；验证协议中的五种方法在预算 20 下均发现零故障，Memory 对 NoMemory 的精确检验 `p=1.0`。该验证没有检验出方法间差异；其五方法冻结协议也未包含 v3。详见 `interaction_holdout_age080/validation_report.md`。

## 结论

当前结果支持把 Exploit-v3 保留为优于旧 Memory 的**候选版本**：在冻结旧库的预算 20 总数高 4.7%，三个 legacy 簇均未落后；对 NoMemory 的汇总数高 22.9%。但簇级显著性未达到 `p<0.05`，独立 legacy 簇只有三个；age080 的真实交互目标没有失败可供选择器检出。故目前不能称为已实现显著提升，也不能把汇总查询数、选择器重复或零故障并列解释成确认性证据。

复核来源：

- `repair_exploit_v3_fullbank/summary_by_task.csv`、`task_inputs.jsonl`、`manifest.json`
- `memory_exploit_v3/analysis.json`（留一物理种子评估）
- `interaction_holdout_age080/validation_report.md`、`evaluation/analysis.json`
