# 离线算法后续实验

## 目的和边界

age080 独立验证的目标回归池为零，不能用于证明选择器改进。为继续改善算法，只使用仓库里已经冻结的旧测量银行；本报告不包含新物理回合，也不把旧银行的回放称为独立确认。

现有共同随机种子回放在三个 legacy simulator-seed 簇的九个故障目标上给出：Memory 943 次发现、NoMemory 787 次发现，提升 156 次（19.7%）。这证明当前 Memory 在该旧银行上的描述性效果，但三个簇的精确双侧符号翻转检验为 `p=0.25`，未达到预先要求的 `p<0.05`。该效应不能替代新交互任务上的确认。

## KDE 原型

当时的局部核证据原型源码保存在 [kernel_selector.py](source_snapshot/kernel_selector.py)，按场景输入邻域融合历史结果和本轮已观测结果，并对历史来源权重做局部支持度折减。超参数在两个 simulator-seed 簇上选择、第三簇评估，轮换三折；总预算、随机种子配对和第 10/20 次覆盖采样保持一致。

留出回放中，KDE 版本发现 542 次，低于 NoMemory 的 787 次和现有 Memory 的 943 次。三个折均未优于 NoMemory。这个版本不作为主算法。

## 历史权重先验版本

当时的权重原型源码保存在 [weighted_selector.py](../memory_weight_v3/source_snapshot/weighted_selector.py)，保持 v2 RBF 与在线似然更新，只在另外两个 seed 簇上选择初始历史分支权重，再评估留出的簇。三个留出簇选择的权重依次为 0.95、0.95、0.35；汇总发现数为 942，现有 Memory 为 943，NoMemory 为 787。相对 NoMemory 多 155 次（19.7%），与现有 Memory 持平；簇级检验 `p=0.25`。权重调参没有产生可复现的额外提升。

## 覆盖预算版本

随后单独检验了现有策略在第 10 和第 20 次查询强制做 maximin 覆盖的成本。新版本只改变覆盖查询数，其它 RBF 预测、历史/目标双分支和在线权重更新保持不变；每个留出折的覆盖数只由另外两个 seed 簇选择，候选值为 0、1、2、4。

三个折都选择 0 个固定覆盖查询。留出总数为 989 次发现，NoMemory 为 787 次，增加 202 次（25.7%）；现有 Memory 为 943 次，新版本再增加 46 次（4.9%）。三个 simulator-seed 簇相对 NoMemory 的差值分别为 +63、+54、+85，相对现有 Memory 分别为 +9、+21、+16。九个“种子 × 故障构建”任务中，新版本均不低于 Memory，并在其中五个任务有正差；对 NoMemory 则九个任务均有正差。

按九个任务计算的符号翻转检验对 NoMemory 得到 `p=0.0039`，对现有 Memory 得到 `p=0.0625`；但同一种子下三个故障构建共用场景，不能把九个任务当成九个独立物理簇。按冻结的 simulator-seed 聚类，对 NoMemory 和现有 Memory 的双侧精确检验均为 `p=0.25`。该结果是目前最强的算法进展，尚未通过预设的 cluster-level `p<0.05` 门槛，也尚未在有回归池的 IA/IB 目标上验证。

当前实现已合并至 `method_chains/failure_memory_regression/selector.py`；实验时的原始源码保存在 [exploit_selector.py](../repair_exploit_v3_fullbank/source_snapshot/exploit_selector.py)。留出记录及逐折配置位于 `../memory_exploit_v3/analysis.json` 和 `../memory_exploit_v3/heldout_queries.jsonl`。

同一零覆盖策略也在未参与 seed 参数选择的 compact native MOBIL 回归任务上做了离线外部检查：80 个 parent-pass 候选中有 12 个有效回归，十次预算 20 的总发现数为新版本 117、Memory 113、NoMemory 113。新版本四次比 Memory 多发现一个，其余六次持平。它支持新策略并非只在旧 Profiled-IDM 任务上可用，但这个单任务检查没有足够独立样本用于显著性结论。逐次结果见 `../memory_exploit_v3/compact_probe.json`。

## 完整协方差版本

保留 logistic 权重先验中不同 RBF 区域之间的协方差，固定历史权重 0.5 做相同留一验证。新版本总计发现 941 次，现有 Memory 943 次，NoMemory 787 次；三个簇差值相对 NoMemory 为 +54、+27、+73，相对现有 Memory 为 0、-6、+4，cluster-level `p=0.25`。因此完整协方差没有带来提升，不作为主算法。

## 常规回放集成

把零覆盖版本接入 `repair_replay --offline-only` 后，对完整冻结银行完成了 23 个任务的回放，共 13,800 次逻辑查询，没有新增物理回合，输入指纹保持不变。预算 20 下，Exploit-v3/Memory/NoMemory 分别发现 1106/1056/900 个回归，v3 相对 Memory 增加 50（4.7%），相对 NoMemory 增加 206（22.9%）。跨算法唯一有故障池的任务为 17/19，与 Memory 和 NoMemory 持平。

三个 legacy 种子簇的 v3 对 Memory 差值均为正（+0.9、+2.1、+1.6 次均值），对 NoMemory 也均为正（+6.3、+5.4、+8.5）；其 cluster-level 精确检验仍为 `p=0.25`。加入一个 compact 上下文簇也只有四簇，检验为 `p=0.125`。完整比较见 [algorithm_comparison.md](../repair_exploit_v3_fullbank/algorithm_comparison.md)。这是已接入常规离线框架的可复现算法提升候选，但未达到预设的至少六个独立簇和 `p<0.05` 门槛。

## 当前结论

已有证据支持保留 v2 Memory 作为强于 NoMemory 的旧库算法。覆盖预算版本是当前最有希望的改进，旧库留出发现较 v2 多 4.9%，但独立簇级样本数不足，且尚未在新的 IA/IB 回归任务上验证；KDE、初始权重调节和完整协方差均未提升 v2。年龄 0.80 s 目标的 1024 个回合中，状态年龄机制全部实际生效，但目标故障池为零。

下一次显著性确认需要一个独立、有效回归池非零的目标软件构建和新的物理种子簇。之前冻结的 age080 协议明确规定不追加种子、不调整年龄。若没有新的真实目标构建，现有状态下无法完成预注册的显著提升门槛。

可复核产物：`analysis.json` / `heldout_queries.jsonl`（KDE）、`../memory_weight_v3/analysis.json` / `../memory_weight_v3/heldout_queries.jsonl`（权重调参）、`../memory_full_cov_v3/analysis.json`（完整协方差）、`../memory_exploit_v3/analysis.json` / `../memory_exploit_v3/heldout_queries.jsonl` / `../memory_exploit_v3/compact_probe.json`（覆盖预算）、`../interaction_v1/legacy_paired_seed_audit/repair_report.md`（旧库基线）。
