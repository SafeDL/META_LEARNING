# AdaTE highway-env 复现结果分析

## 总结

| 问题 | 结论 | 证据 |
| --- | --- | --- |
| 核心代码是否可运行 | 是 | 机制、物理、概率与归档结构测试通过；真实 highway-env GIF 和轨迹可回放 |
| AdaTE 核心机制是否出现 | 是 | 关键状态恢复、expected-φ TD、gap-UCB、simplex QP、策略混合及独立 IS 均有日志 |
| 是否完成跨目标确认 | 是 | 2 个 source seed × 3 个完全 held-out target，共 6 个完整案例 |
| AdaTE 是否稳定提高估计效率 | 否 | AdaTE-QP 只在 SUT-C 优于等权，且三个目标均未优于 Uniform-QP |
| 是否复现论文原始数值 | 否 | 当前是 Cut-in 适配环境，不是论文 overtaking/NDE 平台与原始 AV 组合 |

因此，准确表述是：**AdaTE 的可移植算法核心已经在本工程环境中复现并完成跨目标
运行，但论文所报告的效率优势没有在当前 Cut-in 配置中稳定复现。**

## A0：响应混合

- 128 个 anchors 产生 768 次 source episode；6 个 held-out SUT、2 种响应、5 个
  对照、固定预算 20，共 1200 次 target reveal。
- 276 次 simplex QP 求解均无回退，最大 simplex 违反量为 `2.22e-16`，说明约束
  求解在数值上可靠。
- vulnerability held-out MSE：Uniform `0.05317`，Sequential `0.06194`；顺序
  AdaTE 高约 16.5%。平均 critical count 为 `19.50` 对 `19.33`，仅多 `0.17`。
- collision-only MSE：Sequential `0.03268`，Uniform `0.03235`，两者基本相当；
  静态 K=1/2/4 在饱和候选池中与 Uniform 相同。

A0 证明了隐藏目标反馈和凸组合更新链可以工作，但没有证明更好的泛化误差或发现
效率。正式数据在 `mixture/`，主要入口为 `method_summary.csv`、
`target_query_trace.csv` 和 `qp_diagnostics.csv`。

## A1：跨目标 DenseRL 与独立估计

协议固定 source 为 SUT-A/B/D，目标为完全 held-out 的 SUT-C/E/F。每个 seed 的
source stage 在任何目标交互前冻结，目标反馈不会回写 source Q 表。

- 2 个 source seed，共 2400 个 source episode、61093 个 source TD transition。
- 6 个 `(seed, target)` 案例，共 1440 个 target adaptation episode。
- 7 个冻结策略在每个案例各执行 192 次独立评价，共 8064 个 IS draw。
- 归档含 6 个真实 highway-env GIF、对应物理轨迹、逐步 proposal probability、
  importance weight、Q 学习诊断和成本台账。

两 seed 的每次运行 RHW 先分别计算，再按目标取平均：

| Held-out target | NDE-φ | Equal mixture | Uniform-QP | AdaTE-QP | AdaTE-QP 解读 |
| --- | ---: | ---: | ---: | ---: | --- |
| SUT-C | 0.177 | 0.318 | 0.201 | 0.276 | 比等权低 13.1%，但比 Uniform-QP 高 37.4% |
| SUT-E | 0.239 | 0.395 | 0.365 | 0.496 | 比等权高 25.5%，比 Uniform-QP 高 35.7% |
| SUT-F | 0.138 | 0.211 | 0.152 | 0.230 | 比等权高 8.8%，比 Uniform-QP 高 51.0% |
| 三目标均值 | 0.185 | 0.308 | 0.239 | 0.334 | 比等权高 8.4%，比 Uniform-QP 高 39.4% |

RHW 越低越好。这里 AdaTE-QP 的三目标平均 RHW 不仅没有下降，反而高于两个混合
对照；因此不能声称样本效率提升。两个 seed 只用于确认结果不是单次偶然，不足以进行
严格的总体显著性推断，也没有被合并成伪造的单一置信区间。

AdaTE-QP 的平均最终系数为：

- SUT-C：`[0.250, 0.178, 0.572]`
- SUT-E：`[0.361, 0.251, 0.388]`
- SUT-F：`[0.269, 0.506, 0.225]`

系数随目标改变，说明 QP 对反馈有响应；但 C/E 在两个 seed 间波动较大，系数变化
本身不是收益证据。

## 为什么没有观察到论文中的稳定收益

1. 当前声明的自然策略 φ 下，参考碰撞率约为 SUT-C `0.391`、SUT-E `0.260`、
   SUT-F `0.513`，事件并不稀有。直接 NDE 的方差已经很低，重要性采样难以产生论文
   稀有事故估计中的加速空间。
2. source Q 只有在三个 surrogate 对同一状态动作联合可观测时才参与混合；已知步比例
   仅为 C `0.137`、E `0.107`、F `0.140`。其余步骤显式回退到 φ，既避免伪造 Q 表，
   也显著限制自适应策略的作用范围。
3. source SUT 与 held-out target 的动态差异会改变高风险动作排序。QP 能改变 α，但有限
   target adaptation 不能保证把策略质量转换为更低方差。
4. 每案例 192 个评价 draw、每目标 2 个 seed 足以审计机制，不足以支持接近论文表格
   精度的强统计结论。

## 正式结果结构

```text
results/highway_replications/adate/
  shared_pool/                      统一三模式、六 SUT 对比
  mixture/                          A0 日志、QP 诊断和语义化图名
  dense/
    case_strategy_summary.csv       六案例逐策略汇总
    cross_seed_summary.csv          按 target/strategy 的跨 seed 汇总
    final_mixture_coefficients.csv  最终 α
    seed_*/source_stage/            冻结 source Q 与关键快照
    seed_*/target_*/                适配、评价、回放及 manifest
  rare_event/                       稀有 passing 校准与筛选
  report.md                         本分析
```

种子值、SUT 名称和 surrogate 索引是实验身份，不是版本号；配置、结果目录和图文件均
使用阶段或内容语义命名。逐项环境差异见
`replications/adate_highway_env/adate/deviations.md`。
