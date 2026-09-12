# Highway-env 快速验证 DIVA-Mine Idea 的最小可信实验设计

## 0. 目标：先证明“这个 idea 值得做”，而不是先做完整论文系统
系统gpu在：
conda activate metadrive

当前阶段只验证一个核心命题：

> **面对一个从未见过的目标 SUT，利用历史 SUT 的测试结果学习一个可迁移的 vulnerability prior，并用极少量 diagnostic tests 识别目标 SUT 的个性化脆弱性后，能否在固定总测试预算内比“不适配”或“随机适配”发现更多危险场景？**

这与现有 Few-Shot Testing 的目标不同。已有 FST 工作主要优化的是少量测试下的**整体性能估计误差**，而这里要验证的是：  
**few-shot target adaptation 是否能提高 failure-oriented scenario mining。**

因此第一阶段**不要训练 Transformer，不要做 Meta-RL，不要上复杂深度网络，不做多功能场景，不做大规模消融**。

---

# 1. 为什么用 highway-env

对于当前这一步，highway-env 更适合做“概念验证”。

原因很简单：

- 场景维度低，适合做 Cut-in / Merge 这类参数化场景；
- IDM 类 SUT 容易构造多个行为不同但结构相关的控制器；
- 仿真轻量，适合一次性生成 source-SUT × scenario response matrix；
- 不需要 GPU；
- 后续算法部分基本都是 NumPy / SciPy 级别计算；
- 可以很快做 leave-one-SUT-out 验证；
- 和当前 DIVA-Mine 的方法假设高度匹配：  
  **“不同 SUT 的漏洞函数具有共享结构，但失效边界存在个体差异。”**

需要强调：

> highway-env 在这里是**快速机制验证环境**，不是最终证明平台。

如果 idea 在 highway-env 都无法成立，就没有必要马上投入 MetaDrive。  
如果在 highway-env 可以稳定成立，再迁移回 MetaDrive 做真实性和复杂性交叉验证。

---

# 2. 第一版实验只用一个功能场景：Cut-in

不要同时做 Merge、Lane Change、Intersection。

第一版只做：

> **前车 / 邻车切入目标车道，目标 SUT 需要减速避免碰撞。**

这样有三个好处：

1. 和现有 META_LEARNING 中的 `cutin` 工作最接近；
2. 和 FST 文献中常用的二维 Cut-in 实验设置一致；
3. failure boundary 容易形成，最适合验证 few-shot adaptation。

---

# 3. 场景参数：先只用 3 个连续变量

为了既简单又保留足够差异，建议场景定义为：

\[
x = [R_0,\ \Delta v,\ T_{cutin}]
\]

其中：

- `R0`：切入完成时目标车与前车间距；
- `Δv`：前车相对目标车速度；
- `T_cutin`：切入动作持续时间 / 切入激进程度。

推荐初始范围：

| 参数 | 建议范围 |
|---|---:|
| `R0` | 5 – 40 m |
| `Δv` | -8 – 2 m/s |
| `T_cutin` | 1.0 – 3.0 s |

如果 highway-env 中精确控制 `T_cutin` 比较麻烦，第一版可以进一步简化成二维：

\[
x=[R_0,\Delta v]
\]

二维版本其实更适合第一轮快速画出 vulnerability map。

---

# 4. SUT 设计：关键不是“保守到激进”，而是制造不同漏洞模式

这是整个实验是否成功的关键。

不要只做：

- cautious
- normal
- aggressive

因为这容易形成“所有风险边界只是平移”的单一结构。

更好的做法是构造 **6 个 source/target controller profiles**，让它们在不同维度上有差异。

建议：

| SUT | 主要特征 |
|---|---|
| SUT-A | 大期望时距，弱制动 |
| SUT-B | 小期望时距，强制动 |
| SUT-C | 中等时距，但反应慢 |
| SUT-D | 中等时距，制动保守 |
| SUT-E | 激进跟驰，但最大减速度较大 |
| SUT-F | 正常 IDM + 人为 reaction delay |

核心思想：

> **不要让不同 SUT 只是“整体更安全/更危险”，而要让它们在不同区域更容易失败。**

这样 diagnostic support 才有意义。

---

# 5. 响应值：训练用连续 vulnerability，评价用真实 failure

不要只保存 crash=0/1。

对每个 `(SUT, scenario)` episode，同时记录：

- collision；
- minimum TTC；
- minimum distance；
- 是否有效完成；
- episode termination；
- scenario parameters。

定义连续 vulnerability response，例如：

\[
y =
\begin{cases}
1.0, & collision\\
0.75 + 0.25 g(TTC_{min}), & near\text{-}miss\\
0.75 g(TTC_{min}), & otherwise
\end{cases}
\]

其中 \(g(\cdot)\) 只需要是单调函数即可，例如：

\[
g(TTC)=\exp(-TTC/3)
\]

第一版不需要追求“最完美”的 vulnerability 公式。

需要遵守一个原则：

> **continuous response 用于学习和后验更新；最终实验成功与否必须看真实 collision / near-miss discovery。**

---

# 6. 数据生成：一次性建立 response matrix

## 6.1 场景锚点

用 Sobol 或 Latin Hypercube 一次性生成：

> **128 个共同场景 anchors**

所有 SUT 都跑同样的 128 个场景。

得到矩阵：

\[
Y \in \mathbb{R}^{M\times128}
\]

其中：

\[
Y_{u,j} = vulnerability(SUT_u, x_j)
\]

推荐：

- SUT 数：6
- Anchors：128
- 总仿真：6 × 128 = **768 episodes**

如果 highway-env 足够快，可以直接做：

- 8 SUT × 256 anchors = 2048 episodes

但第一轮没必要。

---

# 7. 最快的算法版本：不要 GP，先用“离散低秩先验”

第一轮完全不需要连续 GP 插值。

直接对 source-SUT response matrix 做：

1. 每个场景列求 source mean；
2. 中心化；
3. SVD；
4. 取 rank=2。

得到：

\[
f_u(x_j)\approx\mu_j+b_j^Tz_u
\]

其中：

- \(\mu_j\)：第 j 个场景上的公共风险；
- \(b_j\)：该场景的低秩 vulnerability basis；
- \(z_u\)：SUT-specific latent profile。

这样做的优势：

- 没有神经网络；
- 没有 GP 训练；
- 没有 GPU；
- 每次 target adaptation 只需要更新一个 2 维 Gaussian posterior；
- 可以非常快地判断 research assumption 是否成立。

---

# 8. 第一阶段实验：Leave-One-SUT-Out

6 个 SUT 轮流做 target。

例如：

- 训练 prior：A/B/C/D/E
- unseen target：F

然后换：

- 训练：A/B/C/D/F
- target：E

依次完成 6 folds。

**Target SUT 的完整 response row 不能用于构造 prior。**

只允许算法逐步“揭示”被选中的 target scenario outcome。

这一步是可信度的核心。

---

# 9. Support budget：第一轮只用 K=4

不要一开始同时扫：

- K=1
- K=2
- K=4
- K=8
- K=16

第一轮固定：

> **K = 4**

原因：

- 足够体现 few-shot；
- 对 rank=2 latent 已有较合理辨识能力；
- 结果更容易稳定；
- 减少实验数量。

idea 跑通后再补 K=1/2。

---

# 10. 总测试预算：B=20

目标 SUT 总共允许：

\[
B=20
\]

其中：

- 前 4 次：diagnostic support；
- 后 16 次：failure mining。

必须注意：

> **support 4 次也算进总预算。**

不能把 diagnostic cost 免费掉。

---

# 11. 只比较 4 个方法

第一轮只保留最必要的四个方法。

## Method 1：Random

完全随机选择 20 个场景。

作用：

> 没有历史知识的下限。

---

## Method 2：Shared Prior

不用 target support。

直接根据 source mean risk 排序，选择 20 个场景。

作用：

> 检查“历史公共风险知识”本身够不够。

---

## Method 3：Random Support + Adaptation

随机选 4 个 support。

利用这 4 个 target response 更新 latent posterior。

再根据 posterior risk 选择剩余 16 个场景。

作用：

> 判断“target adaptation 本身”有没有价值。

---

## Method 4：Diagnostic Support + Adaptation（DIVA-Mine MVP）

前 4 个 support 不随机，而选择最有辨识力的场景。

最简单可以用：

\[
score_{diag}(x_j)=b_j^T\Sigma b_j
\]

更贴近当前方法可以使用：

\[
score_{diag}(x_j)
=
IG(x_j)\times Boundary(x_j)
\]

然后 posterior update，剩余 16 个场景按 posterior risk 排序。

作用：

> 判断“诊断性 support”是不是比随机 support 更有效。

---

# 12. 第一版不要加这些东西

下面这些都先不要做：

- Transformer；
- CNP；
- Deep Kernel GP；
- BoTorch MultiTaskGP；
- SAC；
- PPO；
- PEARL；
- target-only BO；
- validity neural network；
- novelty model；
- failure signature diversity；
- 多功能场景；
- 连续在线 acquisition；
- 大规模 hyperparameter search。

第一轮只回答：

> **diagnose → adapt → mine 是否真的成立？**

---

# 13. 最关键的评价指标只保留 3 个

## 13.1 Critical Score@20

例如：

- collision = 1
- near-miss = 0.5
- otherwise = 0

然后：

\[
Score@20 = \sum_{t=1}^{20}R_t
\]

这是主指标。

---

## 13.2 Failure Count@20

统计：

- collision 数；
- collision + near-miss 数。

这是最直观的结果。

---

## 13.3 Ranking NDCG@10

利用 target 的完整 128-anchor row **仅用于最终离线评估**：

比较 K=4 posterior 预测出来的风险排序，与真实 target vulnerability ranking 的一致性。

它回答：

> “4 个 support 是否真的帮我们认出了新 SUT？”

但要注意：

> NDCG 只是机制指标，不能替代 failure discovery。

---

# 14. 最快的实验执行顺序

## Step 1：先确认结构存在

运行：

- source matrix SVD；
- 画 singular value ratio；
- 看 rank=2 是否能解释大部分跨 SUT variation。

建议判据：

\[
EVR_{rank2} > 80\%
\]

如果 rank=2 只有 40%-50%，不要继续纠结当前 low-rank prior。

---

## Step 2：只做离线 few-shot ranking

对每个 held-out SUT：

- K=0 shared prior；
- K=4 random support；
- K=4 diagnostic support。

比较：

- NDCG@10；
- Top-10 recall；
- Spearman。

如果：

> diagnostic support 连 ranking 都不能稳定优于 random support，

就应该先修改 SUT diversity 或 diagnostic criterion，而不是继续做 mining。

---

## Step 3：再做真正 B=20 failure mining

只有 Step 2 成立后才做：

- Random；
- Shared Prior；
- Random Support；
- Diagnostic Support。

比较 Score@20 和 FailureCount@20。

---

# 15. 什么结果才算“idea 跑通”

不要求第一轮就达到论文级显著性。

只要满足下面三个现象，idea 就值得继续：

### 条件 A：有可迁移结构

rank-2/3 能解释明显跨 SUT variation。

---

### 条件 B：少量 target tests 真有用

K=4 后：

\[
NDCG_{diagnostic}
>
NDCG_{shared}
\]

并且通常：

\[
NDCG_{diagnostic}
>
NDCG_{random-support}
\]

---

### 条件 C：最终能多找到危险场景

固定 B=20：

\[
Score@20_{DIVA}
>
Score@20_{shared}
\]

并且最好：

\[
Score@20_{DIVA}
>
Score@20_{random-support}
\]

如果这三条都能稳定看到，核心 idea 就基本成立。

---

# 16. 什么结果说明应该立即停下来检查，而不是继续加模型

如果出现下面任一种情况，应先停：

## 情况 1：所有 SUT failure boundary 基本一样

说明：

> target-specific adaptation 没必要。

解决方式：

- 修改 SUT 参数，使不同控制缺陷形成不同 vulnerability profile。

---

## 情况 2：所有方法 Score@20 一样

说明：

> 场景池太小、failure 太少，或者 B 太大，所有方法最终都扫完了 failure。

解决方式优先顺序：

1. anchors 从 128 增加到 256；
2. B 从 20 降到 10；
3. 扩大 boundary 附近场景密度。

---

## 情况 3：diagnostic support 提升 NDCG，但不提升 Score@20

说明：

> vulnerability surrogate 与真实 failure mining 目标之间没有有效转换。

这时再检查：

- vulnerability response 定义；
- failure threshold；
- mining score。

---

# 17. 随机性与可信度设计

为了让结果可信但又不拖慢：

## 数据生成阶段

对于 deterministic IDM 类 SUT：

> 第一版每个 `(SUT, scenario)` 只跑 1 次即可。

不要为了形式上的 seed 重复浪费算力。

---

## Random Support

随机 support 方法重复：

> **20 次随机种子**

因为它本身有随机性。

报告：

- mean
- std
- 95% bootstrap CI

---

## Diagnostic / Shared Prior

如果算法本身确定：

> 不需要重复完全相同的实验。

---

# 18. 防止数据泄漏的规则

必须写死：

1. target SUT 不参与 prior SVD；
2. target 完整 response row 只用于最后评估；
3. adaptation 时只揭示被选中的 K 个 support；
4. query outcome 不用于重新定义 support posterior；
5. 所有方法使用相同 candidate pool；
6. 所有方法使用相同总预算 B；
7. support 也计入 B。

---

# 19. 建议的代码目录

不要马上改现有 MetaDrive 主线。

建议新建：

```text
mvr/
├── highway/
│   ├── envs/cutin_env.py
│   ├── sut/idm_profiles.py
│   ├── data/{generate_anchor_bank,response_bank}.py
│   ├── diva/{low_rank_prior,posterior,acquisition,mining}.py
│   └── experiments/{run_loso_ranking,run_loso_mining}.py
├── scripts/
│   ├── run_diva_highway_mvp.py
│   └── render_diva_highway_mvp.py
└── tests/
    ├── test_diva_highway.py
    └── test_diva_highway_cutin.py

results/diva_highway/cutin_mvp/
```

第一版尽量不要和当前 `mvr/metadrive/diva_ai` 耦合。

等结果成立后，再把稳定实现迁回主仓库公共模块。

---

# 20. 最小输出图表

第一轮只需要 4 张图。

## 图 1：不同 SUT vulnerability map

二维场景空间：

- x = relative speed
- y = initial gap
- color = vulnerability

目的：

> 直观看到不同 SUT 的 vulnerability boundary 确实不同。

---

## 图 2：SVD explained variance

目的：

> 证明历史 SUT 风险函数具有低维公共结构。

---

## 图 3：K=4 后 ranking performance

柱状图：

- Shared Prior
- Random Support
- Diagnostic Support

指标：

- NDCG@10

目的：

> 证明 few-shot diagnosis 有效。

---

## 图 4：Fixed-budget mining curve

横轴：

\[
1...20\ tests
\]

纵轴：

\[
Cumulative\ Critical\ Score
\]

画：

- Random
- Shared Prior
- Random Support
- Diagnostic Support

目的：

> 一张图直接体现核心贡献。

---

# 21. 推荐第一轮规模

最推荐的配置：

```yaml
environment: highway-env
scenario: cut-in

num_suts: 6
num_anchors: 128

scenario_dim: 2   # initial_gap, relative_speed

prior_rank: 2

support_budget: 4
total_budget: 20

random_support_repeats: 20
split: leave-one-sut-out
```

总 source bank：

\[
6\times128=768\ episodes
\]

如果单次 episode 很快，这个规模通常已经足够做机制验证。

---

# 22. 如果第一轮成功，再做什么

只有第一轮成立以后，再按这个顺序扩展：

1. K = 1 / 2 / 4；
2. anchors = 256；
3. 加第 3 个 Cut-in 参数；
4. 增加 FVDM / delayed controller 等异质 SUT；
5. 加 target-only GP/BO 强基线；
6. 再迁移回 MetaDrive；
7. 最后才考虑 neural prior / Transformer / CNP。

---

# 23. 最终建议

当前最快、最可信的验证路径不是“把 MetaDrive 系统完整搬到 highway-env”，而是：

> **用 highway-env 快速建立一个小而干净的 SUT × Scenario vulnerability matrix，然后严格做 leave-one-SUT-out few-shot adaptation。**

最小方法只保留：

\[
\text{source response matrix}
\rightarrow
\text{low-rank prior}
\rightarrow
\text{4 diagnostic tests}
\rightarrow
\text{posterior}
\rightarrow
\text{16 failure-mining tests}
\]

只要这条链在固定 B=20 下稳定优于：

- Random；
- Shared Prior；
- Random Support；

就说明：

> **“历史 SUT 漏洞知识 + 极少目标测试 → 个性化 failure mining”这个研究 idea 是可行的。**

此时再投入 MetaDrive、复杂 surrogate 和更强 baseline 才是值得的。

---

# 参考依据

本实验设计遵循当前研究定位中提出的核心问题：严格 few-shot target budget、跨历史 SUT vulnerability transfer 与 failure-oriented mining 三者结合，并优先验证 H1-H4，即共享结构、few-shot adaptation、diagnostic support 和下游 failure discovery 增益。

同时沿用当前技术方案中的低秩 vulnerability prior、K-shot diagnostic support、解析 Bayesian posterior 与 posterior-guided mining，但为追求最快机制验证，第一版主动删除 GP、Transformer、Meta-RL、多场景和复杂 diversity 模块。
