# 方法探索的文献与实现核对

核对日期：2026-11-07。这里记录方法对应关系，不作为新的实验规划。文献原文及作者代码来源见下表；本地下载缓存已清除，需要时可按链接重新获取。FST同时核对用户提供的Google Drive发表版全文。

| 文献 | 核对位置与作者代码 | 与本项目的关系 |
| --- | --- | --- |
| [Wang 等，LOL-GP，JUQ 2026](https://epubs.siam.org/doi/10.1137/24M1703057) | 正文第 3–5 节、算法 3.1/3.2；[补充材料](https://epubs.siam.org/doi/suppl/10.1137/24M1703057/suppl_file/supplement.pdf) SM1–SM6：重合坐标、权重条件分布、训练及嵌套设计。原文和公开检索未找到对应作者代码链接 | 通过 ReLU 变换 GP 权重限制局部迁移，用 Gibbs 采样推断。E1 使用已有来源碰撞概率和付费查询前似然估计局部可靠性，不实现 LOL-GP，也不宣称首次提出局部迁移。 |
| [Wilson 等，GPRN，ICML 2012](https://icml.cc/2012/papers/329.pdf) | 正文第 2–5 节、式 (1)–(7)；[补充材料](https://www.cs.cmu.edu/~andrewgw/gprnsupp.pdf) 第 1–5 节；[作者实现](https://github.com/davidaknowles/gprn)，`gpnetworkVB/gpnetworkModel.cs::NetworkModel`。作者主页另列的 MATLAB 包是 Nguyen/Bonilla 2013 的后续推断实现 | 随输入变化的混合矩阵连接潜 GP；原文为高斯多输出回归，ESS/VB 推断。本文碰撞为 Bernoulli，RSS 为 Gaussian，E2 必须给出自身有效协方差及关闭共享的极限，不将简化模型称为 GPRN 复现。 |
| [Moreno-Muñoz、Artés、Álvarez，异质多输出GP，NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/165a59f7cf3b5c4396ba65953d679f17-Abstract.html) | [全文](https://proceedings.neurips.cc/paper_files/paper/2018/file/165a59f7cf3b5c4396ba65953d679f17-Paper.pdf)第2–3节，LMC及异质观测推断；[作者代码](https://github.com/pmorenoz/HetMOGP)。本轮核对模型依据，没有引入该代码或声称复现其训练 | 不同输出可使用Bernoulli、Gaussian等各自的似然，再以共享潜GP关联。E2属于小规模共享/独立潜过程组合，采用RSS高斯条件化及碰撞Laplace近似；没有实现原文的诱导变量随机变分推断。混合观测似然和LMC不作为本文通用模型首创。 |
| [Li、He、Yang、Hu、Zhang、Feng，FST，T-ITS 2025](https://doi.org/10.1109/TITS.2025.3622257) | 发表版26(12):22804–22817，2025年12月；用户全文及[作者公开PDF](https://www.jingxuanyang.com/file_upload/2025-FST-TITS.pdf)，第III–IV节、式(1)/(4)/(7)/(8)，相似性学习及代理模型凸包；[预印本](https://arxiv.org/abs/2409.14369)为2024年12页版本，不能据此误标发表年份或页码 | 在少量测试下，优化固定测试集对整体性能指标的估计误差，使用代理系统与跨注意力场景相似性。本文逐轮依据目标反馈，在有限候选集最大化预算内失效发现；两者预算目标不同。须引用其少样本测试和跨系统先验思想，不能宣称首次提出少样本ADS测试。没有为本轮新增不对应目标的基线。 |
| [Jiang 等，ENS，ICML 2017](https://proceedings.mlr.press/v70/jiang17d.html) | 第 2–4 节，式 (5)–(7)、复杂度与剪枝；[附录](https://proceedings.mlr.press/v70/jiang17d/jiang17d-supp.pdf)；[作者代码](https://github.com/shalijiang/efficient_nonmyopic_active_search)，`query_strategies/ens_with_pruning.m` | 一步幻想标签后，对剩余最高概率求和。E3 的预算效用直接沿用此近似思想，新增问题是本次碰撞与 RSS 的联合幻想及来源更新；不声称非短视效用本身是原创。 |
| [Nguyen 等，MF–ENS，ICML 2021](https://proceedings.mlr.press/v139/nguyen21f.html) | 第 2–5 节，式 (1)–(5)、异步高低保真查询；补充 ZIP 中 `supp.pdf` 的算法 1/2 与剪枝结果；[作者代码](https://github.com/KrisNguyen135/multifidelity-active-search)，`as_async/policies/h_policies/replace_shortened_h.m` | 低保真结果不直接计入目标发现，使用额外低保真查询改进后续高保真批次。本文 RSS 与碰撞来自同一次已付费执行，不增加低保真查询预算，不能直接照搬其调度或宣称同一问题。 |
| [Parashar 等，配对系统主动评估，2026 预印本](https://arxiv.org/html/2608.13719v1) | 第 3–5 节、式 (1)–(3)，附录 A–I；局部控制变量、支持 MI、批量候选与 nuPlan 设置。论文页未给出该方法作者代码链接，本文现有实现另有复现说明 | 基于代理响应及目标预测构建局部控制变量，并以支持 MI 促进探索。E1 调整多个历史来源的后验混合，E3 对联合反馈的后续发现价值计算效用。两者与局部修正及覆盖导向存在相邻关系，贡献须由具体差异和发现证据支撑。 |

E1 固定使用高斯邻域、预训练核带宽及 20 条记录的收缩强度，无新增可训练参数；完整公式与资源额度已在 `protocol.json` 登记。诊断只复用旧 Random 付费记录，不读取未查询响应。旧目标回放属于开发证据，后续新平台固定比较才检验能否复现。

E3 的实际对应为 `JointLookahead.choose` 中的 ENS 式 (7) 剩余预算效用：先计算本次碰撞概率，再对模型模拟的碰撞与 RSS 联合响应求后续最高概率之和。短名单固定为两点风险排序加两点分量分歧，每点使用四次联合模拟；这是有限短名单上的 Monte Carlo 近似。动态来源权重及参数适配可能使负反馈提高其他候选的概率，本文不具备原 ENS 安全剪枝所需的单调性条件。`sample_responses` 从当前分量混合的二维潜后验及观测模型生成模拟响应，`fantasy_prediction` 使用与实际查询相同的来源更新及适配步骤。RSS 来自同一次执行，不构成 MF–ENS 的额外低保真查询；模拟分支与真实查询分别计数。

## 当前方法可以提出的贡献

最接近本文评价目标的是配对系统主动评估：均需利用相关系统响应，在有限目标预算内发现失效。其局部控制变量估计与支持互信息选例，与本文保留各历史ADS联合后验、查询前碰撞似然来源加权、冻结表征的目标残差适配及逐次贪心风险选例不同。来源数量、联合反馈和适配机制的差异需要通过同权限基线与内部对照体现，不能仅凭系统模块更多判断创新更强。

E2为碰撞与RSS保留共享及各自的变化尺度，工程检查可证明关闭共享时的条件预测隔离。旧开发数据上发现数与AUC同向改善，支持这一建模流程有价值；同时变化的结构、尺度及梯度分组尚无独立归因。LMC、局部迁移和非短视搜索均已有成熟先例，本文的方法创新应围绕跨ADS响应差异和少量目标执行反馈如何改善具体驾驶交互中的失效发现。

RSS条件排名在部分前缀改善，概率评分总体变差；现有证据不支持“RSS普遍提高发现效率”，也不证明“RSS完全无信息”。任务后归档、来源构建和模型发布有实现，当前新平台轮次未开展积累效果比较。上述边界来自已有证据，不增加一组试验来补足主张。集合与时机、控制器家族、查询前权重的独立复核见[收益分析](reports/gain_analysis.md)。
