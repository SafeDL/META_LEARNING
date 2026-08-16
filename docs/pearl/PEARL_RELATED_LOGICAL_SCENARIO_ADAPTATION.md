# PEARL 原论文理论与本工程实现校验

## 结论

本工程此前并非完全符合 PEARL 的理论/实验口径：context encoder 在 episode 内先平均再做高斯乘积、训练 context 来自长期 replay、query 只报告后验均值、ID/OOD 测试混合解释、无效事件惩罚缺少全回报上界，以及采集 meta-batch 可能因先有放回再去重而缩小。上述问题已在当前实现中修正。所有旧 checkpoint 和旧实验结论均已删除；当前尚无正式性能结论。

## 原论文中与代码直接相关的理论陈述

PEARL 将任务不确定性表示为潜变量 `z`。context `c={(s,a,r,s')}` 经概率 encoder 得到 `q_phi(z|c)`，策略和 critic 均以 `z` 为条件。在新任务上，网络参数保持冻结，只通过新收集的 context 更新后验，因此是无梯度快速适应。

论文采用 permutation-invariant 的 Product-of-Gaussians encoder。每条 transition 产生一个高斯因子；对角高斯乘积等价于精度相加：

`sigma^-2 = sum_i sigma_i^-2`，`mu = sigma^2 sum_i(mu_i sigma_i^-2)`。

因此，对同一组 transition 重新划分 episode 不应改变 posterior。先对 episode 内转移取均值再相乘改变了因子数量和统计含义，只能作为非论文默认的消融。

PEARL 的关键效率来自 off-policy actor-critic 与可复用 replay；但用于当前任务推断的 context 应代表近期行为分布。后验采样还承担结构化探索作用，所以只用后验均值评估会遗漏不确定性传播，而只用采样评估又会增加方差。当前实现同时报告两者。

## 当前实现逐项对应

| 理论/实验要求 | 当前实现 |
|---|---|
| transition-level PoG | `ContextEncoder(..., aggregation="transition_product")` 展平 episode/transition 维后聚合；有 regrouping invariance 测试 |
| 明确非标准变体 | `episode_product` 必须显式配置并写入 architecture metadata |
| exact prior | 空 context 返回 `N(0,I)` |
| recent context | 每任务独立 recent context buffer，16 集滚动上限，每 1,000 optimizer update 清空 |
| off-policy replay | 长期 task replay 保留；RL batch 排除本次 context episode |
| meta-batch | 采集与优化均 `replace=False` |
| 无梯度适应 | evaluation 前后校验完整参数与模块 hash |
| 后验不确定性 | 同时输出 posterior mean deterministic 和 seeded posterior sampled query |
| 泛化边界 | ID `meta_test_template` 与 OOD `meta_test_logical` 分表报告 |
| reward 有效性 | 三类 invalid penalty 在环境构造时验证大于全 episode 最大正回报 |
| 可追溯性 | 仅接受 `pearl_checkpoint` 与 `transition_product_recent_context` 方法契约；旧产物无 fallback |

## 数据与泄漏边界

观测为车辆状态、交互状态和允许的拓扑描述，不输入 task ID、geometry ID、split 或隐藏目标接触规则。support case 用于 posterior，query case 只用于评价。MoE router、表示解耦和主动 support 选择均是论文外扩展，必须作为独立方法/消融报告，不能写成原始 PEARL 的组成部分。

## 仍未由代码证明的事项

代码契约通过不等于方法有效。正式结论仍需要重新训练多个独立 seed，冻结 validation 选择，按任务做层次统计，并分别在 ID/OOD、两种 query mode、全部 K 和等交互预算基线上报告。当前只保留冻结 taskbook/casebook 输入，不保留任何旧性能数字。
