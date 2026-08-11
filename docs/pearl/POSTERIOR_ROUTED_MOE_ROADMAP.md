# Posterior-Routed MoE 研发路线

本文是四份研究目标文档的入口。四个阶段必须顺序执行；前一阶段未满足证据要求时，不得把后一阶段的代码或结果并入主方法。路线的最终研究对象是：

> 在固定 SUT、二维连续低层动作空间和统一奖励下，利用新任务 support 轨迹形成的概率后验进行任务级专家路由，使场景挖掘策略在未见 merge-family 逻辑场景上无梯度适应，并生成严格有效的安全关键场景。

这里的专家是 `scenario-mining behavior experts`，控制对抗车辆的转向与加减速，不是地图、天气或出生点编辑器。

当前已实现 PEARL 基线与保留实验见 [PEARL 逻辑场景少样本适应](PEARL_RELATED_LOGICAL_SCENARIO_ADAPTATION.md)，迁移性扩展的既有边界见 [Transferability-Aware Meta-RL 新目标](Transferability_Aware_Meta_RL_New_Objectives.md)。本路线不会把后者的可选表示解耦、主动 support 或拒绝器与首轮 MoE 同时启用。

历史实验目录已迁移到语义化路径。历史 JSON/checkpoint 内部记录的是运行当时的原始路径和 schema，作为不可变 provenance 保留；新配置、脚本和新实验只使用本文定义的语义化名称。

## 研究阶段

| 阶段 | 目标文档 | 必须回答的问题 | 当前状态 |
| --- | --- | --- | --- |
| 后验适应验证 | [后验适应有效性验证](POSTERIOR_ADAPTATION_VALIDATION.md) | support 是否真的改变后验并改善未见任务表现？ | INCOMPLETE；协议与三 seed smoke pilot 已完成，正式训练和 holdout 未执行 |
| 路由 MoE 工程验证 | [后验路由 MoE 实现](POSTERIOR_ROUTED_MOE_IMPLEMENTATION.md) | MoE 是否按无泄漏、可复现的工程契约正确运行？ | **PASS（工程正确性）**；真实 smoke 通过 |
| 路由 MoE 机制验证 | [MoE 机制验证](POSTERIOR_ROUTED_MOE_MECHANISM_VALIDATION.md) | 增益是否来自后验驱动路由，而非容量、拓扑或随机性？ | **PILOT PASS（代码/机制契约）**；10k-step 预实验未显示正向交互或 frozen-router 退化，正式状态仍为 INCOMPLETE |
| 约束场景挖掘 | [约束式场景挖掘](CONSTRAINED_SCENARIO_MINING.md) | 能否在关键性、有效性、合理性和多样性之间取得可验证权衡？ | 等待机制验证 |

## 当前基线事实

- dense actor 与 dense twin critics 仍保留；[moe.py](../../pearl_learning/src/moe.py) 已新增 posterior router 和 residual MoE actor，只有 actor 使用 MoE，critic 未改变。
- [pearl_agent.py](../../pearl_learning/src/pearl_agent.py) 在 actor 和 critic 更新时使用分离梯度的 latent；测试适应只推断后验，不更新网络参数。
- [context_encoder.py](../../pearl_learning/src/context_encoder.py) 先在 episode 内池化，再用 Product-of-Gaussians 聚合。方差会随证据因子增加而结构性收缩，尚不能直接视为经过校准的认知不确定性。
- 当前配置使用 37 维观测、2 维动作和 5 维 latent。评估上下文为 `256 / 32 = 8` 个 episode 容量，因此原 K=10 不代表后验同时使用全部 10 个 support episode。
- [key_results.json](../../results/pearl_learning/key_results.json) 中 K=0/1/2/5/10 的严格有效关键率分别为 0.588/0.575/0.588/0.600/0.575。结果证明了低预算元策略有竞争力，但尚未证明 support 驱动的稳定适应。
- 当前正式结果只选择了一个 PEARL 训练 checkpoint；[checkpoint manifest](../../results/pearl_learning/models/selected_pearl/best_model.manifest.json) 记录其为 training seed 2、step 40004。配置中的 1,500,000 总环境步是训练上限，不应误写成该 checkpoint 已完成的步数。正式机制结论需要至少 3 个独立训练种子。

路由 MoE 的完整工程审计见 [manifest](../../results/pearl_learning/posterior_routed_moe/manifest.json)。后验适应验证仍为 `INCOMPLETE`；路由 MoE 是用户在审阅物理可控性不对称后明确授权的工程例外，不构成后验适应通过或 MoE 性能优越性证据。

## 共同实验定义

- `K`：未见任务上实际执行的完整 support episode 数，不是梯度步数或 query 数。
- `B_K`：前 K 个 support episode 的累计真实环境步数，必须与 K 同时报送。
- 主指标：`valid_critical_strict_rate`，简称 VCSR。
- 统计单位：任务。一个任务内的多个 query case 是重复测量，不能伪装成独立任务。
- 适应边界：测试阶段仅更新 `q(z | C_K)` 和由它计算的路由；actor、critic、context encoder、router、expert 参数均不得更新。
- 数据边界：support 只用于后验与路由，query 只用于最终评价。task ID、geometry ID、split、隐藏规则和 query 结果不得进入策略或 router。
- 正式 few-shot 轴：`K ∈ {0, 1, 2, 4, 8}`，并冻结 support 顺序、每个 episode 的 transition 子样本和 query case 顺序。
- 复现边界：所有正式产物保存配置哈希、taskbook/casebook 哈希、checkpoint 哈希、Git commit、训练种子和评估种子。

## 硬停止规则

1. 后验适应未证明 support 必要性，不实现正式 MoE；先修复任务分布、上下文协议或后验学习。
2. 路由 MoE 未通过无泄漏、checkpoint、梯度和路由稳定性测试，不运行正式 MoE 实验。
3. 机制验证未证明正向交互与任务特异路由，不把 MoE 写成主创新，也不进入 RSS 主实验。
4. 约束场景挖掘未形成经验证的责任定义和 Pareto 证据，只能表述为有效性/合理性约束，不使用“RSS 责任有效”或“安全保证”等强结论。

## 共同执行约束

- 保留当前 dense PEARL 的配置、checkpoint manifest 和正式结果，新增配置与产物不得覆盖它们。
- 遵守 [style.md](../style.md)：配置集中、实现直接、错误显式；不增加一次性 wrapper、无效 fallback 或遗留兼容层。
- 每次只引入当前阶段必要的一个主要变量。可选表示解耦、主动 support、迁移拒绝器和风险记忆不得与首轮 MoE 同时开启。
- 每个阶段交付时，必须列出改动文件、运行命令、测试结果、产物哈希、未满足项和是否准许进入下一阶段。
