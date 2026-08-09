# PEARL 逻辑场景少样本适应：当前实现与实验说明

本文档说明仓库中已经实现的 PEARL-SAC 流程、冻结实验协议与当前保留的正式结果；它不再是待实现的需求清单。唯一运行配置为 [`../pearl_learning/configs/merge_family_pearl.yaml`](../pearl_learning/configs/merge_family_pearl.yaml)，最终结果的机器可读来源为 [`../results/pearl_learning/key_results.json`](../results/pearl_learning/key_results.json)。

## 1. 研究问题与当前结论范围

目标是在**固定 SUT**、统一二维连续动作空间和统一奖励下，令场景挖掘策略在未见逻辑场景上先执行少量 support episode，再仅通过后验推断适应，并在独立 query case 上寻找严格有效的安全关键场景。

当前结果支持以下受限结论：在固定 taskbook 的四个未见 Y-merge 任务上、以 PEARL 已实际执行的 support 环境步数作为在线基线预算时，PEARL 的无梯度少样本适应优于 scratch SAC 与 pooled fine-tune SAC。该结论不外推到任意任务分布、任意训练预算或统计显著性结论。

## 2. 冻结任务、case 与数据隔离

一个任务由实际地图、路线、冲突几何、优先权/目标接触规则和固定 SUT 共同定义；case 只给出同一任务内的可复现实例初始条件（例如速度、出生位置和随机种子），不能改变任务逻辑。

`build_taskbook.py` 将几何目录解析为冻结 taskbook，并记录地图、路线和冲突区哈希；`casebook.py` 为每个任务冻结互不重叠的 case 集。默认每任务有 32 个 `train_pool`、10 个验证 support、20 个验证 query、10 个测试 support 和 20 个测试 query case。加载、训练和评估都会核验输入内容哈希与 split 隔离；当前正式 taskbook 哈希为：

```text
d28933ade2878c82bcd228883565394f8f8f7ae86147929c119fee1ca1be966a
```

当前划分为：

| Split | 物理几何 / 逻辑类型 | 用途 |
| --- | --- | --- |
| `meta_train` | on-ramp、lane-drop、bottleneck；其中若干几何各有两种隐藏目标接触规则变体 | 学习共享策略与上下文推断 |
| `meta_validation` | `lane_drop_40`、`bottleneck_40` | checkpoint 选择、门控与验证性分析 |
| `meta_test_template` | `lane_drop_48`、`bottleneck_48` | 未见模板测试 |
| `meta_test_logical` | `y_merge_{24,32,40,48}` | 未见逻辑类型的正式少样本测试 |

`meta_train` 与 `meta_test_logical` 的 `logical_type` 不重叠。训练中，同一物理几何包含不同的隐藏规则变体，因此拓扑描述符不能作为目标接触规则的代理标签；任务 ID、任务哈希和模板索引均不进入网络输入。

## 3. 实际环境与适配器实现

`LogicalMergeEnv`（`pearl_learning/src/task_env.py`）控制对抗车，动作空间为二维连续动作；SUT 是参数固定、关闭换道的 IDM 控制器，默认目标速度为 12 m/s，单 episode 上限为 180 步。每次重置都会复建实际 MetaDrive 环境、设置冻结 case，并复核运行时地图哈希。

代码中有三个独立的适配器实现：

- `OnRampMergeAdapter`：`on_ramp_merge`；
- `BottleneckMergeAdapter`：`lane_drop_merge` 与 `bottleneck_merge`；二者共享车道收缩/汇入实现，但通过车道数、瓶颈长度、路线和冲突规格区分几何；
- `YMergeAdapter`：`y_merge`。

因此，旧版“lane-drop 必须使用独立 `LaneDropMergeAdapter`”的表述与当前代码不符，不能作为当前实现已满足的条件。当前的泛化结论基于已冻结并审计的任务/地图哈希，而不是“每个逻辑名称都对应一个独立 Adapter 类”的前提。

## 4. 观测、奖励与主指标

观测模式为 `logical_merge_obs`，维度 37，归一化到 `[-1, 1]`。其中包含两车相对冲突区的路线状态、到达时间差、TTC、相对速度、可见优先权和拓扑描述符（分支数、车道数、汇入长度、路线曲率等）。目标接触资格规则不会输入观测。

奖励对低 TTC 与近距离给出稠密项；满足目标条件的 adversary–SUT 接触奖励为 +200，并对非目标碰撞、出界、错误路线和不平滑动作进行惩罚。

正式主指标为四任务平均 `valid_critical_strict_rate`。一个 query episode 只有同时满足以下条件才计为成功：出现目标 adversary–SUT 接触或低 TTC 的关键事件，且没有非目标碰撞、任一车辆出界或错误路线。它不等同于任意碰撞率；`target_collision_rate`、`critical_rate`、`invalid_rate` 和最小 TTC 均为辅助诊断指标。

## 5. PEARL-SAC 与无梯度 K-shot 适应

PEARL 的后验为 \(q_\phi(z\mid C)\)，默认潜变量维度为 5。context transition 为 `[obs, action, reward / 200, next_obs, terminated, truncated]`；每个 episode 先池化 32 条 transition，再将 episode 证据用 Product-of-Gaussians 合并。actor 输入 `(obs, z)`，双 critic 输入 `(obs, action, z)`；不存在任务标签输入通路。

训练时，对每个 meta-train 任务先以先验采样 `z` 收集一集 `prior_support`，再由该 context 推断后验并收集 `posterior_rollout`；二者均进入任务独立 replay。优化使用双 Q SAC TD 损失、`0.1 × KL(q(z|C)||N(0,I))`、actor 损失和自动熵温度，target critic 的软更新系数为 0.005。训练 context 会随机使用 1 到 8 个 episode，以覆盖少样本后验条件。

评估严格按以下顺序执行：

1. K=0 时直接以先验后验均值执行独立 query，绝不先收集 support。
2. 每执行一集 support 后，重新由已收集的 support 推断后验；K=1、2、5、10 分别表示真实执行的 support episode 数。
3. 每个 K 下使用后验均值确定性地执行冻结、独立的 query case；query 不进入 support 选择、后验或梯度计算。
4. 评估前后计算 context encoder、actor、critic、target critic 与温度参数哈希；发生任何变化即报错。

每个 context 最多容纳 256 条 transition，即至多 8 个 episode × 32 条 transition。因此 K=10 的成本仍是实际执行 10 个 support episode，但后验只从其中按固定随机种子抽取最多 8 集证据；不可将 K=10 解读为编码器同时使用了全部 10 集。

默认 support 选择策略为 `fixed`。代码还提供随机、初始条件多样性和 posterior-action-disagreement 策略，它们只能作为单独声明的候选实验；不能与当前固定 support 的正式结果混用。

## 6. 实验门控、基线与验证状态

正式 PEARL 训练必须显式传入 `--formal-run` 和与 taskbook 哈希相匹配的 formal gate。gate 会要求 topology/integrity audit 通过、任务异质性审计表明 pooled SAC 弱于 per-task SAC、存在正的匹配环境步数预算，并已完成 `per_task_sac`、`cross_task_policy_matrix`、`topology_conditioned_pooled_sac`、`scratch_sac`、`pooled_finetune_sac` 与 `oracle_task_conditioned_sac` 基线。

少样本公平对比由 `run_equal_budget_analysis.py` 完成：对每个 K，将 scratch SAC 和 pooled fine-tune SAC 的新任务环境步数设置为 PEARL 前 K 个 support episode 的实际累计步数，并使用同一组冻结 query case。在线 SAC 每环境步均执行一次梯度更新，计算上对 SAC 更有利；长预算 SAC 只作为充分在线训练的参考，不能证明低样本优势。

`--smoke` 只用于快速验证训练、checkpoint 加载、support 后验、独立 query 与参数哈希不变约束。smoke 结果可由 README 中的命令重建，不作为正式性能材料长期保留。

## 7. 正式结果（冻结 `meta_test_logical`）

正式评估使用选定 checkpoint `results/pearl_learning/models/selected_pearl/best_model.pt`（训练 seed 2、step 40,004）。测试含 4 个未见的 Y-merge 任务，每任务 20 个独立 query case，主指标为跨任务平均 `valid_critical_strict_rate`。

| Support episodes K | PEARL 严格有效关键率 |
| ---: | ---: |
| 0 | 0.588 |
| 1 | 0.575 |
| 2 | 0.588 |
| 5 | **0.600** |
| 10 | 0.575 |

K=5 时平均需 57 个新任务环境步，且不更新模型参数。移除拓扑描述符的 K=5 消融为 0.575，低于完整观测模型的 0.600。

| K | 平均新任务交互步数 | PEARL（无梯度） | scratch SAC（3 seeds） | pooled fine-tune SAC（3 seeds） |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11.0 | **0.575** | 0.317 ± 0.204 | 0.267 ± 0.208 |
| 2 | 19.0 | **0.588** | 0.079 ± 0.044 | 0.275 ± 0.099 |
| 5 | 57.0 | **0.600** | 0.171 ± 0.115 | 0.325 ± 0.094 |
| 10 | 119.5 | **0.575** | 0.196 ± 0.059 | 0.217 ± 0.072 |

作为非等预算参考，在每个新任务额外训练 5,000 环境步时，scratch SAC、pooled fine-tune SAC 和 topology-conditioned pooled SAC 分别达到 0.587、0.562 和 0.525。该组数值不能与上表作低样本效率比较。

## 8. 复核入口

- 实现与命令：[`../pearl_learning/README.md`](../pearl_learning/README.md)
- 最终结果说明：[`../results/pearl_learning/README.md`](../results/pearl_learning/README.md)
- 机器可读汇总：[`../results/pearl_learning/key_results.json`](../results/pearl_learning/key_results.json)
- 关键契约测试：`conda run -n metadrive python -m unittest pearl_learning.tests.test_contract`

测试覆盖冻结输入、观测维度、support/query 隔离、无梯度适应、匹配预算、任务异质性审计与迁移诊断的关键约束。正式复跑仍应使用对应的 taskbook、casebook、checkpoint manifest 和解析配置，以维持 provenance 一致性。
