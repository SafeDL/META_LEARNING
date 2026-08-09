# 面向可迁移性的可解释元强化学习：研究目标与实施路线

本文以当前工程代码和冻结的最终实验材料为依据编写。除非明确标为“已完成”或给出冻结结果数值，否则内容均为待检验的研究目标，不能视为实验结论。

本工程的 GPU 训练环境为：

```powershell
conda activate metadrive
```

## 术语约定

本文将 `support episode` 称为“支持回合”：在新任务上允许收集、用于推断任务后验的一整段交互轨迹；将 `query case` 称为“查询场景”：只用于独立评估，不能用于适应、阈值选择或策略更新。K 表示支持回合数，不表示环境步数、梯度步数或查询场景数量。代码、命令行参数和指标名保持原英文拼写，以便与工程一一对应。

## 1. 工程基础与问题定位

本工程已实现面向自动驾驶逻辑汇入场景的 PEARL 元强化学习流程。每个逻辑任务固定地图几何、对抗车与被测车辆（SUT）的路线、出生区间、优先权、冲突定义及案例随机种子；训练、验证和测试案例分别冻结在 taskbook 与 casebook 中。模型在多个 `meta_train` 任务上完成离线元训练后，面对未参与训练的 `meta_test_logical` 任务，仅依据 K 个 support episode 推断任务后验，再在独立的 query case 上执行策略。该阶段不计算梯度、不更新网络参数，也不进行针对新任务的重新强化学习。

当前保留的最终模型使用 37 维无任务标签观测，其中包含车辆状态、相对运动与 TTC 等交互量，以及入/出支路数、车道数、汇入长度、冲突区尺度、路线曲率和限速等拓扑描述符；任务潜变量维数为 5。速度关系和到达顺序等目标接触规则不作为观测字段，必须借助 support 交互结果辨识。上下文编码器将 episode 内的转移先聚合，再以高斯乘积形成任务后验。因而，当前 PEARL 已具备“以交互历史调节策略”的能力，但尚不能解释后验各维的语义，也不能判断一个新场景是否确实处于可迁移范围内。

工程现已提供统一入口 `audit_support.py posterior`，可在不生成 query rollout 的条件下导出各任务、各 K 的后验均值、对角方差、support case ID 和实际环境步数，并以参数哈希验证诊断过程未修改模型。在当前冻结的未见逻辑任务上，平均后验方差由 K=0 的 1.0000 下降至 K=1 的 0.1229、K=2 的 0.0655 和 K=5 的 0.0259；K=5 的平均 support 交互为 57.0 个环境步。该结果证明 support 会收缩当前模型的高斯后验，但不赋予潜变量预定义语义，也不能单独证明后验收缩带来迁移收益。

```powershell
conda run -n metadrive python -m pearl_learning.scripts.audit_support posterior `
  --config pearl_learning/configs/merge_family_pearl.yaml `
  --checkpoint results/pearl_learning/models/selected_pearl/best_model.pt `
  --taskbook results/pearl_learning/taskbooks `
  --casebook-root results/pearl_learning `
  --split meta_test_logical --shots 0 1 2 5 `
  --output <support-posterior-report.json>
```

最终测试覆盖 4 个未见的 Y 型汇入逻辑任务，每个任务在 20 个独立 query case 上评估。主指标为四任务平均 `valid_critical_strict_rate`：一个 query episode 只有在出现目标 adversary–SUT 接触或低 TTC 等关键事件，且未发生非目标碰撞、车辆出界或错误路线时才计为成功。因此，该指标不是简单的碰撞率，也不应解释为“60% 的场景发生了碰撞”。当前 K=5 时该指标为 0.600，平均消耗 57 个新任务环境步；相同新任务交互预算下，scratch SAC 和 pooled fine-tune SAC 分别为 0.171 ± 0.115 与 0.325 ± 0.094。该证据支持少样本适应的样本效率，不支持“PEARL 在任意训练预算下均优于重新训练 SAC”的结论：给定每个新任务 5,000 个环境步时，scratch SAC 的参考值为 0.587。

后续研究不应在现有流程上堆叠孤立模块，而应围绕一个可检验的问题展开：**对于一个新逻辑场景，系统能否先判断既有元知识是否可迁移，再以最少、最有信息量的交互完成可靠且可解释的关键场景挖掘？**

## 2. 总体目标与工作流

拟构建“迁移性评估—主动适应—风险感知挖掘”一体化框架。输入为一个符合现有 taskbook 规范的新逻辑任务及其候选案例；系统先由可解释的任务表示刻画几何、交互和规则特征，随后给出迁移性评分与不确定性。当评分足够高且不确定性可接受时，系统选择信息量最大的少量 support case 进行无梯度后验推断，并在独立 query case 中挖掘严格有效的关键场景；当评分低或置信度不足时，系统应明确拒绝直接迁移，并转入受限在线微调、补充元训练任务或从头训练等预先声明的后备策略。

这里的“迁移性”是指：在规定的新任务交互预算和相同冻结 query 集上，使用已有元模型的适应结果相对无迁移或在线基线所带来的预期收益。它不是地图外观相似度，也不是由测试集结果反推得到的标签。所有用于迁移判断的特征和阈值都必须只依赖新任务的先验描述与允许收集的 support 交互。

## 3. 可解释的任务表示

### 3.1 现有基础与局限

现有潜变量可记为 \(z\sim q_\phi(z\mid C)\)，其中 \(C\) 是 support episode 的状态、动作、奖励、下一状态和终止信息。它能够服务于策略适应，却没有预定义的语义分工。虽然 37 维观测已经输入拓扑描述符，且 K=5 的去拓扑消融从 0.600 降至 0.575，但这一差异只能说明拓扑输入在该冻结测试上的经验贡献，不能说明潜变量已经分离出可解释的几何或规则因素。

### 3.2 研究目标

工程现已提供可选的解耦训练实现，但尚无正式性能结论。启用 `train_pearl.py --disentangled-representation` 后，仍使用原有 5 维后验和同一 actor/critic；其中前 2 维经辅助头预测训练任务的几何标量，接续 2 维预测 support 转移中无标签的到达时差、相对速度和 TTC 统计，最后 1 维预测训练期冻结的二元到达顺序。辅助目标只在 `meta_train` 反向传播，任务 ID、哈希和 query 标签从不输入编码器或策略；元测试仍只以 support 轨迹推断后验。工程通过统一入口 `audit_support.py representation` 审计该表示：它只收集 support rollout，导出各分量预测、观测到的交互统计和参数不变性；几何与冻结到达顺序仅在推断完成后用作事后评分标签，绝不输入模型。审计还会对已收集的 support 上下文分别掩蔽几何描述符、交互状态和可见优先权字段，重推后验并报告各潜变量块的变化量。这只能作为表示敏感性诊断，不能被表述为因果解耦证明；当前也没有保留足以支持性能结论的完整实验结果。

将任务表示显式拆为

\[
z=[z_G,z_I,z_R],
\]

其中 \(z_G\) 表示道路几何与拓扑（汇入分支、车道组织、汇入长度、冲突区和路线曲率），\(z_I\) 表示交互动力学（相对速度、到达时序、间距与 TTC 演化），\(z_R\) 表示规则和有效性约束（优先权、目标接触判据、路线约束及无效事件）。各分量应来自当前 taskbook、观测和 support 轨迹中可追溯的变量，而不是从任务 ID、模板编号或 query 标签中泄漏获得。

研究重点不是要求每个潜变量与某个物理量一一对应，而是验证其可用性：保持其他因素不变时，改变几何、交互或规则中的一个因素，应主要改变相应子表示；相似性检索、迁移判断和策略性能应能从该结构化表示获益。可采用监督辅助头、因素置换一致性、互信息约束或对比学习，但必须保留原始 PEARL 作为消融基线。

## 4. 迁移性与拒绝迁移机制

### 4.1 迁移性评分

针对新任务 \(T_{new}\) 与元训练任务集合 \(\mathcal{T}_{train}\)，定义可学习且可校准的迁移性评分

\[
S(T_{new},\mathcal{T}_{train})=
f\bigl(S_G,S_I,S_R,U\bigr),
\]

其中 \(S_G\)、\(S_I\)、\(S_R\) 分别衡量结构化表示中的几何、交互和规则兼容性，\(U\) 为评分或后验的不确定性。评分模型的监督目标应是预注册的、预算匹配的实际迁移收益，例如

\[
\Delta_K=\operatorname{VCSR}_{\mathrm{PEARL},K}-
\max\left(\operatorname{VCSR}_{\mathrm{scratch},B_K},
\operatorname{VCSR}_{\mathrm{pooled\text{-}ft},B_K}\right),
\]

其中 \(\operatorname{VCSR}\) 表示 `valid_critical_strict_rate`，\(B_K\) 是 PEARL 前 K 个 support episode 实际消耗的环境步数。评分拟预测的是该收益或“收益为正”的概率，而非直接预测 query 是否会发生碰撞。

### 4.2 决策与后备策略

设定在验证任务上确定的阈值 \(\tau_S\) 和不确定性阈值 \(\tau_U\)。当 \(S\geq\tau_S\) 且 \(U\leq\tau_U\) 时，执行元适应；否则输出“迁移证据不足”，并采用明确的后备分支：补充代表性 support、有限预算在线 SAC、纳入后续元训练队列，或停止对该 OOD 任务作性能承诺。拒绝迁移的正确性需要与“盲目迁移”一同报告，包括接受率、误接受率、误拒绝率、风险—覆盖率曲线以及拒绝后策略的资源成本。

当前工程已实现 `pearl_learning.scripts.audit_transferability`：它从冻结 taskbook 和允许使用的 support case 元数据生成可解释任务描述、最近邻元训练任务、未见逻辑类型/地图类型标记及相似度报告。该命令不读取 query case 或 query 指标，默认也不读取隐藏接触规则；因此它是迁移性**诊断**而不是迁移收益预测器。验证集阈值校准接口已实现：若同时提供同一验证 split、同一 K 的 `audit_support posterior` 输出，校准器会将 support-only 后验方差均值作为第二个候选条件，并只在独立验证任务数与正负迁移收益类别均足够时联合选择相似度下限和方差上限；若未提供该审计则保持相似度单阈值。`decide_transferability.py` 仅在联合阈值已经校准时要求待决任务的 support-only posterior audit；缺失该输入或证据不足时，均会保守输出 `defer`，不会伪造阈值或允许直接迁移。这仍不是学习式收益评分器或完整 OOD 泛化器，必须以留一任务验证和独立留出测试证明其效用。

```powershell
conda run -n metadrive python -m pearl_learning.scripts.audit_transferability `
  --taskbook results/pearl_learning/taskbooks `
  --casebook-root results/pearl_learning `
  --candidate-split meta_test_logical `
  --output <transferability-report.json>
```

工程现已补充验证集校准接口：`run_equal_budget_analysis.py --split meta_validation --taskwise-summary` 将每个验证任务的 PEARL support 环境步数逐一匹配给 scratch SAC 与 pooled fine-tune SAC，并保留逐任务、无 query 轨迹的汇总；`calibrate_transferability.py` 将其与描述报告合并，定义 \(\Delta_K\) 并尝试选择相似度阈值。若额外传入同 split、同 K 的 support-only posterior audit，它会联合搜索后验方差上限；否则不虚构不确定性阈值。该工具将“任务”而不是 K 或随机种子视为独立样本，且只有在预设数量的验证任务同时包含正、负迁移收益时才输出阈值。除阈值本身外，报告还对每个验证任务执行留一任务阈值选择：被留出的任务不能参与自身阈值拟合，并报告可评估覆盖率、准确率和类别平衡准确率；该诊断不足时同样不能被解释为泛化证据。

```powershell
conda run -n metadrive python -m pearl_learning.scripts.calibrate_transferability `
  --descriptor-report <validation-descriptor.json> `
  --taskwise-summary <validation-equal-budget-taskwise.json> `
  --posterior-audit <validation-support-posterior.json> `
  --shot 5 --minimum-independent-tasks 8 `
  --output <validation-calibration.json>
```

当前冻结任务簿只有 2 个验证任务。以 K=5 运行的预算匹配验证中，bottleneck_40 与 lane_drop_40 的 \(\Delta_5\) 分别为 +0.667 与 +0.050；两者均为正。因此校准器按设计返回 `insufficient_validation_evidence_no_threshold`（独立任务数不足且标签只有一个类别），没有生成阈值。该结果是对当前证据边界的正确处理，而不是“迁移性已被证明”。

当且仅当校准器产生验证阈值后，可使用下列命令生成运行时决定；在当前 2 个验证任务证据下，它会输出 `deferred_insufficient_validation_evidence` 并建议收集代表性 support 或执行受限在线 SAC：

```powershell
conda run -n metadrive python -m pearl_learning.scripts.decide_transferability `
  --descriptor-report <candidate-descriptor.json> `
  --calibration <validation-calibration.json> `
  --posterior-audit <candidate-support-posterior.json> `
  --output <transfer-decision.json>
```

## 5. 主动少样本适应

### 5.1 现有协议

当前测试中的 support case 已预先冻结，并按顺序收集；为形成后验，评估器从已获得的 support episode 中用固定种子采样转移。元训练阶段会随机化上下文 episode 数量，以覆盖 K=1、2 等少样本条件。这保证了可复现与公平的预算匹配，但不构成主动选择。换言之，K 是完整 support episode 的数量，不是 K 次环境步，也不是 K 个 query 测试场景。

### 5.2 新目标

工程现已提供 `fixed`、确定性 `random`、`initial_condition_diversity` 与逐回合 `posterior_action_disagreement` 四种 support 选择接口。`initial_condition_diversity` 只对冻结候选池中执行前即可见的 `adversary_speed_mps`、`adversary_spawn_m` 和 `sut_spawn_m` 做归一化最远点贪心排序。`posterior_action_disagreement` 在每个已完成 support 后，只读取尚未执行候选的初始观测；以当前 PEARL 后验采样得到的确定性动作方差作为**决策不确定性代理**，选择分歧最大的候选，再执行该 support 并更新后验。它不读取 query、未执行候选的 rollout/奖励/终止结果或隐藏接触规则；初始观测探测不调用 `step`，因此记录为零环境步，同时逐轮记录探测候选数量。两种策略均记录候选、逐轮分数和选中 case ID。该接口是主动适应实验的可复核起点，**不是**“后验信息增益已被估计”或“主动方法已优于固定顺序”的证据。正式比较必须仍在相同候选池、独立 query 集和真实累计环境步数下，对 `fixed`、`random`、`initial_condition_diversity`、`posterior_action_disagreement` 与预算匹配 SAC 一并报告；后者使用 PEARL 实际选出的同一前 K 个 support case。

在不窥视 query 结果的前提下，从允许的候选 support case 中顺序选择案例 \(c_t\)，以尽快降低任务不确定性并提高预算匹配的 query 表现。一个可操作的效用函数为

\[
c_t^*=\arg\max_{c\in\mathcal{C}_t}
\left[I(z;\tau_c\mid C_{1:t-1})-\lambda\,\operatorname{Cost}(c)
-\eta\,\operatorname{Risk}_{invalid}(c)\right],
\]

其中 \(\tau_c\) 是执行候选案例得到的 support 轨迹，\(I(\cdot;\cdot)\) 可由后验熵下降、模型分歧或近似信息增益估计，成本以实际环境步数计，最后一项避免通过明显无效的轨迹换取表面信息量。若候选 case 不能在执行前完全预测结果，应仅使用其已知初始条件、任务描述和当前模型预测。

主动方法必须与三类对照比较：现有固定顺序 support、随机选择 support，以及不使用元后验的预算匹配 SAC。评价应同时报告达到给定严格有效关键率所需的环境步数、在固定环境步预算下的严格有效关键率、后验不确定性校准与有效轨迹比例。禁止将为主动策略挑选的 support 与另一方法使用的 query 混用。

## 6. 风险感知的关键场景挖掘

现有奖励以目标接触、低 TTC、接近程度和动作平滑等项引导对抗车行为，并用非目标碰撞、出界和错误路线的惩罚约束有效性；最终指标已通过严格有效性筛除无意义失败。下一阶段应在此基础上建模“关键性可信度”，而不是单纯提高接触事件数量。

建议将候选场景的排序目标扩展为

\[
J_{risk}=w_c P(\text{valid critical}\mid T,C)
+w_u U_{risk}+w_n N-w_i P(\text{invalid}\mid T,C),
\]

其中第一项估计严格有效关键事件的概率，\(U_{risk}\) 描述模型认识不足但值得验证的区域，\(N\) 衡量与已发现关键案例的差异性，最后一项惩罚无效事件风险。概率应使用冻结的独立 query case 或额外验证案例校准；不能把训练奖励、目标碰撞率或未筛选的低 TTC 比例直接当作安全风险结论。

除 `valid_critical_strict_rate` 外，实验应分解报告目标接触率、关键事件率、无效率、最小 TTC/最小距离分布、首次严格有效关键事件所需 episode 数，以及案例多样性。工程的 `summarize` 已输出前述分解指标，并新增 `valid_critical_initial_condition_diversity`：在同一冻结 query 池的 `adversary_speed_mps`、`adversary_spawn_m`、`sut_spawn_m` 三个可见初始条件上归一化后，计算严格有效关键 case 的平均两两距离；少于两个有效关键 case 时该值为 `null`，并同时报告 case 元数据覆盖率。它只是次要诊断，不能取代严格有效关键率，也不表示真实道路风险的全覆盖度。这样可以区分“发现更多有意义且初始条件不同的关键场景”和“以出界、非目标碰撞或重复轨迹换取高奖励”两种情况。

## 7. 持续元学习与任务记忆

道路拓扑、交通规则、SUT 版本和场景分布可能逐步扩展。长期目标是在冻结 taskbook 的可复现实验协议之上建立版本化任务记忆库，记录每个任务的结构化表示、案例哈希、训练版本、迁移表现和不确定性。新任务可检索相似历史任务用于支持选择和迁移性估计；新增任务进入元训练时，应通过回放或正则化防止遗忘。

持续学习实验必须按时间或版本划分任务，不能把未来任务的 query case 回流为当前模型选择依据。需报告旧任务保持率、新任务适应收益、任务库增长成本与灾难性遗忘指标，并保留 taskbook/casebook/provenance 以确保每次比较可复核。

## 8. 实验设计与验收标准

后续实验沿用当前冻结输入、独立 support/query 划分和严格指标，新增内容应满足以下原则。

1. **表示验证。** 比较原始 5 维 PEARL 后验、结构化表示、仅拓扑、仅交互、仅规则及去除对应分量的模型。除性能外，应报告因素干预后的表示一致性和可解释性检验。
2. **迁移性验证。** 将任务按几何、交互和规则的近邻与远离程度分层，并构造未覆盖组合。仅用允许的任务描述和 support 数据预测 \(\Delta_K\)，报告相关性、校准误差、接受—风险曲线和拒绝迁移后的整体效用。
3. **主动适应验证。** 在每个 K 或环境步预算下，对固定顺序、随机、信息增益和不确定性策略使用相同的候选池与独立 query 集。若主动方法改变了已执行 support 的集合，应按真实累计环境步匹配在线 SAC，而不能只匹配 episode 数。
4. **公平基线。** 保留原始 PEARL、PEARL 无上下文、scratch SAC、pooled fine-tune SAC、拓扑条件 pooled SAC、per-task SAC，以及明确标记为特权信息的 oracle task-conditioned SAC。少样本结论使用预算匹配基线和多个随机种子；长预算 SAC 仅作为性能上限或参考，不与少样本结论混用。
5. **统计与复现。** 对含随机训练的基线至少报告多个独立种子的均值和标准差；明确预先指定主指标、K 值、阈值和淘汰规则。结果目录只保留选定模型、配置、任务/案例输入、哈希和汇总指标；大模型二进制可按仓库规则排除，但其 manifest、配置与可重建说明必须保留。

在运行任何留出测试前，工程可用 `freeze_validation_protocol.py` 将验证集上的 support 策略、同 checkpoint 哈希、等预算报告、表示审计与校准报告绑定为 `validation_frozen_for_holdout` manifest。随后 `evaluate_fewshot.py --validation-freeze-manifest <manifest>` 会在测试 split 上验证 taskbook 与 checkpoint 未被替换。该门控本身不读取留出 query 结果，也不宣称验证方法有效；它只阻止在测试得分出现后悄然切换模型或策略。

## 9. 分阶段实施目标

第一阶段已建立不改变当前主流程的诊断接口：从 taskbook 和允许的 support case 导出无 query、无默认隐藏规则的任务描述与最近邻覆盖报告；从 support 轨迹导出 PEARL 后验及方差；在验证任务上计算逐任务预算匹配收益，并在证据不足时拒绝生成迁移阈值。工程还提供 8 个验证几何的候选 taskbook 构建器和预算对称的异质性审计。候选实验若未形成冻结、可复核的最终结果，不作为当前结论保留。

第二阶段完善结构化表示与迁移性拒绝器。当前辅助头、支持回合语义审计和“校准不足即 defer”的运行时接口均已具备；仍需在独立验证任务上校准收益阈值与后验不确定性阈值，并以原始 PEARL 和现有 K=5 拓扑消融为对照。若评分器不能预测收益，或拒绝机制不能降低误迁移风险，则不将其纳入主方法。

第三阶段实现主动 support 选择。在与现有固定协议相同的冻结候选池、独立 query 集和环境步预算下，验证是否能够以更少的实际环境步达到相同或更高的严格有效关键率。

第四阶段再引入风险排序与持续任务记忆，并开展跨任务簇、任务版本和 SUT 版本的长期验证。每一阶段都应保存与当前工程一致的任务哈希、案例哈希、模型 manifest 和紧凑汇总结果，避免历史临时输出取代可复核证据。

## 10. 最终研究目标

研究目标应从“在未见场景上使用元强化学习进行少样本适应”，推进为：**在严格有效关键场景挖掘任务中，学习可解释的可迁移结构，事先判断何时值得迁移，以主动且预算可控的交互完成后验适应，并在迁移证据不足时可靠地拒绝或切换策略。**

该目标强调的是可验证的迁移收益、适用边界和资源效率，而不是将关键事件数量、低 TTC 或单一测试集得分直接等同于真实道路安全性或无条件的算法优势。
