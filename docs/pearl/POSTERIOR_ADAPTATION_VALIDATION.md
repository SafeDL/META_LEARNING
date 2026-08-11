# 后验适应验证：证明 support 驱动的后验适应

## 验证目标

在不引入 MoE 的前提下，建立一套冻结、无 query 泄漏、按任务统计的验证协议，回答：support episode 是否使 PEARL 后验识别隐藏任务差异，并在独立 query case 上产生稳定、可归因的收益。

这是路由 MoE 的硬前置条件。完成代码或跑通实验不等于通过；只有证据链成立才允许进行路由 MoE 工程验证。

## 2026-08-09 执行状态

当前状态为 **INCOMPLETE，禁止进入路由 MoE 工程验证**。本轮已在 `conda activate metadrive` 环境中完成协议实现、真实地图审计和三 seed pilot，但没有将 smoke 结果冒充正式结论。

已完成：

- 新增 [后验适应协议](../../pearl_learning/configs/posterior_adaptation_protocol.yaml)，固定 K=`0/1/2/4/8`、每 episode 32 个 transition、总容量 256，并使用 `fixed_nested_v1`。
- 构建 10/4/4/8 个 meta-train/meta-validation/meta-test-template/meta-test-logical 任务；验证与测试的每个物理几何均有 adversary-first 与 SUT-first 配对。冻结 taskbook 哈希为 `e4b9ca30d1b37ae7f24b082e23dcfed0906f18673b3dc917fe99e9d0981aec9f`。
- 配对任务共享完全相同的数值初始条件，但 case ID 和 seed 全局不重复；26/26 真实地图拓扑审计通过，完整性审计通过。
- 39 项自动测试通过。评估输出证明 K=0/1/2/4/8 的上下文 transition 数严格为 0/32/64/128/256，且整体参数与 context encoder、actor、双 critic、target critic、alpha 的逐模块哈希均不变化。
- 独立执行纯 support 后验审计；产物记录 `uses_query_cases=false`、`no_gradient_adaptation=true`，适应前后整体参数与逐模块哈希完全一致，并采用冻结的 `fixed_nested_v1` 上下文。
- 为五种规定方法各执行三个 seed 的 pilot：每个 checkpoint 请求最多 2,000 环境步，实际因完整 episode 边界结束于 2,001–2,262 步；每个方法、任务和 K 只使用 5 个 validation query case。

pilot 结果只能用于诊断：

- PEARL full 的平均 validation VCSR 在 K=0/1/2/4/8 为 0.1833/0.1833/0.1833/0.2000/0.2000。
- K=4 时 full 相对 no-context 的任务级成对均值为 `+0.0167`，95% task-cluster bootstrap 区间为 `[0, 0.0667]`，未通过预注册的严格正下界条件。
- K=4 的同几何异规则 leave-one-pair-out 后验识别准确率为 0.25，95% 区间为 `[0, 0.5]`，未超过机会水平。
- 旧 taskbook 上现有 seed 2、step 40004 checkpoint 的固定嵌套诊断在 K=0/1/2/4/8 得到 validation VCSR 0.650/0.700/0.650/0.675/0.725；相对 no-context 的平均差值为 0/0.050/0/0.025/0.075。该结果只有一个 seed、两个未配对 validation 任务，不能替代新的后验适应证据。

### 中等规模诊断（seed 11，非正式后验适应证据）

为定位 smoke pilot 的失败来源，新增了明确标记为 `medium_diagnostic` 的全 meta-train 训练路径；它绕过正式 baseline 验证只用于诊断，绝不解封后验适应结论或 holdout。训练在完整 10 个 meta-train 任务上从同一可恢复状态推进到 20,037、50,051 和 80,115 环境步；80k 后未落盘的 100k 尾段因 replay/更新工作集持续扩张而停止，80k checkpoint 已完整保留。

- 每个预算点均在 4 个 meta-validation 任务上使用完整 20 个 query case，并以同一 checkpoint 比较 full 与 no-context；support-only 审计不读取 query。
- full 相对 no-context 的平均 VCSR 差在 K=4/K=8 时分别为：20k 的 `-0.0125/0.0000`，50k 的 `+0.0875/+0.1375`，80k 的 `+0.3500/+0.3750`。训练量确实决定 posterior 是否能转化为策略收益。
- 80k、K=8 时两个 `adversary_first` 任务的差值为 `+0.80/+0.75`，两个 `sut_first` 任务为 `-0.05/0.00`；K=4 时为 `+0.70/+0.75/-0.10/+0.05`。单侧偏置较 50k 缓解，但尚未消失。
- support 并不缺少规则相关事件：K=8 的 32 条 support 轨迹在 20k/50k/80k 分别有 `24/23/30` 条物理接触。80k 后验平均规则对距离为 `6.65`，平均 posterior 标准差为 `0.028`，表明模型已强烈区分任务，但还没有在所有规则上对称地利用这一信息。
- 因为该诊断只有一个训练种子、四个 validation 任务，且 80k 低于冻结的正式 1.5M 预算，结果只支持“训练不足是主因且存在残余单侧负迁移”的工程诊断，不支持后验适应验证通过。

机器可读产物见 [中等规模目录](../../results/pearl_learning/posterior_adaptation/medium_diagnostic/)、[50k 汇总](../../results/pearl_learning/posterior_adaptation/medium_diagnostic/diagnostic_step_50000.json) 和 [80k 汇总](../../results/pearl_learning/posterior_adaptation/medium_diagnostic/diagnostic_step_80000.json)。

未完成项是新 taskbook 的正式前置异质性验证、五种方法的冻结正式训练预算、每任务 20 个 validation query case，以及 validation 通过后的独立 holdout。机器可读状态见 [manifest.json](../../results/pearl_learning/posterior_adaptation/manifest.json)、[frozen_protocol.json](../../results/pearl_learning/posterior_adaptation/frozen_protocol.json)、[compact_results.json](../../results/pearl_learning/posterior_adaptation/compact_results.json)、[posterior_pair_audit.json](../../results/pearl_learning/posterior_adaptation/posterior_pair_audit.json) 和 [support_only_posterior_seed_11.json](../../results/pearl_learning/posterior_adaptation/support_only_posterior_seed_11.json)。

## 当前代码判断

现有工程已经具备 PEARL 的基本适应闭环：

- [evaluator.py](../../pearl_learning/src/evaluator.py) 在 support rollout 后调用 context encoder，query 阶段使用 `torch.no_grad()`，并比较适应前后的参数哈希。
- [context_encoder.py](../../pearl_learning/src/context_encoder.py) 输出后验均值与对角方差；空上下文使用标准正态先验。
- [metrics.py](../../pearl_learning/src/metrics.py) 的 `summarize()` 已输出平均 query return、平均 episode 长度、query 环境步数和首次严格有效关键场景的环境步数，并保留 VCSR、invalid rate、TTC、距离、首次严格有效关键 episode 和初始条件多样性。
- [dense_pearl_baseline.yaml](../../pearl_learning/configs/dense_pearl_baseline.yaml) 的正式配置是 K=0/1/2/5/10、每个 support episode 抽 32 个 transition、总上下文 256 个 transition。

现有证据不足以通过后验适应验证：

- 平均 VCSR 在 K=0/1/2/5/10 上为 0.588/0.575/0.588/0.600/0.575，不呈稳定提升。
- K=10 已执行 10 个 support episode，但后验容量最多覆盖 8 个 episode。
- `_sample_episode_context()` 会在每个 shot 重新抽取既有 episode 的 transition；因此 K 路径同时混入了新增 support 与旧 episode 重采样的影响。
- 当前正式 PEARL 结论来自一个选定训练 checkpoint，无法估计训练种子不确定性。

## 范围

本阶段只允许修改评估协议、任务/案例构造、指标汇总、审计输出和必要测试。禁止新增 router、expert、MoE loss、主动 support 选择、表示解耦辅助头或 RSS 约束。

当前可辩护的任务范围是 `merge-family cross-topology adaptation`。在没有新增 intersection、roundabout 等环境适配器前，不得扩写为任意道路逻辑场景泛化。

## 必须实现的工作

### 1. 冻结新的后验适应配置

新增独立协议配置，不覆盖 `dense_pearl_baseline.yaml`。正式评估固定：

```yaml
evaluation:
  shots: [0, 1, 2, 4, 8]
pearl:
  context_transitions_per_episode: 32
  context_sample_size_eval: 256
```

训练与评估的上下文容量必须一致支持 8 个 episode。任何超参数选择只使用 meta-validation；meta-test 在冻结选择后执行一次。

### 2. 改为固定嵌套上下文

在 [evaluator.py](../../pearl_learning/src/evaluator.py) 中实现以下协议：

1. support case 顺序在任务级冻结。
2. 每个 support episode 完成时，按由任务、case 和评估种子确定的种子抽取一次 32 个 transition，并缓存其索引或内容。
3. `C_K` 只能由前 K 个缓存块组成；`C_1 ⊂ C_2 ⊂ C_4 ⊂ C_8`。
4. 同一任务的所有 K 使用完全相同的 query cases 和 query 顺序。
5. 输出每个 K 的 support episode 数、各 episode 长度、累计 `B_K`、上下文 transition 数和抽样哈希。

K=0 必须严格使用 `N(0, I)`；不得用 validation/test 任务的任何 support 或 query 信息估计先验。

### 3. 构造同几何、异隐藏规则任务对

在 meta-validation 和 meta-test 中加入成对任务：地图、路线、拓扑和初始条件分布相同，只改变隐藏目标接触规则，例如：

```text
T_A = (same_geometry, adversary_first)
T_B = (same_geometry, sut_first)
```

要求：

- task ID、case ID 和随机种子继续满足 [casebook.py](../../pearl_learning/src/casebook.py) 的全局不重复约束。
- 任务对使用匹配的数值初始条件或严格相同的生成分布，具体方式写入 manifest。
- 隐藏规则只用于环境判定和训练后审计，不进入观测、router（本阶段尚无 router）或策略输入。
- train、validation、test 的任务哈希和 case 哈希保持不相交。

当前 meta-train 的 10 个任务对应 5 个物理地图及两类隐藏规则，可用于调试，但正式验证还需要独立的成对 validation/test 任务。扩展后的任务规模与生成规则必须在产物中完整记录。

### 4. 补齐指标和后验审计

扩展 [metrics.py](../../pearl_learning/src/metrics.py) 或专用后验适应汇总，使每个训练种子、任务和 K 至少包含：

- VCSR、target collision rate、critical rate、invalid rate；
- median min TTC、median min distance、mean query episode return；
- 首个严格有效关键场景的 query episode 序号；
- 到首个严格有效关键场景的累计 query 环境步数；
- 有效关键初始条件数量与多样性；
- posterior mean、log variance、variance、相对 K=0 与前一 K 的变化量；
- K 和真实 support 预算 `B_K`。

后验方差由 Product-of-Gaussians 产生，会随证据数机械收缩。因此本阶段只把它报告为模型后验统计量，不把它宣称为已校准认知不确定性，也不用于动态温度。

对同几何异规则任务对，报告后验距离、成对分类/检索准确率及其置信区间。分类器只能在 meta-validation 上拟合或选择，test 只做冻结评估。

### 5. 完成必要消融

在相同 checkpoint 训练预算、support/query cases 和评估协议下比较：

| 方法 | 用途 |
| --- | --- |
| PEARL full | 当前完整 dense 基线 |
| PEARL no-context | 始终使用先验，检验 support 必要性 |
| deterministic latent | 使用后验均值但去除采样，检验概率表示作用 |
| topology-only | 只依赖经审计的物理拓扑描述，不使用 support 后验 |
| context-only/no-topology | 屏蔽观测中的拓扑字段，检验交互证据与拓扑的互补性 |

现有 `topology_conditioned_pooled_sac` 不是严格匹配的 topology-only PEARL，不能替代上表消融。首轮实验关闭可选 disentangled representation，support selection 固定为 `fixed`，避免同时引入新变量。

### 6. 多训练种子与统计协议

- dense PEARL 正式训练至少 3 个独立种子；资源允许时使用 5 个。
- 所有方法共享冻结 taskbook、casebook、K、query cases 和预算定义。
- 以任务为统计单位，报告逐任务结果、训练种子分布和成对差值。
- 主结论使用层次化 bootstrap 或 mixed-effects model；不能把同一任务的 20 个 query case 当作 20 个独立任务。
- 同时绘制 K 轴和 `B_K` 轴的性能曲线，防止 episode 长度差异造成预算误读。

## 目标代码位置

优先扩展现有入口，不创建重复实验 wrapper：

- [evaluator.py](../../pearl_learning/src/evaluator.py)：固定嵌套上下文、逐 K 预算和后验审计。
- [metrics.py](../../pearl_learning/src/metrics.py)：query return 与累计步数汇总。
- [casebook.py](../../pearl_learning/src/casebook.py) 及现有 taskbook/casebook 构造路径：匹配任务对与不相交校验。
- [validation_freeze.py](../../pearl_learning/src/validation_freeze.py)：冻结选择与 test 前校验。
- `pearl_learning/configs/posterior_adaptation_protocol.yaml`：唯一的后验适应正式配置。
- `pearl_learning/tests/`：待新增的嵌套上下文、无泄漏、匹配任务对和指标测试。

## 必须通过的自动测试

- 对任意任务验证 `C_1 ⊂ C_2 ⊂ C_4 ⊂ C_8`，旧 episode 的 transition 索引不随 K 改变。
- 更改 query case 内容不会改变 posterior、support 顺序或上下文哈希。
- K=0 posterior 精确等于先验；K>0 的上下文 episode 数和 transition 数符合配置。
- 评估前后 `parameter_hash` 相同；对 actor、critic、encoder 做逐模块哈希检查。
- 同几何任务对的可观测静态拓扑完全一致，隐藏规则不同且不在观测字段中。
- taskbook/casebook 的 ID、seed 和 split 哈希校验通过。
- `mean_episode_return` 与累计 query 步数可由原始 episode records 精确重算。

## 后验适应验证通过条件

只有同时满足以下条件才标记为通过：

1. 协议审计证明无 query 泄漏、无梯度适应、固定嵌套上下文和预算记录正确。
2. 在独立任务和多个训练种子上，PEARL full 相对 no-context 的成对收益方向稳定，区间估计支持“support 有效”，而不是只依赖一个 K 或一个 checkpoint。
3. 同几何异规则任务在 K=0 不可由拓扑区分，K>0 后 posterior 能以高于机会水平的稳定结果区分，且分离变化与 query 表现变化相关。
4. context-only、topology-only 和 deterministic latent 消融足以排除“只靠静态拓扑”这一解释。
5. 结论在 K 和 `B_K` 两种预算视角下均成立，未隐藏负迁移任务。

具体最小效应量、置信区间和多重比较规则应在查看 meta-test 前写入 validation freeze manifest。不得在看到 test 结果后调整通过阈值。

若上述任一核心条件失败，后验适应验证状态为“未通过”。应停止 MoE 主线，定位是任务不可识别、上下文采样、训练不足还是 encoder 表达问题，并生成失败报告；不得用平均曲线或拓扑增益替代 support 证据。

## 交付物

```text
results/pearl_learning/posterior_adaptation/
  manifest.json
  frozen_protocol.json
  taskbooks/
  casebooks/
  metrics_by_seed_task_k.jsonl
  posterior_pair_audit.json
  statistical_summary.json
  compact_results.json
```

`manifest.json` 必须给出验证状态、代码提交、全部哈希、训练/评估种子、预注册判据、运行命令和缺失项。原始 episode records 可以分片保存，但紧凑结果必须能追溯到原始产物。

## 交付报告格式

```text
后验适应验证：PASS | FAIL | INCOMPLETE
代码改动：...
自动测试：...
正式训练种子：...
冻结产物与哈希：...
主要成对效应及区间：...
失败/负迁移任务：...
是否允许进入路由 MoE 工程验证：YES | NO
```
