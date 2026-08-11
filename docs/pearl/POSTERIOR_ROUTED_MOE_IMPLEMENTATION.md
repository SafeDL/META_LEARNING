# 后验路由 MoE 工程实现

## 实现目标

在后验适应验证通过后，以最小可解释改动实现 actor-only、task-level、posterior-routed residual MoE，并通过接口、梯度、checkpoint、无泄漏和路由日志测试。

本阶段的完成标准是“实现正确且可进入正式实验”，不是“性能优于 dense PEARL”。任何 smoke test 上的收益都不能写成研究结论。

## 当前代码判断

- [networks.py](../../pearl_learning/src/networks.py) 的 `GaussianActor` 将 `[observation, latent]` 输入两层 MLP，直接输出动作均值和 log standard deviation；没有共享主干、expert 或 router。
- [pearl_agent.py](../../pearl_learning/src/pearl_agent.py) 使用 dense twin critics。`actor_z = expanded_z.detach()` 明确阻断 actor loss 到 context encoder 的梯度，这是本阶段必须保留的训练边界。
- [collector.py](../../pearl_learning/src/collector.py) 当前只向 `agent.act()` 传 observation 和 sampled latent，没有任务级 route context。
- [checkpoint.py](../../pearl_learning/src/checkpoint.py) 通过 `agent.state_dict()` 保存模块，但加载时依赖 `agent.actor.backbone[0]` 推断观测维度；该假设不适用于 MoE。
- [observation.py](../../pearl_learning/src/observation.py) 的 37 维观测包含动态量和拓扑量。`adversary_route_remaining`、`sut_route_remaining` 会随 step 改变，不能作为任务级静态路由输入。
- [adapters/base.py](../../pearl_learning/src/adapters/base.py) 中 branch 数由图节点/边数量近似，speed limit 固定为 30，conflict zone 固定为 1。这些字段在修正语义前不能直接作为“物理静态描述符”。

## 方法契约

### 1. 路由输入

对任务 T 和 support 数 K，定义：

```text
r_K = concat(h_T, stop_gradient(mu_K), stop_gradient(log_var_K))
w_K = TopKNormalize(router(r_K))
```

- `h_T` 是单独定义并带 schema 的物理静态描述符，不是从任意时刻的完整 observation 切片得到的临时向量。
- `mu_K`、`log_var_K` 来自 PEARL posterior。router 和 experts 接收 actor loss 梯度，但 actor loss 不得借此更新 context encoder。
- v1 可把 `log_var_K` 当作普通特征；不得让它直接控制 routing temperature，因为当前方差尚未校准。
- task ID、geometry ID、logical type one-hot、split、隐藏接触规则、support outcome 标签和 query 数据均禁止输入 router。

### 2. 静态描述符

新增一个唯一的描述符 schema，例如 `merge_physical_task_descriptor_v1`。首版只纳入语义已核实且任务内不变的连续物理量：

- adversary/SUT lane count；
- merge length；
- conflict radius；
- adversary/SUT route curvature。

branch count 只有在 [adapters/base.py](../../pearl_learning/src/adapters/base.py) 改为真实入口/出口分支定义并通过地图测试后才能加入。硬编码 speed limit 和 conflict zone 常量不进入 v1。route remaining 永远不进入 task-level 描述符。

描述符必须：

- 在环境/任务构造时计算一次并冻结；
- 使用配置中显式定义的归一化尺度；
- 记录字段名、原始值、归一化值、schema 和 content hash；
- 对同几何异规则任务完全相同。

### 3. Actor 结构

使用“共享主干 + 残差专家 + 统一 Gaussian head”：

```text
x = concat(observation, latent)
u_shared = shared_trunk(x)
u = u_shared + sum_e(w_K[e] * residual_expert_e(x))
(mean, log_std) = gaussian_head(u)
action ~ TanhNormal(mean, exp(log_std))
```

critic 保持现有 dense twin critics，context encoder 保持现有结构。不要在同一阶段修改 critic、context posterior、support selection 或 optional disentangled representation。

建议分两级配置：

- 正确性/调试：2 experts、全部 soft routing，以最小规模验证梯度和日志。
- 机制验证候选：4 experts、Top-2 routing，在 validation 上冻结后用于正式实验。

Top-2 应对 softmax 权重取 top-k、屏蔽其余专家并重新归一化。禁止把训练前匿名专家硬编码为某种道路类型或隐藏规则。

### 4. 任务级路由生命周期

`posterior_version` 是路由更新的唯一时钟：

1. K=0 用 prior posterior 和任务静态描述符计算 `w_0`。
2. 新 support episode 完成并重新推断 posterior 后，版本加一，计算新的 `w_K`。
3. 一次 support 或 query episode 内 route 固定；不得逐 step 根据 observation 重算。
4. 同一 posterior version 的重复确定性 query 必须得到相同 route。
5. 环境收集阶段可在 `torch.no_grad()` 下缓存 route；训练 actor 更新时必须重新执行 router 前向，使 router 参数获得梯度。

训练 replay 已记录 task ID，但 task ID 只能用于在可信的 taskbook 映射中查找物理描述符，不能被编码为数值或 one-hot 后输入网络。

## 必须实现的代码

### 1. 网络模块

优先新增 `pearl_learning/src/moe.py`，避免让 [networks.py](../../pearl_learning/src/networks.py) 同时承载 dense 与 MoE 的全部细节。至少提供：

- `PosteriorRouter`：校验输入维度、输出 logits、soft weights、top-k mask 和归一化 weights。
- `PosteriorRoutedMoEActor`：保持与 `GaussianActor.sample()` 一致的 Tanh-Gaussian 语义。
- `RoutingOutput`：包含 logits、weights、entropy、top-k indexes 和 descriptor hash 所需字段。

维度、expert 数、top-k 和网络宽度只从配置读取。非法 top-k、非有限输入或权重和不为 1 时立即报错，不做静默 fallback。

### 2. Agent 与优化器

修改 [pearl_agent.py](../../pearl_learning/src/pearl_agent.py)：

- 按显式 `actor_architecture: dense | posterior_routed_moe` 构建 actor。
- agent 直接保存 `observation_dim`、`action_dim`、`latent_dim` 和 architecture metadata，checkpoint 不再反查 actor 内部层。
- actor optimizer 覆盖 shared trunk、experts、Gaussian head 和 router。
- router 使用 `mu.detach()`、`log_var.detach()`；experts 和 router 从 actor objective 获得梯度。
- v1 总损失只增加一个 load-balancing term：`actor_loss + lambda_balance * balance_loss`。
- routing consistency 和 expert diversity 只作为关闭的可选研究项，未进入消融前不默认启用。
- dense 分支的数值语义、默认配置和现有测试保持不变。

### 3. 收集器、训练器和评估器

修改 [collector.py](../../pearl_learning/src/collector.py)、[pearl_trainer.py](../../pearl_learning/src/pearl_trainer.py) 和 [evaluator.py](../../pearl_learning/src/evaluator.py)：

- 用清晰的 route context 参数传递 task-level route，不把 route 偷塞进 observation 或 latent。
- 训练更新按 replay 中的任务分组，使用相应的 posterior stats 和静态描述符计算可微 route。
- 收集和 query 记录 `posterior_version`、router schema 与 route hash。
- 固定后验适应的 nested context 和 support/query 协议。
- 评估输出完整 router audit，不只保留平均 entropy。

如果统一接口会导致 dense 路径充满空 route 参数，应在 agent 层封装 architecture 分支；不要保留废弃 wrapper 或自动猜测模型类型。

### 4. Checkpoint 与 provenance

修改 [checkpoint.py](../../pearl_learning/src/checkpoint.py)：

- manifest 明确保存 architecture schema、descriptor schema、expert count、top-k、router input fields 和 load-balance weight。
- `state_dict()`、`load_state_dict()` 和 `parameter_hash()` 覆盖 router、全部 experts、head 和 optimizer state。
- 加载时比较显式 agent metadata，不再依赖 `actor.backbone[0]`。
- dense checkpoint 由 dense architecture 显式加载；MoE checkpoint 由 MoE architecture 显式加载。架构不匹配直接拒绝，不实现隐式转换或缺失权重 fallback。

### 5. 配置

新增一份路由 MoE 配置，建议结构：

```yaml
networks:
  actor_architecture: posterior_routed_moe
  moe:
    descriptor_schema: merge_physical_task_descriptor_v1
    num_experts: 2
    top_k: 2
    routing: soft
    router_hidden_sizes: [64, 64]
    expert_hidden_size: 128
    load_balance_weight: 0.01
```

以上数值是 smoke-test 起点，不是默认宣称的最优值。正式候选配置只能依据 meta-validation 冻结。

## 路由日志契约

每个训练种子、task、K/posterior version 至少记录：

- posterior mean、log variance；
- 静态描述符 schema、字段、归一化值和 hash；
- router logits、归一化 weights、top-k indexes、entropy；
- batch/任务级 expert load、load coefficient of variation；
- actor 主损失、balance loss；
- route 计算时是否启用梯度；
- 参数哈希和 query-free 标记。

原始路由日志不得包含隐藏规则。按规则或逻辑类型分组只能作为训练后的审计视图生成，并标明 `posthoc_only: true`。

## 必须通过的自动测试

### 网络与数值

- weights 非负、和为 1，非 Top-k 专家权重精确为 0。
- deterministic 与 stochastic action 的形状、范围和 log probability 与 dense actor 契约一致。
- 对相同输入、seed、posterior version，route 可复现。
- 非有限 posterior/descriptor、schema 不匹配、非法 top-k 直接报错。

### 梯度

- actor update 后 shared trunk、被激活 experts 和 router 均有有限非零梯度。
- actor loss 不向 context encoder 传播梯度。
- critic update 不更新 router/expert；actor update 不意外更新 critic。
- 未激活 expert 的梯度行为与 Top-k 设计一致，且日志能观察到长期无梯度专家。

### 生命周期与泄漏

- 同一 episode 内 route hash 不变。
- route 只在 posterior version 变化后重算；更改 query 内容不改变 route。
- 同几何异规则任务在 K=0 的静态描述符相同。
- 静态描述符输入中不存在 ID、split、隐藏规则或动态 route remaining。

### Checkpoint

- dense 和 MoE 各自完成 save-load round trip，恢复后 action、route、参数哈希一致。
- architecture 或 descriptor schema 不匹配时明确失败。
- resume 后 optimizer、RNG 和 trainer state 可继续训练。
- 评估前后包括 router/experts 在内的完整参数哈希不变。

### 回归与 smoke

- dense 配置的现有单元/集成测试全部通过。
- 2-expert soft routing 完成短训练，无 NaN、无权重越界、无单 expert 永久独占。
- 固定 posterior 的重复 query 路由一致；新增 support 后允许路由改变。

## 工程实现通过条件

1. 上述自动测试全部通过，dense 基线无行为回归。
2. checkpoint 和 manifest 能完整恢复并审计 MoE 架构。
3. router 没有任何 task label、隐藏规则或 query 泄漏路径。
4. 路由严格是 task-level，且只随 posterior version 改变。
5. router 和 experts 获得预期梯度，context encoder 保持现有 actor-loss 梯度边界。
6. 至少两个训练种子的 smoke run 无 NaN、无确定性 collapse；这只验证工程稳定性，不用于性能主张。

若出现 expert collapse，可以在不改变主结构的前提下调节初始化、balance weight 或 batch/task mixing；不得同时加入 consistency、diversity、uncertainty temperature 三个损失掩盖根因。

## 交付物

```text
results/pearl_learning/posterior_routed_moe/
  manifest.json
  architecture_contract.json
  descriptor_schema.json
  smoke_metrics_by_seed.jsonl
  router_audit.jsonl
  checkpoint_roundtrip.json
  test_report.json
```

## 交付报告格式

```text
后验路由 MoE 工程实现：PASS | FAIL | INCOMPLETE
新增架构与配置：...
梯度边界测试：...
路由生命周期/泄漏测试：...
checkpoint round trip：...
smoke seeds 与 collapse 检查：...
dense 回归：...
是否允许进入机制验证：YES | NO
```

## 2026-08-10 执行结果

```text
后验路由 MoE 工程实现：PASS
新增架构与配置：actor-only、task-level posterior-routed residual MoE；2 experts、soft routing、共享主干、残差 experts、统一 Gaussian head；critic 与 context encoder 结构不变。
梯度边界测试：router、共享主干、Gaussian head 和两个 soft-routed expert 均得到有限非零梯度；actor loss 对 context encoder 的梯度为 0；critic/actor 两个优化阶段互不修改对方参数。
路由生命周期/泄漏测试：真实 few-shot 评估的 K=0/1/2/4/8 均在同一 posterior version 的重复 query 上得到相同 route hash；描述符 hash 跨版本不变；原始 router 日志不含 task ID、规则标签、split、logical type 或动态 route remaining。
checkpoint round trip：dense 与 MoE 均通过；MoE 的 action、route、参数、optimizer、RNG 和 trainer state 完整恢复；架构不匹配会显式拒绝。
smoke seeds 与 collapse 检查：seed 17/29 均完成真实 MetaDrive 短训练，无 NaN；两个 expert 梯度均非零，权重范围分别为 [0.499592, 0.500408] 和 [0.497916, 0.502084]，未出现单 expert 独占。
dense 回归：完整测试 53 passed，其中原有 39 项 dense/协议回归全部保留通过。
是否允许进入机制验证：YES（仅允许工程与机制验证；不等于性能优越性或迁移性结论）。
```

可审计交付物位于 [posterior_routed_moe](../../results/pearl_learning/posterior_routed_moe)。核心实现见 [moe.py](../../pearl_learning/src/moe.py)、[pearl_agent.py](../../pearl_learning/src/pearl_agent.py)、[collector.py](../../pearl_learning/src/collector.py)、[pearl_trainer.py](../../pearl_learning/src/pearl_trainer.py)、[evaluator.py](../../pearl_learning/src/evaluator.py) 和 [checkpoint.py](../../pearl_learning/src/checkpoint.py)。配置为 [posterior_routed_moe.yaml](../../pearl_learning/configs/posterior_routed_moe.yaml)，专项测试为 [test_routed_moe.py](../../pearl_learning/tests/test_routed_moe.py)。

边界说明：后验适应验证的正式统计状态仍是 `INCOMPLETE`。本次路由 MoE 是在用户审阅“物理可控性导致的单侧差异”后明确授权开展的工程实现例外。该例外不会把后验适应验证改写为通过，也不会授权正式性能、迁移性或优越性主张；这一事实已写入工程审计 manifest。

历史 checkpoint 边界：路由 MoE 工程实现前生成的 dense checkpoint 没有 architecture metadata，原文件与结果均保持不变。新加载器按本阶段的硬契约拒绝猜测其架构，不提供隐式兼容或缺失字段 fallback；复核这些历史结果应使用其 manifest 记录的旧 Git commit，或另做保留源文件 hash 的显式迁移。新生成的 dense/MoE checkpoint 均已通过严格 round trip。
