# Meta-RL Gate 3 Gamma-only FiLM Critic 执行计划

> 目的：在 Gate 3 Stage A 已通过、Stage `B_Q` 仍失败的证据基础上，只修改
> Critic 的 latent conditioning 方式，检验旧 FiLM Critic 的 `beta(z)` 加性路径是否
> 造成了与 action 无关的 task-specific Q offset。
>
> 本文只规定下一轮如何实施。本次新增本文档，不修改代码、配置、历史结果或既有状态报告，
> 也不启动训练。

## 1. 当前证据与唯一问题

上一轮 `terminal_stratified_v1 + Latent-FiLM Critic` 的 Gate 3 v4 结果为：

| Task | `D_cw` | `R_sep` | `critic_q_grid.argmax_action_distance_mean` | 首个失败阶段 |
| --- | ---: | ---: | ---: | --- |
| `meta_train_lane_drop_24__logical_order_adversary_first` | 31.9510 | 0.6867 | 0.0 | `B_Q` |
| `meta_train_lane_drop_24__logical_order_sut_first` | 31.7938 | 0.6841 | 0.0 | `B_Q` |

因此当前已经证明：

```text
correct/wrong context
  → 不同 posterior（Stage A PASS）
  → 相同 Critic argmax action（Stage B_Q FAIL）
```

旧 FiLM Critic 为：

\[
Q(s,a,z)=\operatorname{head}\left((1+\gamma(z))h(s,a)+\beta(z)\right).
\]

其中 `beta(z)` 不依赖 action，可能形成：

\[
Q(s,a,z)=Q_0(s,a)+B(z),
\]

使不同 latent 显著改变 Q 的绝对值，却不改变动作排序。本轮只检验这一 shortcut 假设，
不把已有结果解释为 Context Encoder、Actor、reward、Task 或 replay sampler 的失败。

上一轮已经对 checkpoint replay buffer 完成零环境步审计；RL batch 中
`terminal OR conflict-near` proxy 约为 5.6%–6.4%。该结果保留为背景诊断，
不授权本轮修改 replay distribution。

## 2. 冻结范围与实验边界

以下内容全部保持上一轮 resolved config 原值：

- Context Encoder、PoG 聚合、latent dim、unit-normal prior、KL beta；
- Dense Actor 及其优化方式；
- reward、Task、环境、screened casebook、taskbook、query oracle；
- `terminal_stratified_v1` context sampler 和 RL replay sampler；
- actor/critic/context 学习率及其他 SAC 超参数；
- seed 1、命令行 `--max-env-steps 20000` 和 mechanism-gate 运行方式；
- Scenario Encoder、Structure-Aware Prior、MoE 及其他消融开关。

本轮只允许一组新训练，不补跑 seed、不追加预算、不并行试验其他 Critic，不从旧
FiLM checkpoint 恢复。训练脚本按 episode 边界停止，因此 summary 中实际环境步数可能
略高于 20k；预算契约仍以命令行 `--max-env-steps 20000` 为准。

## 3. Gamma-only Critic 与 checkpoint 隔离

### 3.1 保留旧架构

现有 `LatentFiLMCritic` 和枚举 `latent_film_dense` 必须完全冻结，包括：

- 类的计算图、参数名称、维度和初始化；
- `state_dict` 键和历史 checkpoint 加载行为；
- 已有配置、训练结果和 causal audit 的可复核性。

不得将旧类原地改成 gamma-only，也不得让旧枚举静默映射到新实现。

### 3.2 新架构接口

在 `pearl_learning/src/networks.py` 新增 `LatentGammaOnlyFiLMCritic`，在
`PEARLAgent` 中新增独立枚举：

```text
latent_film_gamma_only
```

三个 Critic 枚举必须显式、互斥地映射：

| 枚举 | 实现 |
| --- | --- |
| `dense` | 现有 concat Dense Critic |
| `latent_film_dense` | 现有 gamma+beta FiLM Critic |
| `latent_film_gamma_only` | 新 gamma-only FiLM Critic |

checkpoint 的 `architecture_metadata.critic_architecture` 继续记录枚举字符串。
旧 FiLM 与 gamma-only checkpoint 双向加载时，必须在加载参数前因 metadata 不一致而拒绝；
不提供权重迁移、参数补齐或兼容回退。

### 3.3 严格单变量网络定义

设 `F = hidden_sizes[-1]`，`H = max(16, latent_dim * 4)`。新 Critic 的
state-action trunk、head、modulator 隐藏深度和隐藏宽度均与旧 FiLM Critic一致：

```python
class LatentGammaOnlyFiLMCritic(nn.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        latent_dim,
        hidden_sizes,
    ):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("Gamma-only FiLM critic needs a non-empty hidden size list")

        self.feature_dim = int(hidden_sizes[-1])
        self.trunk = mlp(
            observation_dim + action_dim,
            list(hidden_sizes),
            self.feature_dim,
        )

        # 保留旧 FiLM modulator 的两个隐藏层 H → 2F；仅将最终输出从 2F 改为 F。
        self.modulator = mlp(
            latent_dim,
            [max(16, latent_dim * 4), 2 * self.feature_dim],
            self.feature_dim,
        )
        last_layer = self.modulator[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

        self.head = nn.Linear(self.feature_dim, 1)

    def forward(self, observation, action, latent):
        features = self.trunk(torch.cat([observation, action], dim=-1))
        gamma = torch.tanh(self.modulator(latent))
        return self.head((1.0 + gamma) * features)
```

新计算图严格为：

\[
\boxed{Q(s,a,z)=\operatorname{head}\left((1+\tanh(\gamma(z)))h(s,a)\right)}.
\]

不得添加 latent-only bias、residual、skip connection 或其他等价的加性路径。
最终层零初始化意味着初始 `gamma(z)=0`，所以训练开始时新 Critic 等价于普通
`Q(s,a)`。不同 latent 的初始 Q 完全相同是预期行为，不是初始化测试失败。

首次反向传播时，modulator 早期层和 latent 的梯度可以为零，但最终零初始化层必须收到
有效梯度；该层更新后，后续更新应建立对 latent 的敏感性。因此不得用“第一个 update 的
`critic_latent_gradient_norm > 0`”作为硬性断言，应检查后续 update 是否仍持续为零。

## 4. 新配置与固定路径

新增配置：

```text
pearl_learning/configs/merge_method_flow_gate3_context_sampling_gamma_only_film_critic.yaml
```

它继承上一轮配置：

```yaml
extends: merge_method_flow_gate3_context_sampling_film_critic.yaml

project:
  output_root: results/pearl_learning/merge_method_flow_gate3_context_sampling_gamma_only_film_critic

experiment:
  method_variant: vanilla_gate3_context_sampling_gamma_only_film_critic

networks:
  critic_architecture: latent_film_gamma_only
```

相对上一轮 resolved config，只允许以下差异：

```text
project.output_root
experiment.method_variant
networks.critic_architecture
```

固定运行名与结果位置：

```text
run-name:
  gate3_context_sampling_gamma_only_film_critic_20k_seed1

checkpoint directory:
  results/pearl_learning/merge_method_flow_gate3_context_sampling_gamma_only_film_critic/
    mechanism_gate/gate3_context_sampling_gamma_only_film_critic_20k_seed1/

causal audit directory:
  results/pearl_learning/merge_method_flow_gate3_context_sampling_gamma_only_film_critic/
    gate_3_causal_audit/
```

## 5. 训练前验证

先运行包含以下新测试的定向测试与完整回归测试：

```powershell
conda run -n metadrive python -m unittest `
  pearl_learning.tests.test_method_flow_v2.MethodFlowV2Tests.test_gamma_only_film_critic_structure_and_zero_initialization `
  pearl_learning.tests.test_method_flow_v2.MethodFlowV2Tests.test_gamma_only_film_critic_checkpoint_isolation `
  pearl_learning.tests.test_method_flow_v2.MethodFlowV2Tests.test_gate3_gamma_only_config_changes_only_critic

conda run -n metadrive python -m unittest pearl_learning.tests.test_method_flow_v2
```

测试名可按现有测试类命名约定微调，但必须覆盖第 8 节的全部验收条件。

确认 CUDA 可用：

```powershell
conda run -n metadrive python -c "import torch; assert torch.cuda.is_available(), 'CUDA is required'; print(torch.cuda.get_device_name(0))"
```

在测试中解析两个继承配置并比较 resolved dict。除 output/provenance 字段外，删除
`critic_architecture` 后的 `networks` 必须完全相等；环境、reward、PEARL、sampler、SAC、
meta-training、case、scenario prior/representation 和 MoE 等冻结 section 必须逐项相等。
任何额外差异都应阻止训练启动。

## 6. 唯一允许的训练与 v4 审计

训练命令：

```powershell
conda run -n metadrive python -m pearl_learning.scripts.train_pearl `
  --config pearl_learning/configs/merge_method_flow_gate3_context_sampling_gamma_only_film_critic.yaml `
  --taskbook results/pearl_learning/merge_method_flow_logical_order_interpolated/taskbooks `
  --casebook-root results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/mechanism_assets_screened `
  --critical-thresholds results/pearl_learning/merge_method_flow_pilot/v2_assets/calibration/critical_thresholds.json `
  --seed 1 `
  --max-env-steps 20000 `
  --run-name gate3_context_sampling_gamma_only_film_critic_20k_seed1 `
  --mechanism-gate
```

训练结束后，先核对 `config_resolved.json`、checkpoint manifest 和 training summary：

- `critic_architecture == latent_film_gamma_only`；
- Actor 仍为 `dense`；
- sampler 仍为 `terminal_stratified_v1`；
- prior 仍为 unit normal；
- Scenario Encoder 与 MoE 均关闭；
- seed 为 1，命令行预算为 20k，checkpoint 可恢复状态完整。

然后执行现有 Gate 3 v4 causal audit：

```powershell
conda run -n metadrive python -m pearl_learning.scripts.audit_gate3_vanilla_pearl_mechanism `
  --config pearl_learning/configs/merge_method_flow_gate3_context_sampling_gamma_only_film_critic.yaml `
  --checkpoint results/pearl_learning/merge_method_flow_gate3_context_sampling_gamma_only_film_critic/mechanism_gate/gate3_context_sampling_gamma_only_film_critic_20k_seed1/best_model.pt `
  --taskbook results/pearl_learning/merge_method_flow_logical_order_interpolated/taskbooks `
  --casebook-root results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/mechanism_assets_screened `
  --critical-thresholds results/pearl_learning/merge_method_flow_pilot/v2_assets/calibration/critical_thresholds.json `
  --output results/pearl_learning/merge_method_flow_gate3_context_sampling_gamma_only_film_critic/gate_3_causal_audit `
  --task-id meta_train_lane_drop_24__logical_order_adversary_first `
  --task-id meta_train_lane_drop_24__logical_order_sut_first `
  --shots 1 2 4 `
  --oracle-audit results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/gate_3_query_oracle_screened/gate3_query_oracle_audit.json
```

核心读取字段保持：

```text
tasks.<task-id>.shots.4.stage_b_diagnostics.critic_q_grid.argmax_action_distance_mean
```

正式 verdict 读取：

```text
gate3_causal_chain_gate_v4.json
  .stages.stage_a_context_to_posterior.status
  .stages.stage_b_q_posterior_to_critic_action_preference.status
```

## 7. 判定顺序与停止条件

v4 阈值不得因新结果改变：

- Stage A：两个 Task 均满足 `D_cw >= 0.5` 且 `R_sep >= 0.25`；
- Stage `B_Q`：两个 Task 均满足
  `critic_q_grid.argmax_action_distance_mean >= 0.10`；
- `D_Q > 0` 但 `< 0.10` 只能报告为改善，正式 Stage `B_Q` 仍为 FAIL。

只根据首个失败阶段决策：

| 结果 | 本轮结论 | 唯一允许的后续动作 | 禁止动作 |
| --- | --- | --- | --- |
| Stage A FAIL | gamma-only 训练伴随 representation/gradient regression，不能判断 beta 假设 | 停止并核查训练诊断 | 判断 B_Q、改 sampler/Actor、追加训练 |
| A PASS、B_Q FAIL | gamma-only 未达到 Critic action-preference 门槛 | 对新 checkpoint 做一次零环境步 replay-signal audit | 修改 replay、reward、预算或其他网络 |
| A PASS、B_Q PASS | beta-shortcut 假设得到支持，Critic 已形成 task-dependent action preference | 记录现有 B_pi/C 顺序审计结果并停止本轮 | 修改 Actor、补 seed、扩大预算、进入 Gate 4 |

如果 B_Q FAIL，使用新 checkpoint 的 buffer 执行：

```powershell
conda run -n metadrive python -m pearl_learning.scripts.audit_gate3_critic_replay_signal `
  --checkpoint results/pearl_learning/merge_method_flow_gate3_context_sampling_gamma_only_film_critic/mechanism_gate/gate3_context_sampling_gamma_only_film_critic_20k_seed1/best_model.pt `
  --output results/pearl_learning/merge_method_flow_gate3_context_sampling_gamma_only_film_critic/gate_3_causal_audit/gate3_critic_replay_signal_audit.json `
  --sampled-batches 128
```

该审计必须保持 `environment_steps: 0`、`training_updates: 0`，只读取 checkpoint 的
`trainer_state.buffers`，不得据此在同一轮直接修改 sampler。

## 8. 测试与验收清单

### 8.1 结构和初始化

- 空 `hidden_sizes` 明确报错；
- gamma-only trunk 和 head 的结构与旧 FiLM Critic一致；
- modulator 的两个隐藏层宽度保持 `H`、`2F`，最终输出维度为 `F`；
- state dict 中不存在 beta 模块或 beta 参数；
- modulator 最终层 weight、bias 全零；
- 初始 `gamma` 为零且经 `tanh` 有界；
- 相同 `(s,a)`、不同 `z` 的初始化 Q 完全相同。

### 8.2 梯度行为

- 构造确定性的非退化 loss，首次反向传播时最终零初始化层获得有限、非零梯度；
- 首次 optimizer step 后最终层不再全零；
- 使用不同 latent 和非退化 state-action feature 的后续前向/更新能够产生 latent sensitivity；
- 允许第一个 update 的 `critic_latent_gradient_norm == 0`，但测试应验证后续 update 不持续为零。

### 8.3 兼容性

- 旧 `latent_film_dense` 的 state-dict 键、形状和计算行为保持不变；
- 旧 FiLM checkpoint 在旧枚举下正常 round-trip；
- gamma-only checkpoint 在新枚举下正常 round-trip；
- 旧 FiLM checkpoint 加载到 gamma-only agent 时提前拒绝；
- gamma-only checkpoint 加载到旧 FiLM agent 时提前拒绝；
- 既有 Dense checkpoint 向后兼容行为不变。

### 8.4 配置与回归

- 新配置相对上一轮只改变 Critic 架构及 output/provenance 字段；
- v3/v4 Gate schema、阈值、41 点动作网格和历史 JSON 均不变；
- 完整测试通过：

```powershell
conda run -n metadrive python -m unittest pearl_learning.tests.test_method_flow_v2
```

## 9. 本轮完成定义

实现阶段只有同时满足以下条件才算完成：

1. 新旧 Critic 枚举和 checkpoint 严格隔离，旧 FiLM 实现未发生变化；
2. gamma-only 结构、零初始化、梯度、配置冻结和 checkpoint 测试全部通过；
3. 唯一一次 20k、seed 1 训练完成且 provenance 完整；
4. v4 audit 使用冻结输入并输出独立结果；
5. 明确报告两个 Task 的 Stage A 指标、`D_Q` 和首个失败阶段；
6. 严格执行停止条件，不在同一轮引入 Actor、sampler、reward、Task 或预算改动；
7. 不修改既有状态报告、历史规划和历史结果目录；只有后续单独授权时才更新状态汇总。
