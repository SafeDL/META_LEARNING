# 当前层级地图感知元强化学习方案：稳定化技术路线与代码修改建议

**项目：** 元强化学习的场景挖掘  
**代码仓库：** `SafeDL/META_LEARNING`  
**审查基线：** `main @ 4d807a8ea0b302ae89c0c5e47ebd982cd3d70c59`  
**日期：** 2026-08-21

---

## 0. 本文档的目的

当前方案已经经过多轮讨论，继续频繁更换 PEARL / PPO / SAC 的组合会带来三个问题：

1. **论文叙事不稳定**：审稿人难以判断真正创新来自何处；
2. **代码重构成本过高**：容易把时间消耗在优化器替换，而非科学问题验证；
3. **实验变量过多**：最终性能变化可能无法归因于 meta-learning、hierarchy、map encoder 还是具体 RL optimizer。

因此，从本文件开始，建议将主技术路线冻结。后续除非关键 Scientific Gate 明确失败，否则**不再改变主框架和主优化算法**。

本文档同时兼顾三个目标：

- **创新性**：足以支撑一篇围绕 few-shot autonomous-driving testing / scenario mining 的论文；
- **简洁性**：避免无必要的大型网络、复杂混合 SAC、全网络 test-time fine-tuning；
- **可行性**：能够在单张 RTX 4090 + MetaDrive 环境下完成 POC、消融和主体实验。

---

# 1. 最终冻结的技术路线

## 1.1 方法定位

建议最终统一称为：

> **Hierarchical Map-Aware Latent-Context Meta-RL for Few-Shot Autonomous Driving Testing**

中文可表述为：

> **面向少样本自动驾驶测试的层级地图感知潜变量上下文元强化学习方法**

在论文中说明其 **latent task inference 受到 PEARL 启发（PEARL-inspired）**，但**不要直接把当前完整方法称为原始 PEARL**。

原因是当前方法已经对 PEARL 做了实质性改变：

- task 不再是道路几何，而是未知 SUT 行为/脆弱性；
- map 是显式已知条件，不进入 latent；
- context unit 从 timestep transition 改为 complete simulator test；
- posterior 使用 trajectory-level evidence；
- policy 从单层 policy 变成 episode-level Outer + timestep-level Inner；
- Outer 是混合动作策略；
- latent 同时影响两种时间尺度的决策。

因此，最准确的说法是：

\[
\boxed{
\text{PEARL-inspired latent-context Meta-RL}
+
\text{hierarchical testing policy}
+
\text{explicit map conditioning}
}
\]

而不是：

\[
\text{standard PEARL}
\]

---

## 1.2 算法选择冻结

### Meta inference：保留当前 \(q_\phi(z|C)\)

继续使用当前：

\[
q_\phi(z|C_K)
\]

其中：

\[
C_K=\{e_1,e_2,\ldots,e_K\}
\]

每一个 \(e_k\) 对应一个**完整 simulator test episode**。

\(z\) 的语义固定为：

> **当前未知 SUT 在当前交互任务分布中的 vulnerability / failure-response latent representation。**

不允许把以下信息作为模型输入：

- `sut_id`
- `profile_id`
- `algorithm_id`
- IDM 参数
- controller one-hot

这些信息只能用于实验分组和 ground-truth analysis。

---

### Outer：继续使用当前 Hybrid PPO，不改成 Hybrid SAC

冻结为：

\[
\boxed{
\pi_{\text{scene}}(x_k\mid g_G,z_k,H_k)
}
\]

其中 Outer action 是：

\[
x_k=(c_k,\mathbf{x}_k,o_k)
\]

包含：

- discrete conflict / route candidate；
- continuous initial configuration；
- discrete adversarial option。

当前 `meta_testing/policy/scene_policy.py` 已经直接实现：

- categorical candidate head；
- tanh-Gaussian continuous head；
- categorical option head；
- value head。

因此现阶段**保留 PPO**。

### 为什么不再次改成 Hybrid SAC？

不是因为 SAC 不可行，而是因为它不是当前论文需要解决的问题。

如果为了“PEARL 原来使用 SAC”而把 Outer 改成 Hybrid SAC，需要额外处理：

- discrete + continuous entropy；
- mixed-action critic；
- discrete action 的 reparameterization / marginalization；
- target value；
- hybrid replay semantics；
- 更多超参数与稳定性问题。

这些都属于**新的算法工程变量**，并不会直接增强论文最关键的创新：

> “少量测试能否识别未知 SUT 的脆弱性，并指导后续测试更快发现失效？”

因此：

\[
\boxed{
\text{不要为了算法名称统一而重写 Outer optimizer}
}
\]

Outer PPO 是层级策略的一个 base RL optimizer；Meta-RL 的“Meta”来自跨 task 的 latent inference 和 few-shot posterior update，而不是要求所有 policy 都必须由 SAC 优化。

---

### Inner：继续使用 SAC

冻结为：

\[
\boxed{
\pi_{\text{adv}}
(a_t\mid s_t,\text{map},z_k,x_k)
}
\]

原因：

- action 是连续控制；
- timestep 数多；
- replay buffer 可显著提升样本利用率；
- 当前代码已经完成 actor、twin critics、target critics、temperature 和 soft update。

因此 Inner SAC 不需要更换。

---

## 1.3 Joint training 的定位

当前代码定义：

```text
inner_pretrain → posterior → outer → joint
```

建议修改为：

```text
inner_pretrain
    ↓
posterior
    ↓
outer
    ↓
optional light joint calibration
```

即：

\[
\boxed{\text{Joint 不再作为必须的大规模第四阶段}}
\]

Joint 只在以下情况出现时启用：

- posterior 能预测 vulnerability，但策略利用 \(z\) 的效果弱；
- `z_correct` 与 `z_swapped` 的策略行为差异不足；
- Outer/Inner 对 posterior distribution shift 敏感。

建议 joint calibration 只做低学习率、少量 alternating updates，而不是重新联合训练所有模块。

---

# 2. 当前研究的科学问题应固定为什么？

最终论文的核心问题不要写成：

> “我们提出一个比 PPO / SAC 更强的强化学习算法。”

也不要写成：

> “我们提出一个新的 PEARL optimizer。”

真正应该回答的是：

> **对于一个仅通过少量测试可观测、内部未知的 SUT，能否利用历史测试轨迹快速推断其脆弱性，并在固定测试预算下自适应地选择和执行更有效的危险场景？**

形式化为：

\[
\mathcal{T}_{i,j}
=
(\text{SUT behavior profile}_i,\text{scenario family}_j)
\]

给定未知 task：

\[
\mathcal{T}^{*}
\]

经过少量交互：

\[
C_K
\]

推断：

\[
z_K\sim q_\phi(z|C_K)
\]

并使：

\[
\pi_{\text{scene}}(\cdot|z_K)
\]

和：

\[
\pi_{\text{adv}}(\cdot|z_K)
\]

快速改变，从而在相同 simulator budget 下得到更多 unique failures。

---

# 3. 现阶段使用差异化 IDM 模拟不同 ADS：如何合理定位

## 3.1 可以使用，而且建议作为当前 POC 主方案

目前代码尚不能直接无缝替换不同真实 ADS。现阶段建议不再等待真实 ADS adapter，而是构造一组**行为差异明显的 IDM profiles**，作为 surrogate SUT tasks。

但论文措辞必须准确。

当前阶段建议使用：

> **different SUT behavioral profiles**

或：

> **different parameterized driving-controller profiles**

不要直接写：

> **different ADS algorithms**

因为参数变化仍来自同一种 IDM controller family。

---

## 3.2 现有 IDM profile 参数还需要改进

当前 `ControllerProfile` 主要包含：

```text
target_speed_mps
enable_lane_change
yield_gap_m
brake_gap_m
```

而 `IDMSUTAdapter` 实际做了：

```text
target_speed_mps → policy.target_speed
yield_gap_m     → policy.DISTANCE_WANTED
brake_gap_m / target_speed_mps → policy.TIME_WANTED
```

这里建议做一个小而重要的语义修正：

### 修改 1：将 `brake_gap_m` 改成直接的 `time_headway_s`

现在 `brake_gap_m` 最终被转换成时间头距，变量名称容易误导。

建议改为：

```text
target_speed_mps
enable_lane_change
distance_wanted_m
time_headway_s
```

如果当前 MetaDrive 版本可稳定配置其他 IDM 行为参数，可以后续增加 1–2 个参数；但不要一开始扩展成十几个 controller 参数。

---

## 3.3 建议使用 6–8 个明显不同的 IDM profile

优先构造行为上可解释的 profile，例如：

| Profile | target speed | lane change | distance wanted | time headway | 行为语义 |
|---|---:|---|---:|---:|---|
| cautious | 8 | off | large | large | 保守、早让行 |
| defensive | 10 | off | large | medium-large | 稳健跟驰 |
| normal | 11–12 | on | medium | medium | 普通 |
| assertive | 14 | on | small | small | 激进 |
| fast-small-gap | 16 | on | very small | small | 高速小间距 |
| late-response | 14–15 | on | small | very small | 晚反应 |

具体数值不要仅凭名称决定，而应由 failure-landscape audit 反向校准。

---

## 3.4 profile 是否有效，不看参数差异，而看 failure landscape 差异

这是整个 Meta-RL 能否成立的第一前提。

定义：

\[
f_i(x)
=
P(\text{failure}\mid x,\text{profile}_i)
\]

需要看到：

\[
f_i(x)\neq f_j(x)
\]

而不仅仅是：

\[
\theta_i\neq\theta_j
\]

因此开发第一阶段不是立即训练 Meta-RL，而是：

1. 对每个 family 生成同一批 Sobol / random scene configurations；
2. 对不同 IDM profile 重复测试；
3. 比较 failure rate、severity、dangerous-region overlap；
4. 计算 profile 间 failure landscape distance；
5. 如果差异很小，先重新设计 IDM profiles，而不是继续训练 Meta-RL。

---

# 4. 稳定后的完整 Framework

## 4.1 总体数据流

```text
                      Known Road Network G
                              │
                       Map Tokenization
                              │
                         HPTR Encoder
                              │
                  ┌───────────┴───────────┐
                  │                       │
              H_G local               g_G global
                  │                       │
                  │                       │
                  │                 Outer Policy
                  │                Hybrid PPO
                  │             π_scene(g_G,z,H)
                  │                       │
                  │        candidate + config + option
                  │                       │
                  │                 Scenario Reset
                  │                       │
                  └──────────────┐        │
                                 ▼        ▼
                              Simulator Episode
                                 │
                       Inner SAC at each timestep
                                 │
                 a_t ~ π_adv(s_t,map,z,x_k)
                                 │
                                 ▼
                           Full Trajectory
                                 │
                       Trajectory Encoder
                                 │
                              h_tau
                                 │
          g_G + config + option + h_tau + outcome
                                 │
                          Episode Token e_k
                                 │
                          Context Set C_K
                                 │
                          q_phi(z | C_K)
                                 │
                           posterior z_K
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
           next Outer decision           next Inner behavior
```

---

## 4.2 Component A：Map Tokenizer + HPTR

输入：

\[
G
\]

经过 polyline tokenization 后得到：

- lane geometry；
- heading；
- curvature；
- lane width；
- speed limit；
- road relations。

HPTR 输出：

\[
(H_G,g_G)
\]

其中：

\[
H_G\in\mathbb{R}^{N\times d}
\]

表示每一个 map polyline 的 contextualized local representation；

\[
g_G\in\mathbb{R}^{d}
\]

表示整个 map 的 pooled global representation。

当前默认：

\[
d=128
\]

### 冻结建议

- HPTR 维度保持 128；
- layers 保持 2；
- heads 保持 4；
- 不增加大型 map transformer；
- map encoder 冻结后缓存 embedding。

---

## 4.3 Component B：Trajectory Encoder

每次完整测试得到：

\[
X_k\in\mathbb{R}^{T\times12}
\]

当前 12-D trajectory evidence 已经包含：

- relative position；
- relative speed；
- SUT acceleration；
- SUT speed；
- lateral offset；
- adversary / SUT progress；
- TTC；
- PET；
- pair distance；
- conflict timing。

经过：

\[
h_{\tau,k}=E_{\text{traj}}(X_k)
\]

得到 episode-level interaction representation。

这个设计应保留，因为它比原始 PEARL 的高度相关 timestep context 更适合当前任务。

---

## 4.4 Component C：Episode Token

一个完整 simulator test 被压缩为一个 token：

\[
e_k=
E_{\text{episode}}
(g_G,\mathbf{x}_k,o_k,h_{\tau,k},y_k)
\]

其中：

- \(g_G\)：已知道路环境；
- \(\mathbf{x}_k\)：本次 scene configuration；
- \(o_k\)：高层 adversarial intent；
- \(h_{\tau,k}\)：SUT 在测试过程中的实际响应；
- \(y_k\)：最终 failure / severity outcome。

因此：

\[
\boxed{1\ simulator\ episode = 1\ meta-context\ shot}
\]

这是当前方案最值得保留的设计之一。

---

## 4.5 Component D：Latent Context Posterior

使用：

\[
q_\phi(z|C_K)
\]

输出：

\[
\mu_K,\log\sigma_K^2
\]

并：

\[
z_K=
\mu_K+\sigma_K\epsilon
\]

K=0 时：

\[
q(z|C_0)=\mathcal{N}(0,I)
\]

当前 `SetPosterior` 的 masked mean pooling + diagonal Gaussian 足够简洁，不建议重新恢复旧 PEARL 的 transition-level Product-of-Gaussians。

---

## 4.6 Component E：Outer Hybrid PPO

Outer 工作在 **episode timescale**。

输入建议最终为：

\[
[g_G,z_K,H_K]
\]

输出：

\[
(c_k,\mathbf{x}_k,o_k)
\]

其中：

- \(c_k\)：route / conflict candidate；
- \(\mathbf{x}_k\)：连续初始配置；
- \(o_k\)：对抗 intent。

### 当前 POC

可先使用：

\[
[g_G,z_K]
\]

即 `outer_history_dim=0`。

### 后续正式实验

如果 Outer reward 使用 novelty，建议添加一个**很小的显式 history vector**：

\[
H_K
\]

只包含：

- used-budget ratio；
- unique failure count；
- recent failure indicator；
- recent severity；
- candidate visitation counts / coverage summary。

不要引入 RNN history encoder。

---

## 4.7 Component F：Scenario Executor

Outer 输出必须真正作用于物理 simulator：

\[
(c_k,\mathbf{x}_k,o_k)
\rightarrow
\text{MetaDrive reset / spawn / route / speed}
\]

这一点当前代码已经建立了较好的物理闭环，应保留。

---

## 4.8 Component G：Inner SAC

Inner 工作在 **timestep timescale**。

当前实现：

\[
\pi_{\text{adv}}
(a_t|s_t,g_G,z_K,o_k,\mathbf{x}_k)
\]

输出：

\[
a_t\in[-1,1]^2
\]

对应 adversarial vehicle 的连续控制。

### 关于 \(H_G\) 是否必须立刻输入 Inner

理论上更完整的形式是：

\[
\pi_{\text{adv}}
(a_t|s_t,H_G,z_K,o_k,\mathbf{x}_k)
\]

但为了避免再次扩张模型，建议：

- **P0 继续使用当前 \(g_G\)**；
- 在核心 Meta-RL 闭环稳定后，再增加轻量 local-map attention；
- 将 \(H_G\rightarrow\) Inner 作为 map-ablation / P1 增强，而不是阻塞主线开发。

如果最终实验显示 global \(g_G\) 已经足够，不需要为了“形式更漂亮”强制加入复杂 cross-attention。

---

# 5. 当前代码中应优先修改什么？

以下建议按优先级划分。

---

# P0：不完成就不应开始大规模训练

## P0-1：实现统一的 Online Meta-Test Loop

当前组件已经存在，但需要一个官方闭环：

```text
z0 = prior
for k in budget:
    Outer(g_G, z_k, H_k)
        ↓
    ScenarioExecutor.reset()
        ↓
    Inner rollout
        ↓
    trajectory + outcome + failure signature
        ↓
    EpisodeToken
        ↓
    update C
        ↓
    posterior q(z | C)
        ↓
    z_{k+1}
```

建议新增：

```text
meta_testing/online_loop.py
```

或：

```text
meta_testing/training/online_meta_test.py
```

这个模块应该成为训练、评估、消融共享的唯一在线协议。

---

## P0-2：修正 Inner reward

当前 `HierarchicalRunner` 直接使用：

```python
reward_inner = float(env_reward)
```

这是当前最需要修正的地方之一。

Inner 的目标不是 MetaDrive 默认驾驶奖励，而是：

> **在保持测试物理有效的情况下制造高风险交互。**

建议最小化设计：

\[
r_t^{inner}
=
w_1r_{\text{risk}}
-
w_2r_{\text{invalid}}
+
w_3r_{\text{intent}}
\]

其中 `risk` 不需要复杂：

- inverse TTC / clipped TTC risk；
- pair-distance risk；
- closing-speed risk；
- conflict-zone proximity。

`intent` 只做轻量 shaping，避免 option 完全被策略忽略。

不要设计十几个 reward terms。

---

## P0-3：明确 optimizer 的参数所有权

当前 stage 文件只写：

```text
inner
posterior
outer
...
```

但模型中实际还有：

- `map_encoder`
- `trajectory_encoder`
- `shared_feature_encoder`
- `option_embedding`
- `scene_policies`
- `inner_sac`

需要明确：

### Inner pretrain

建议 train：

```text
shared_feature_encoder
option_embedding
inner_sac
```

map encoder 不要长期与所有模块一起同时更新。建议先允许其在早期 map-aware feature learning 中更新，稳定后冻结并缓存。

### Posterior

train：

```text
trajectory_encoder
episode_token_builder
posterior
outcome_decoder
```

map encoder默认冻结。

### Outer

train：

```text
scene_policy
```

posterior 和 map 默认冻结。

### Light joint

只允许小学习率更新：

```text
posterior + scene_policy + inner policy heads
```

不建议全网络大幅更新。

---

## P0-4：补全真正可运行的 staged training scripts

当前训练模块主要是 building blocks，需要补充：

```text
train_inner.py
train_posterior.py
train_outer.py
evaluate_meta_test.py
```

`train_joint.py` 可以后置。

每个 script 必须支持：

- checkpoint；
- resume；
- seed；
- family filter；
- SUT profile split；
- episode/step budget；
- device；
- metrics logging。

---

## P0-5：PPO 必须使用真正的 on-policy rollout + GAE

当前已有 `clipped_ppo_loss()`，但正式训练不能用普通 replay buffer 代替 PPO rollout。

Outer rollout buffer 至少应保存：

```text
g_G
z
H
candidate
continuous
option
old_log_prob
value
reward
done
```

并计算：

```text
return
advantage / GAE
```

不要为了代码复用把 PPO 写成“伪 off-policy”。

---

## P0-6：修正 failure / invalid 语义

需要彻底区分：

\[
\text{is_failure}
\]

与：

\[
\text{is_valid_episode}
\]

安全但物理有效的 episode：

```text
failure = False
valid_episode = True
```

不能被统计为 invalid。

最终统一结构建议：

```text
is_valid_episode
is_failure
is_collision
is_near_miss
severity_vector
failure_signature
```

---

# P1：核心闭环稳定后再完成

## P1-1：增强 IDM profile heterogeneity

建议从当前混合 `IDM + rule-based` POC 改为一个更干净的主实验：

```text
meta-train: multiple IDM behavior profiles
validation: held-out IDM profile(s)
meta-test: held-out IDM profile(s)
```

rule-based profile 可以作为额外 OOD robustness experiment，而不是和 IDM 混在主 split 中。

这样论文能够更清楚地回答：

> 在 controller family 相同但行为模式不同的情况下，few-shot latent 是否能快速辨认 failure landscape？

未来接入真实 ADS 后再升级 claim。

---

## P1-2：补全 HPTR relation semantics

当前代码声明：

```text
successor
predecessor
left
right
merge
split
conflict
crossing
route_membership
```

但实际 relation inference 主要完成前四类。

建议补：

- merge；
- split；
- conflict；
- crossing。

`route_membership` 如果没有强需求，可以暂缓。

这比增加 HPTR 层数更有价值。

---

## P1-3：增加轻量 Outer history

如果 novelty 是 Outer reward 的重要组成部分，则添加 compact \(H_K\)，不要加 RNN。

推荐：

```text
budget_fraction
unique_failure_count / budget
last_failure
last_severity
candidate_visit_histogram
```

维度控制在 8–16。

---

## P1-4：验证 \(H_G\) 是否值得给 Inner

先比较：

### A

\[
\pi_{adv}(s_t,g_G,z,x)
\]

### B

\[
\pi_{adv}(s_t,\operatorname{Attn}(s_t,H_G),z,x)
\]

如果 B 没有显著提升：

- failure discovery；
- map generalization；
- option controllability；

则保留 A。

不要无证据增加 cross-attention。

---

# P2：论文主体结果成立后才做

以下功能不是当前 POC 的阻塞项：

- 真实 learned ADS adapter；
- 多种真实 ADS algorithm；
- unseen scenario-family adaptation；
- full-network test-time fine-tuning；
- MoE；
- information-gain objective；
- QCNet / 大型 motion encoder；
- 多 adversarial vehicles；
- Outer Hybrid SAC；
- 更复杂 Bayesian nonparametric posterior。

这些都不应进入当前主开发路径。

---

# 6. PEARL 在当前方法中到底保留了什么？

应统一为下面的解释。

## 原始 PEARL 的关键思想

\[
C\rightarrow q_\phi(z|C)\rightarrow \pi(a|s,z)
\]

即：

> 从少量交互上下文中推断 latent task，并使策略根据 latent 快速适应。

---

## 当前方法保留的核心

仍然是：

\[
C_K\rightarrow q_\phi(z|C_K)\rightarrow z_K
\]

并且：

\[
z_K
\rightarrow
\pi_{\text{scene}}
\]

以及：

\[
z_K
\rightarrow
\pi_{\text{adv}}
\]

因此 Meta-RL 机制仍然成立。

---

## 当前方法不应机械继承的部分

不要为了“还是 PEARL”而强行恢复：

- transition-level context；
- geometry latent；
- 单一 SAC policy；
- Product-of-Gaussians on correlated transitions；
- 完全连续的 action assumption。

这些正是旧方案中导致 meta adaptation 弱的重要原因之一。

---

# 7. 论文创新点应收敛到 3 个，不要再扩散

建议主创新固定为以下三个。

## Innovation 1：SUT-vulnerability-oriented meta-task formulation

把传统 scenario geometry task 改成：

\[
\boxed{
\text{Task}=
\text{SUT behavioral vulnerability}
\times
\text{known scenario family}
}
\]

地图显式给定，latent 只描述未知 SUT。

---

## Innovation 2：Complete-test trajectory context

不同于 transition-level context：

\[
(s_t,a_t,r_t,s_{t+1})
\]

当前使用：

\[
\boxed{
\text{one complete test}
\rightarrow
\text{one episode evidence token}
}
\]

能更自然地对应自动驾驶测试中的“一个测试用例”。

---

## Innovation 3：Dual-timescale hierarchical meta-testing

同一 vulnerability latent \(z\) 同时指导：

### episode-level：

\[
\pi_{\text{scene}}
\]

决定**下一次测试什么**；

### timestep-level：

\[
\pi_{\text{adv}}
\]

决定**这次测试中如何动态施压**。

这三个创新已经足够构成一条完整论文主线。

不要再把：

- PPO；
- SAC；
- HPTR；
- GRU；

分别包装成算法创新。

它们是实现组件。

---

# 8. 单张 RTX 4090 下如何控制开发成本

当前网络并不大，真正成本主要来自 simulator sampling。

因此应使用以下原则。

## 8.1 一个 episode 多用途

一次 simulation 同时产出：

- Inner SAC transitions；
- trajectory evidence；
- posterior training sample；
- episode-level Outer reward；
- failure signature；
- evaluation statistics。

不要为不同模块重复跑同一物理测试。

---

## 8.2 Map embedding 缓存

同一个 map：

\[
G\rightarrow(H_G,g_G)
\]

在 map encoder 冻结后只算一次。

---

## 8.3 Posterior 尽量使用已有 rollouts 离线训练

Inner pretraining 收集的轨迹不要丢弃。

可以重组成：

```text
support episodes
target episode
```

大量训练 posterior，无需每个 gradient step 都重新调用 MetaDrive。

---

## 8.4 开发阶段只使用 1 seed

开发 / debug：

```text
1 family
2–3 profiles
1 seed
small budget
```

正式结果再扩展：

```text
3 families
held-out profiles
3 seeds
```

---

## 8.5 Joint 默认关闭

只有 Gate 证明需要时才开启。

---

## 8.6 不追求大网络

建议保持：

```text
map_dim = 128
trajectory_hidden = 128
latent_dim = 16
policy_hidden = 256
```

在当前问题上，增加模型规模的优先级远低于改善 task heterogeneity 和 reward correctness。

---

# 9. 建议将 Scientific Gates 压缩为 4 个核心 Gate

以前 G1–G8 对工程审计有用，但日常开发容易显得过长。

主研究流程建议只看四个 Gate。

---

## Gate A：Failure Landscape Heterogeneity

问题：

> 不同 IDM profile 是否真的在同一 test space 中表现出不同危险区域？

如果不成立：

\[
\boxed{\text{停止 Meta-RL 训练，先重设 profiles}}
\]

这是最重要的 Gate。

---

## Gate B：Inner Controllability

问题：

> Inner SAC 是否能在不同 family 中稳定制造有意义的风险交互，同时保持 episode 有效？

如果 Inner 自身不会产生有意义的 adversarial interaction，Outer 和 posterior 都没有可靠学习信号。

---

## Gate C：Few-shot Identifiability + Causal Utility

需要同时证明：

### Identifiability

\[
q(z|C_K)
\]

随 K=1/2/4 能逐渐区分 profile vulnerability。

### Causal Utility

比较：

\[
z_{\text{correct}}
\]

与：

\[
z_{\text{zero}}
\]

和：

\[
z_{\text{swapped}}
\]

必须看到 policy/failure discovery 显著变化。

否则 \(z\) 只是一个可视化 embedding，不是 Meta-RL 中真正有用的 adaptation variable。

---

## Gate D：Equal-budget Failure Discovery

最终比较：

\[
\boxed{
\text{unique failures under identical simulator budget}
}
\]

主指标：

- cumulative unique failures；
- discovery AUC；
- tests-to-first；
- tests-to-5；
- tests-to-10；
- severity；
- invalid rate。

所有 K=0/1/2/4 probes 都计入总 simulator budget。

---

# 10. Baseline 不要铺得过多

当前建议论文主体先实现以下 5 组：

1. **Random / Sobol**
2. **CEM 或 BO（二选一先实现，再决定是否两个都保留）**
3. **Hierarchical RL without z**
4. **Meta-RL with z only for Outer / Inner ablation**
5. **Full method**

旧 PEARL：

- 保留作为历史 baseline；
- 只在相同 scenario/task protocol 下重跑；
- 不再继续作为 active implementation 主线。

不要为了表格丰富一次实现十几个 baseline。

---

# 11. 当前代码的具体“保留 / 修改 / 暂缓”清单

| 模块 | 当前状态 | 建议 | 优先级 |
|---|---|---|---|
| `context/set_posterior.py` | episode-set Gaussian posterior | 保留 | 保留 |
| `context/trajectory_encoder.py` | whole-trajectory GRU | 保留 | 保留 |
| `context/episode_token.py` | map+config+option+trajectory+outcome | 保留 | 保留 |
| `map/hptr_encoder.py` | HPTR-style 128-D | 保留，不扩模型 | 保留 |
| `map/relations.py` | 关系类型声明多，实际生成少 | 补 merge/split/conflict/crossing | P1 |
| `policy/scene_policy.py` | Hybrid PPO | **保留，不改 Hybrid SAC** | 冻结 |
| `policy/adversarial_sac.py` | continuous SAC | 保留 | 冻结 |
| `policy/shared_features.py` | Inner 用 global map | P0 保留，P1 评估 local H_G | P1 |
| `training/runner.py` | 使用 env reward | 改为 adversarial inner reward | **P0** |
| `training/updates.py` | posterior ELBO + PPO loss | 保留；补 GAE trainer | **P0** |
| `training/stages.py` | 4 stages | joint 改 optional | **P0** |
| `model.py` | 高层结构正确 | 不重构；只补 history / local map 接口 | 保留 |
| `sut/base.py` | profile 参数语义略混乱 | `brake_gap`→`time_headway` | P1 |
| `sut/registry.py` | IDM + rule-based 混合 split | 主实验改 IDM-profile split | P1 |
| evaluation | budget-aware 基础存在 | 修正 valid/invalid；统一在线 loop | **P0** |
| scripts | 缺完整训练入口 | 增加 4 个正式脚本 | **P0** |
| external ADS bridge | 未完成 | 暂缓 | P2 |
| Hybrid SAC Outer | 未实现 | **不实现**，除非 PPO 被实验证明是瓶颈 | 暂缓 |

---

# 12. 推荐的开发顺序

## Phase 0：代码正确性

完成：

- Inner reward；
- failure validity semantics；
- optimizer ownership；
- online loop；
- training/eval scripts。

目标：

> 一套代码能从 reset → rollout → trajectory → posterior → next test 连续跑完整预算。

---

## Phase 1：不用 Meta-RL，先验证 task heterogeneity

只跑：

```text
Sobol scene configurations
×
multiple IDM profiles
×
3 scenario families
```

目标：

> 证明不同 profile 的危险区域确实不同。

失败则重新设计 profile。

---

## Phase 2：Inner SAC

先训练 universal Inner：

\[
\pi_{\text{adv}}(a_t|s_t,g_G,x_k)
\]

初期甚至可以固定：

\[
z=0
\]

先证明 adversarial controller 本身可用。

---

## Phase 3：Posterior

使用已有 trajectories 训练：

\[
q(z|C_K)
\]

做：

- held-out outcome prediction；
- latent clustering / separability；
- K=0/1/2/4；
- swapped-z test。

---

## Phase 4：Outer PPO

冻结：

- map encoder；
- Inner；
- posterior 主体。

训练：

\[
\pi_{\text{scene}}(x|g_G,z,H)
\]

目标：

> 根据已识别的 vulnerability 选择更有价值的下一测试场景。

---

## Phase 5：Full online evaluation

在 unseen IDM profile 上：

```text
C0 → test1 → C1 → test2 → C2 → ...
```

固定总预算，例如 20 episodes。

比较 baselines。

---

## Phase 6：Optional calibration

只有 Gate C/D 表明必要时，做轻量 joint。

---

# 13. 什么时候才允许重新考虑 Outer Hybrid SAC？

为了防止后续再次因为“理论统一”而改变路线，建议设定明确触发条件。

只有满足下面两个条件之一，才重新讨论 Outer optimizer：

### 条件 1

在已经：

- 冻结 Inner；
- 并行 simulator；
- 合理 PPO batch；
- 正确 GAE；
- 正确 reward normalization；

之后，Outer PPO 仍然显示严重 sample inefficiency，无法达到 Gate D。

### 条件 2

论文实验明确显示：

> on-policy data cost 是整个方法性能/成本的主要瓶颈。

否则：

\[
\boxed{\text{Outer PPO 不改}}
\]

即使 PEARL 原始实现使用 SAC，也不构成改动理由。

---

# 14. 最终论文叙事建议

## 不建议

> We extend PEARL by adding PPO and SAC...

这种写法会让方法看起来像算法拼接。

## 建议

> We formulate autonomous-driving scenario discovery as a latent-context meta-RL problem in which the unknown task factor is the vulnerability of the SUT rather than the road geometry. Complete simulator tests are encoded as episode-level evidence to infer a probabilistic vulnerability latent. The inferred latent jointly conditions an episode-level scene-selection policy and a timestep-level adversarial policy, enabling dual-timescale adaptation under a fixed testing budget.

然后在 implementation 中说明：

- scene policy optimized by PPO；
- adversarial policy optimized by SAC；
- latent inference inspired by PEARL。

这样：

\[
\boxed{
\text{创新 = problem formulation + context + hierarchy}
}
\]

而不是：

\[
\text{创新 = PPO/SAC optimizer}
\]

---

# 15. 最终冻结结论

从现在开始建议主线固定为：

\[
\boxed{
\text{HPTR explicit map}
+
\text{episode-level trajectory context}
+
q_\phi(z|C)
+
\text{Outer Hybrid PPO}
+
\text{Inner SAC}
+
\text{online few-shot posterior update}
}
\]

其中：

### 保留 PEARL 的：

\[
\boxed{
C\rightarrow q(z|C)\rightarrow \text{latent-conditioned policy adaptation}
}
\]

### 不恢复原始 PEARL 的：

- transition-level context；
- geometry-as-task latent；
- single-policy architecture；
- standard continuous-only SAC assumption。

### 不再新增的：

- Outer Hybrid SAC；
- 大型地图网络；
- full joint from scratch；
- test-time full-network fine-tuning。

### 当前最优先的工程工作：

1. **修正 Inner reward**；
2. **建立官方 online meta-test loop**；
3. **完成真正的 staged trainers + PPO GAE**；
4. **修正 failure/invalid 语义**；
5. **用差异显著的 IDM profiles 先通过 failure-landscape Gate**；
6. **再训练 posterior 和 Outer，而不是继续改模型结构**。

---

# 16. 一句话的最终方法定义

> **本方法不是“把 PEARL 换成 PPO”，而是在保留 PEARL-style few-shot latent task inference 的基础上，将自动驾驶失效场景挖掘重新建模为一个地图显式条件化、SUT vulnerability 隐变量化、Outer episode-level scene selection 与 Inner timestep-level adversarial control 协同工作的层级 Meta-RL 问题。**

这条定义建议作为后续代码设计、实验设计和论文写作的统一基准。
