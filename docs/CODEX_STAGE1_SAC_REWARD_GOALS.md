# Codex 执行目标：Stage 1 共享 Inner SAC 的奖励闭环、训练与结果交付

> 目标仓库：`SafeDL/META_LEARNING`  
> 核对基准：`d5e1caaedf426e69cc4de0de9930250817e3a756`  
> 文档日期：2026-09-06  
> 当前动作接口：`frenet_path_longitudinal_v2`  
> 当前物理 observation：31 维  
> 执行原则：保留已有 v2 架构与奖励数值，只补齐必要的奖励分解、时间尺度一致性检查及可追溯输出；不开展反复调参和大规模消融。

## 0. 给 Codex 的直接任务

请阅读本仓库适用的 `AGENTS.md`，核对当前实际分支与本文件的基准差异，然后完成以下工作：

1. 沿用当前 Stage 1 设计，不重新实现 SAC、Frenet、PEARL 或整个 pipeline。
2. 明确实现并记录：每个环境控制步计算一次即时奖励；每个 SAC 宏动作对应一个折扣聚合奖励；训练图显示未折扣 episode return。
3. 在不改变奖励数值、事件定义和控制动作的前提下，增加一次计算即可得到的奖励分项与必要统计。
4. 运行本次变更相关的轻量单元检查，然后最多执行一次既定 Stage 1 训练和一批小规模配对评估。不自动换 seed、扩大参数域或反复重训。
5. 输出代码变更说明、奖励契约、训练记录、逐域结果、出版绘图文件及诚实的结论。无法完成运行时，保留原因，不编造结果。

本文件中标为“当前实现”的内容来自上述 commit 的源码；标为“本轮新增要求”的内容是拟实施的修改，不得写成已经存在或已经验证。

## 1. 研究目标与不可变范围

### 1.1 Stage 1 的问题定义【当前实现】

固定：

- Functional Scenario：`cutin`。
- SUT：`idm_normal`。
- Geometry：`cutin-g01`。
- 变化轴：`close_closing_early`、`balanced_interaction`、`late_tight_cutin` 三个 Logical Domain 的参数域。
- 左右切入候选继续保留。

三个域联合训练一个条件化 Inner SAC，共用 actor、双 critic、target networks 和 replay。不是三个独立 SAC，也不是同一个域训练好以后依次覆盖参数。

目标是：给定具体初态和参数域条件，同一个闭环策略能在三个域中产生语义有效、执行有效的危险交互，并在同一批场景上体现相对随机策略的对抗收益。

Stage 1 不是 few-shot、不是新 SUT / 新道路泛化、不是 Outer 搜索收益实验。`context_meta` 和 Outer/MoE 本轮不训练。

### 1.2 本轮不得擅自修改

- 不改变三个参数域的范围，也不通过筛掉困难场景提高成功率。
- 不降低 TTC、距离、侵入或执行有效性的验收要求。
- 不把空间路径 `d(s)` 重写成完整时域轨迹 `s(t), d(t)`。
- 不改变 4D 动作、31D observation、actor detach、target encoder 或安全投影结构。
- 不修改奖励系数、奖励裁剪、SAC 学习率和熵目标来让曲线变漂亮。
- 不重新打开 event BC，不把 `reward >= 1` 当作事件标签。
- 不删除原始失败记录，不修改旧结果的配置/源码 hash，不将旧 checkpoint 冒充 v2 结果。
- 不把 `archives/sac_scenario_mining` 的历史结果当成本次 Cut-in 结果；归档只能作为实现参考。

发现确定性错误时，记录最小复现和影响范围。仅本文件授权的奖励记录/时间契约问题可以直接修复；需要改变 MDP、执行器或算法结构的问题，列出阻塞原因后停止自动扩展。

## 2. 时间与奖励的三层契约

### 2.1 环境控制步：每一步都有即时奖励【当前实现】

这里的环境步是一次 `env.step(...)`，不是 Bullet 内部的每次物理积分。

```text
状态 s_t
  -> 保持当前 SAC 宏动作 a_k
  -> Frenet 参考 + 跟踪 + 纵向限速 + 物理投影 + shield
  -> env.step(executed_action)
  -> 更新 schedule、semantic monitor、trajectory features
  -> 计算本环境步 reward_inner = r_t
  -> 保存下一状态 s_(t+1)
```

每个环境控制步计算一次 `reward_inner`。有即时奖励不等于每步奖励必为正或非零，也不等于每步更新网络。

动作获得的是其执行后状态转移对应的反馈，而不是仅凭动作向量打分。当前 `InnerRiskReward` 接收的是转移后的轨迹特征与 `info`，并保存少量历史状态。

环境控制周期必须从运行时配置记录：

\[
\Delta t_{env}=physics\_world\_step\_size\times decision\_repeat.
\]

不能只根据数字 5 声称 SAC 决策周期就是 0.5 秒。

### 2.2 SAC 宏步：一个动作、一条 replay、一个宏奖励【当前实现】

SAC 在决策边界输出：

\[
a_k=[\lambda_{length},\beta_{early},\beta_{late},u_{long}]\in[-1,1]^4.
\]

动作通常保持 5 个环境步；机动激活、重规划对齐、结束等情况可能使 block 长度不同。必须使用实际 `duration_steps = H_k`，不能假定永远是 5。

宏奖励定义：

\[
R_k=\sum_{j=0}^{H_k-1}\gamma^j r_{t_k+j},\qquad \gamma=0.99.
\]

宏 transition：

\[
(s_k,a_k,R_k,s_{k+1},done_k,H_k,event_k,task\_context).
\]

具体契约：

- `state` 取 block 第一行的 state。
- `action` 取 block 第一行的 `raw_policy_action`。
- `next_state` 取 block 最后一行的 next_state。
- `duration_steps` 等于 block 行数。
- `reward` 是 micro reward 的折扣和，不是最后一帧奖励，也不是 episode return。
- 只聚合已经逐步裁剪的 micro reward；不要再把 macro reward 裁剪到 `[-3,12]`。
- `event_occurred` 来自显式新捕获的有效事件，不从 reward 大小推断。
- 保留 micro trace 供事件、控制和风险审计，但不得另外把同一批 micro 行重复送入同一个 SAC replay。
- 经过投影的 `planner_action`、车辆动作保留为遥测，不替换 replay 中的原始策略动作。

数值例子，仅用于验证聚合公式，不是实验数据：

```text
r = [0.10, 0.05, -0.02, 0.03, 6.00]
gamma = 0.99
H = 5
R_macro = 5.92258303
bootstrap_discount = 0.9509900499
```

### 2.3 Episode return：用于绘图的统计量【当前实现】

当前 `episode_return_curve.inner_return` 是：

\[
G_{episode}=\sum_{t=0}^{T-1}r_t.
\]

这是未折扣的环境奖励总和。它不同于单步 `r_t`、宏步 `R_k`，也不等于未加全局折扣的 `sum(R_k)`。

若另外计算完整折扣回报，应满足：

\[
\sum_k \gamma^{t_k}R_k=\sum_t\gamma^t r_t.
\]

不要将 episode 总回报重复写入每条 transition。不要将 SAC 熵正则项、actor loss 或 Q 值混入环境 reward 或训练图的 return。

## 3. 当前即时奖励的精确定义

### 3.1 特征解码与危险度【当前实现】

令一次环境转移结束后的特征为 `row`：

```python
ttc = max(0.0, row[8] * 15.0)
distance = max(0.0, row[10] * 100.0)
closing_proxy = distance / max(ttc, 1e-3) if ttc < 15.0 else 0.0
```

这里的 `closing_proxy` 是奖励代码从 TTC/距离重构的量，不能无条件称为未裁剪的真实闭合速度。当前 TTC 与距离来自两车相对几何特征，不等于保险杠间距，也不是完整未来碰撞预测。

在当前阈值配置下：

\[
\phi_{t+1}=C_{t+1}\left[
0.5e^{-TTC_{t+1}/5}
+0.3e^{-D_{t+1}/10}
+0.2\min(closing\_proxy_{t+1}/20,1)
\right].
\]

`C` 是 `semantic_challenge_phase_active`。正式仿真中必须提供这个字段；代码中缺省 `True` 是历史单元探针兼容行为，不能依赖缺省值运行正式训练。

危险度即时分项：

\[
r_t^{risk}=2(\phi_{t+1}-\phi_t^{stored}).
\]

episode reset 时 stored criticality 清零，每次奖励计算后更新一次。

重要边界：

- 当前是普通危险度差分，不是 `gamma * Phi(next) - Phi(current)`。
- 本轮保持公式，不宣称折扣意义下的 potential-based policy invariance。
- 离开 challenge 时当前 criticality 变成零，但上一值可能非零，因此退出当步可以得到负的差分奖励；“challenge 外永远无奖励”不准确。
- 加入 `valid_near_miss_seen` 只补充事件奖励记忆，不应据此宣称已证明整个 observation 严格满足 Markov 性。

### 3.2 事件奖励【当前实现】

本环境步必须 `event_just_captured=True`，事件还必须符合语义和执行条件：

\[
B_t=\begin{cases}
12,&\text{本步新捕获有效目标碰撞};\\
6,&\text{本步新捕获有效 near-miss};\\
0,&\text{否则}.
\end{cases}
\]

同一步若 invalid 标志成立，不发事件奖金。invalid 包括 `non_target_collision`、`adversary_out_of_road`、`sut_out_of_road`、`wrong_route`。

near-miss 候选的数值条件是同一步满足 `0 < distance < 10` 且 `ttc < 5` 且无目标碰撞，再经过语义/执行判定。不能用不同时间发生的全局最小 TTC 和全局最小距离组合判事件。

一次有效 near-miss 不结束 rollout，只奖励一次；之后的有效 collision 可以升级事件并再获得 12。因此当前同一 episode 的事件奖金合计可能是 18，而不是强制二选一。保留这个行为，不擅自改成仅奖励差额。

无效 near-miss 候选不能提前锁死有效事件记录。保留 `raw_near_miss_candidate` 与正式有效事件的区别。

### 3.3 进度奖励与执行惩罚【当前实现】

`p_t^max` 表示此前存储的最高参考进度，当前进度记为 `p_(t+1)`：

\[
r_t^{progress}=0.10\max(0,p_{t+1}-p_t^{max}).
\]

存储值更新为历史最大值，不能只改成上一帧进度。参考进度为路径参考量，不是无条件的物理距离前进量。

当当前参考进度大于零时：

\[
p_t^{tracking}=
0.03\min(1,|e_y|/3.5)
+0.01\min(1,|e_\psi|/(\pi/2)).
\]

否则 tracking penalty 为零。

\[
p_t^{shield}=0.10(traffic\_shield\_intervention\_l2)^2,
\qquad p_t^{invalid}=2I_{invalid}.
\]

### 3.4 总奖励【当前实现】

\[
z_t=r_t^{risk}+B_t+r_t^{progress}
-p_t^{tracking}-p_t^{shield}-p_t^{invalid},
\]

\[
r_t=\operatorname{clip}(z_t,-3,12).
\]

裁剪对象是每个环境控制步的总奖励，不是单独的事件奖金，也不是整个 episode。事件奖金为 12 不保证该步总奖励恰好为 12；其他分项与裁剪都可能影响最终数值。

原生 MetaDrive `reward_env` 与自定义 `reward_inner` 分别保存。Inner SAC 使用后者，禁止混用。

## 4. SAC 更新契约【保留已有实现】

宏步 Bellman target 为：

\[
y_k=R_k+\gamma^{H_k}(1-done_k)
\left[\min(Q_{\bar\omega_1},Q_{\bar\omega_2})
(E_{\bar\psi}(s_{k+1}),a')
-\alpha\log\pi_\theta(a'|E_\psi(s_{k+1}))\right].
\]

- 下一动作由 online encoder + online actor 产生。
- 下一动作的 Q 值由 target task/shared encoders + target critics 评价。
- target 在 `no_grad()` 下计算一次，供两个 critic 使用。
- actor 使用 detach 后的 features，但必须保留 critic 对动作的梯度。
- 保留高斯重参数化、标准 tanh、log-probability 修正、双 critic、自动 alpha、Polyak 更新。
- Stage 1 的 prior latent 为零，不进行 support/query adaptation。
- 当前 `done` 合并 runner 的不同结束原因。本轮如实记录 `termination_reason`，不悄悄改变截断的 bootstrap 语义；若要区分有限时域终点与外部 time limit，必须单独说明设计及版本变化。
- 当前每收集完一个 episode 执行配置数量的更新，不是每次拿到即时奖励就立即优化。
- 小 critic loss / 小 TD-target variance 不是学习有效的证明，可能同时受奖励、采样分布及事件数量影响。

## 5. 本轮新增要求：只补齐必要的实现与输出

### 5.1 奖励分解接口：一次求值，不能重复推进内部状态

在 `mvr/failure/inner_reward.py` 中增加可复用的单次求值接口，例如：

```python
def step_with_components(self, features, info):
    # 在此一次性计算奖励及分项，并更新内部历史状态一次。
    return reward_total, components


def __call__(self, features, info):
    reward_total, _ = self.step_with_components(features, info)
    return reward_total
```

允许采用等价实现，但必须保持旧 `__call__` 返回 float 的兼容性。

runner 每个环境步只调用一次 `step_with_components()`，同时保存 `reward_inner` 和 `reward_components`。禁止先调用 `__call__` 再为了日志调用一次 reward，因为它会把 criticality/progress 历史推进两次，改变学习信号。

`components` 至少包含以下标量，penalty 字段采用非负值：

```text
criticality_previous
criticality_current
reward_risk
reward_event
reward_progress
penalty_tracking
penalty_shield
penalty_invalid
reward_preclip
reward_clip_adjustment
reward_total
```

其中：

```text
reward_preclip = reward_risk + reward_event + reward_progress
                - penalty_tracking - penalty_shield - penalty_invalid
reward_clip_adjustment = reward_total - reward_preclip
```

这样即使发生裁剪，所有分项仍能准确重构总奖励。事件种类、有效性、`event_just_captured` 和 `event_occurred` 使用已有独立字段，不从分项数值反推。

### 5.2 Macro 记录和 episode 汇总

不改变 replay 的 action、reward、duration 和事件标签语义。增加必要的诊断输出即可，不要求把所有遥测都塞进训练 minibatch。

每个 macro block 保存或可重构：

```text
macro_index, start_micro_step, duration_steps
raw_policy_action
micro_reward_sum
macro_reward_discounted
bootstrap_discount
event_occurred
termination_reason
```

每个 episode 的轻量 JSONL 记录至少包含：

```text
run_id, checkpoint/action/reward schema, task_id, logical_domain_id
training_seed, episode_seed, candidate_index, normalized_initial_parameters
warmup, domain_episode_index, environment_steps, macro_transitions, optimizer_updates
inner_return_undiscounted, inner_return_discounted
reward_component_sums, reward_clipped_steps
valid, valid_target_collision, valid_critical_near_miss
challenge_steps, min_challenge_ttc, min_challenge_distance
actual_onset_time, actual_onset_gap, actual_onset_speeds
path_speed_infeasible_steps, path_projection_statistics
longitudinal_override_statistics, termination_reason
```

优先复用已有 onset、challenge 和控制字段。字段缺失用明确的 `null` 或缺失说明；不得用 0 假装测得值。JSON 中非有限数值显式处理，不能静默转为有利指标。

只有需要复核的少量代表轨迹保存完整 micro/macro CSV 或 JSONL；所有 episode 保存轻量汇总。不要为了日志让完整训练明显变慢。

### 5.3 轻量检查：不启动额外训练

复用现有测试；缺失时只增加本次改动必要的测试，不新建大规模 gate 链。

必须覆盖：

- 新分解接口与原奖励实现，在相同事件/特征序列上的数值等价；reset 后无历史泄漏。
- 无正式事件但 shaping 大于 1 时，`event_occurred` 仍为 False。
- 无效 near-miss 后出现有效 near-miss 能记录并发奖，重复有效 near-miss 不重复发奖，后续有效碰撞可升级。
- macro block 的 H=1、H=5、提前终止 H<5，以及前述数值例子的奖励/折扣一致性；不额外裁剪 macro reward。
- micro 回报、macro 折扣回报及 episode 汇总满足第 2 节等式；奖励求值次数等于环境 transition 数。

测试失败时修正最小接口问题；不得通过放宽断言隐藏差异。输出实际运行的命令、测试数量及结果。

## 6. 固定训练计划与计算预算

### 6.1 沿用当前配置【当前实现】

```yaml
interaction_prior:
  action_dim: 4
  batch_size: 64
  episodes_per_task: 40
  updates_per_episode: 12
  replay_capacity: 100000
  learning_rate: 0.0001
  warmup_episodes: 5
  event_sample_fraction: 0.25
  event_action_weight: 0.0
  gamma: 0.99
```

三个 domain 共 120 个 episode；前 5 轮每域随机 warm-up，合计 15 个 episode。若 buffer 满足要求，后续 105 个 episode 对应 1260 次 SAC 更新调用。一次更新调用内部有 critic 更新和 actor/alpha 更新，不能把它们与环境步混淆。

采用一个新的结果目录。本轮默认只执行这一条训练，不自动扩大预算，不承诺 120 集必然收敛。若已存在完全相同版本、配置和完整溯源的本轮训练结果，则复用，不重复跑。

### 6.2 命令

以下是当前仓库支持的训练入口：

```powershell
python -m mvr.scripts.train_mvr --config mvr/configs/cutin_inner.yaml --output results/cutin_stage1_v2_reward_contract --stop-after interaction_prior
```

执行前检查当前 `--help`，不要凭猜测添加未实现的参数。训练只停在 `interaction_prior`。

旧 v1 或 30D 模型不直接用于 v2 的本轮训练。不得通过修改 hash 规避不兼容。

当前 `--resume` 表示进入已完成阶段的下一阶段，不是同阶段 SAC 的中断续训；本轮不要将它当作恢复 replay/optimizer 的功能。若训练中断，如实报告，不自动开始另一套训练。

## 7. 单批配对评估：保持 Stage 1 范围，不扩展 OOD

### 7.1 现有 gate 的定位【当前实现】

`evaluate_cutin_inner_training_gate.py` 当前使用每个域中心点、candidate 0、三个环境 seed。

逐域门限为有效率至少 0.75，且至少出现一次有效事件。它是小规模工程检查，不充分证明覆盖整个域，更不证明优于 Random。环境 seed 不是独立训练 seed。

### 7.2 本轮拟新增的小规模配对模式

为了不额外叠加多个验证流程，用一批配对评估替代重复的中心点重放：

- 保持 `cutin / idm_normal / cutin-g01` 和三个训练参数域。
- 每域预先固定 3 个具体案例，共 9 个案例；每个案例分别运行最终 SAC 和 Random，一共 18 个 episode。
- 每域包含中心案例和两个固定种子的域内样本，覆盖左右候选；在查看模型表现之前保存案例表。
- 域内样本避免与训练具体场景完全重复；若中心案例重复，明确标注，不伪称全部 held-out。
- SAC 使用 deterministic action；Random 仅在同样的策略决策边界采样 `U[-1,1]^4`，使用同一个 v2 decoder 和执行器。
- 每对策略保持具体初态、候选、episode seed、horizon、事件规则完全一致。
- 不调用自动选择 `cutin-g04` 或新 SUT 的 validation slice。本轮不混入新的几何/SUT 分布变化。

优先在现有评估入口增加显式配对模式，默认行为不变；先实现并验证 CLI，再在执行报告中给出实际命令。不要把拟新增参数写成仓库已经支持的参数。

每域分别输出有效事件率、目标碰撞率、near-miss 率、有效率和每例差异。允许报告 pooled 均值，但不能遮盖某一域失效。

18 个 episode 只是开发预算内的工程比较；即使得到正收益，也不宣称统计显著、广泛泛化或论文最终验证完成。

若用户当前只允许生成代码、不允许运行仿真，则交付入口和案例生成逻辑，并将评估状态标为 `not_run`，不声称已完成。

## 8. 出版绘图要求【只使用真实日志】

复用 `mvr/scripts/plot_inner_sac_training.py`。本轮主图只画 `interaction_prior`，不混入 `context_meta`。

主图：

- 按 `logical_domain_id` 分组三条线；它们来自同一共享模型的一次联合训练，不是三个独立模型。
- 横轴明确写 `Training episode per logical domain` 或 `Joint training round`；不能让读者误以为总训练只有 40 集。
- 纵轴写 `Undiscounted inner episodic return`。
- 原始回报用透明散点/细线；趋势使用 trailing mean，窗口固定 8、`min_periods=1`，无零填充、无未来样本。
- 标记每域前 5 集为随机 warm-up，不能把其高回报称为已经学习。
- 三种颜色同时配不同线型，支持黑白打印；图注写明单次训练、平滑窗口和原始数据含义。
- 只在存在独立训练种子时画跨训练 seed 的误差带；本轮不能把 rolling 波动、三个域差异或环境重放种子伪装为置信区间。
- 输出矢量 PDF 和 600 dpi PNG，检查标签、图例与边界无裁切。

奖励分解只需保存可复核数据与一张简洁的逐域分项汇总表，不强制再添加多张主图。不得调整 reward 或删掉低回报点来改善视觉。

## 9. 必须交付的文件与结束条件

建议放在独立的 `results/cutin_stage1_v2_reward_contract/`：

```text
execution_report.md                   # 代码变更、来源、命令、预算、问题及结论
reward_contract.md                    # micro/macro/episode 三层定义与源码对应
run_metadata.json                     # 实际 commit、工作区差异、配置、schema、种子
reward_episode_metrics.jsonl          # 全部 episode 轻量奖励/事件/控制汇总
interaction_prior.pt                  # 本轮实际训练所得，未运行则不伪造
interaction_prior.json
manifest.json
paired_stage1_cases.json               # 配对案例与来源/重复状态
paired_stage1_report.json              # 逐例、逐域 SAC/Random 结果
shared_inner_sac_training.pdf
shared_inner_sac_training.png
```

沿用已有命名也可，但必须提供字段对应关系，避免重复保存同一大文件。

`execution_report.md` 必须回答：

1. 每次 `env.step` 是否恰好计算一次即时 reward？
2. macro replay 奖励与 `gamma^H` 是否一致？图中的 return 究竟是哪一种总和？
3. 有效事件是否来自正式语义标签？重复/无效候选是否被正确处理？
4. 三个域是否共享一个模型、是否各消耗 40 集、warm-up 和实际更新次数是多少？
5. 限速/投影是否大量覆盖策略动作？路径不可行标记出现多少次？
6. 同一 checkpoint 是否在三个域产生有效事件？与配对 Random 的差异是什么？
7. 哪些结论来自本轮实测，哪些只是代码检查、历史结果或尚未检验的假设？

分别记录以下状态，禁止只给一个模糊的 `passed=true`：

- `reward_contract_status`：奖励计算/聚合/标签的工程检查结果。
- `training_status`：`not_run / completed / interrupted / failed`。
- `domain_coverage_status`：按实际评估案例逐域报告，允许部分通过。
- `paired_gain_status`：配对样本内相对 Random 的实际差异；证据不足时为 `inconclusive`。

每域事件率大于零只是基本覆盖。若要称为“本批案例上有初步对抗收益”，还应在有效性不退化的前提下，报告 SAC 相比 Random 的逐域收益与总体收益；不以降低 loss 替代这一要求。

执行预算结束即输出结果。失败时指出最有证据的一个阻塞点，不再自行展开多轮 seed、BC、replay 比例、奖励或架构搜索。

## 10. 已知边界：记录，不在本轮偷偷扩大修复范围

- v2 对单条解码路径做单调性与速度筛查，但最长中性回退路径仍可能标记 `path_speed_feasible=False`；不能声称所有执行轨迹已被严格证明可行。
- 距离速度包络与 jerk 制动筛查是不同机制；检查通过不等于实际 tracker 一定严格复现被检查的制动序列。
- 训练 episode 的域间平衡不意味着 replay minibatch 的域间比例严格相同。
- `cutin_gap_at_start_m` 与 onset 参数用于名义初始化，不自动等于实际 onset 状态；使用新增遥测检查偏差。
- 当前三域联合训练仍是多任务共享先验，不是元适应。
- 当前 120 集是本次计算预算，不是收敛保证。
- 现有 checkpoint 续训和最佳模型选择若不支持，明确说明，不在报告中假称已实现。

## 11. 源码依据

以下链接固定到核对 commit。若本地 HEAD 已变化，先比较再执行，不回滚用户已完成的修改。

- [配置](https://github.com/SafeDL/META_LEARNING/blob/d5e1caaedf426e69cc4de0de9930250817e3a756/mvr/configs/cutin_inner.yaml)
- [即时奖励](https://github.com/SafeDL/META_LEARNING/blob/d5e1caaedf426e69cc4de0de9930250817e3a756/mvr/failure/inner_reward.py)
- [环境执行与 micro transition](https://github.com/SafeDL/META_LEARNING/blob/d5e1caaedf426e69cc4de0de9930250817e3a756/mvr/training/runner.py)
- [宏步聚合](https://github.com/SafeDL/META_LEARNING/blob/d5e1caaedf426e69cc4de0de9930250817e3a756/mvr/training/online_meta_test.py)
- [显式事件 replay](https://github.com/SafeDL/META_LEARNING/blob/d5e1caaedf426e69cc4de0de9930250817e3a756/mvr/training/replay.py)
- [语义与事件状态机](https://github.com/SafeDL/META_LEARNING/blob/d5e1caaedf426e69cc4de0de9930250817e3a756/mvr/scenario/semantics.py)
- [Frenet v2](https://github.com/SafeDL/META_LEARNING/blob/d5e1caaedf426e69cc4de0de9930250817e3a756/mvr/scenario/frenet.py)
- [控制器](https://github.com/SafeDL/META_LEARNING/blob/d5e1caaedf426e69cc4de0de9930250817e3a756/mvr/control/adversary.py)
- [物理状态](https://github.com/SafeDL/META_LEARNING/blob/d5e1caaedf426e69cc4de0de9930250817e3a756/mvr/state.py)
- [模型与 target encoders](https://github.com/SafeDL/META_LEARNING/blob/d5e1caaedf426e69cc4de0de9930250817e3a756/mvr/model.py)
- [SAC 数学实现](https://github.com/SafeDL/META_LEARNING/blob/d5e1caaedf426e69cc4de0de9930250817e3a756/mvr/policy/adversarial_sac.py)
- [SAC 更新](https://github.com/SafeDL/META_LEARNING/blob/d5e1caaedf426e69cc4de0de9930250817e3a756/mvr/training/updates.py)
- [训练与 episode return 统计](https://github.com/SafeDL/META_LEARNING/blob/d5e1caaedf426e69cc4de0de9930250817e3a756/mvr/training/trainers.py)
- [训练入口与阶段续接](https://github.com/SafeDL/META_LEARNING/blob/d5e1caaedf426e69cc4de0de9930250817e3a756/mvr/training/pipeline.py)
- [现有逐域 gate](https://github.com/SafeDL/META_LEARNING/blob/d5e1caaedf426e69cc4de0de9930250817e3a756/mvr/scripts/evaluate_cutin_inner_training_gate.py)
- [绘图入口](https://github.com/SafeDL/META_LEARNING/blob/d5e1caaedf426e69cc4de0de9930250817e3a756/mvr/scripts/plot_inner_sac_training.py)

外部背景只用于解释 SAC，不替代仓库实际契约：[OpenAI Spinning Up — Soft Actor-Critic](https://spinningup.openai.com/en/latest/algorithms/sac.html)。其中环境 reward 与目标中的熵正则分开处理；本项目进一步使用宏步折扣 `gamma^H`。

---

**交付底线：奖励计算正确，不等于学习已经成功；曲线漂亮，不等于性能提高。先保持当前 v2 契约，补足可解释的奖励闭环，用一次受控训练和一批配对场景给出真实结果。**
