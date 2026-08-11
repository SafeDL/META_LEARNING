# PEARL 逻辑场景元强化学习

本目录实现一个面向汇入类危险场景的 PEARL-SAC 流程。它解决的问题不是“对每个新场景重新训练一个策略”，而是：先在多个冻结的 `meta_train` 任务上学习共享策略；面对未见任务时，只通过少量 **support episode** 推断任务后验，再在相互隔离的 **query case** 上执行策略并评估。

dense 基线配置为 `configs/dense_pearl_baseline.yaml`，后验适应协议为 `configs/posterior_adaptation_protocol.yaml`，posterior-routed MoE 配置为 `configs/posterior_routed_moe.yaml`。后两者通过显式 `extends` 复用前一层配置，并在加载时解析为完整配置；checkpoint 保存的是解析后的配置哈希。

> 当前阶段的默认目标是验证主程序闭环，而不是自动拉起正式训练或做消融。`--smoke` 是唯一默认可运行的训练模式；非 smoke 的 PEARL 和 SAC 命令必须显式传入 `--formal-run`，且 PEARL 还需要匹配的 formal validation。

## 1. 实现整体设计

```text
geometry catalog
      │  build_taskbook.py（真实地图解析、哈希）
      ▼
冻结 taskbook + casebook ──► meta-train：prior rollout → posterior rollout → task replay
      │                                              │
      │                                              ▼
      │                               ContextEncoder → z ~ q(z | support)
      │                                              │
      └──────── meta-test：K 个 support ────────────┴──► actor(obs, z) → 独立 query
                                                          （无梯度、参数哈希不变）
```

训练和评估都只接受已经冻结的 taskbook/casebook，而不在运行时生成新场景。这样 `K`、新任务交互步数、query 集和 checkpoint 可以被准确复核。

## 2. 任务、场景与环境

### 2.1 冻结任务输入

- `taskbook.py` 从配置中的 `geometry_catalog` 创建 `meta_train`、`meta_validation`、`meta_test_template`、`meta_test_logical` 四个 split；`build_taskbook.py` 再在真实 MetaDrive 地图中解析并写入地图、路线和冲突区哈希。
- `casebook.py` 为每个任务生成互不重叠的 `train_pool`、`validation_support`、`validation_query`、`test_support`、`test_query`。默认每任务分别为 32、10、20、10、20 个 case；case ID 和随机种子不能跨 split 或任务复用。
- 加载 taskbook/casebook 时会校验 schema、内容 SHA-256 和 split 隔离；训练 checkpoint 也记录 taskbook/casebook/config 哈希。用不同 taskbook 评估 checkpoint 会被拒绝。

`meta_test_logical` 与 `meta_train` 的逻辑类型不重叠。任务 ID、任务哈希和模板索引均不进入策略或编码器输入。

### 2.2 环境接口

`src/task_env.py` 的 `LogicalMergeEnv` 是 Gymnasium 环境：RL 控制对抗车的二维连续动作，SUT 使用固定参数的 IDM 控制器。每回合按冻结 case 复现初始速度、对抗车出生位置与 SUT 出生位置；episode 最长 180 步。

环境通过 on-ramp、lane-drop / bottleneck、Y-merge 适配器建立实际路线和冲突区。重置时会再次比对运行时地图哈希，避免“配置相同但实际地图不同”的隐性偏差。

### 2.3 观测、奖励与严格指标

- `logical_merge_obs` 固定为 37 维、归一化到 `[-1, 1]`：两车相对冲突区的路线距离/速度/加速度/进度、到达时间差、TTC、相对速度、可见优先权，以及分支数、车道数、汇入长度、路线曲率等拓扑描述符。
- 目标接触资格规则不在观测中。元训练中同一物理几何包含不同的隐藏规则变体，使模型不能只把几何描述当作规则标签；规则只能从 support 中的可见运动、奖励和终止结果间接辨识。
- `reward.py` 将低 TTC 和近距离作为稠密奖励，对满足目标条件的 adversary–SUT 接触给出 `+200`，并惩罚非目标碰撞、出界、错误路线和过大/不平滑动作。
- `valid_critical_strict_rate` 才是主指标：query episode 必须出现目标对接触或低 TTC，且没有非目标碰撞、任一车辆出界或错误路线。它不是“任意碰撞率”。

## 3. PEARL-SAC 如何训练

核心实现在 `src/pearl_agent.py`、`src/context_encoder.py` 与 `src/pearl_trainer.py`。

1. 对每个采样 meta-train 任务，先从单位高斯先验采样 `z`，收集一个 `prior_support` episode；以该 episode 推断后验，再收集一个 `posterior_rollout` episode。两类转移都进入该任务独立 replay buffer。
2. 每个 context transition 是 `[obs, action, reward / 200, next_obs, terminated, truncated]`。编码器先对一个 episode 内的 32 条转移平均池化；每个 episode 产生一个对角高斯证据因子；再通过 Product-of-Gaussians 合并为 `q(z | context)`。
3. 默认后验维度为 5；context encoder 为 `200-200` MLP，actor 与双 Q critic 为 `256-256` MLP。actor 输入 `(obs, z)`，critic 输入 `(obs, action, z)`，因此没有任务标签的输入路径。
4. 训练时按任务采样 transition batch，并随机抽取 1 到 8 个 context episode，使编码器在 K=1、K=2 等少样本条件下也接受训练。优化目标为双 Q 的 SAC TD 损失、`0.1 × KL(q(z|c) || N(0,I))`、actor 损失和自动熵温度；target critics 用 `tau=0.005` 软更新。
5. 验证集严格指标提升时写入 `best_model.pt`。checkpoint 同时保存网络、优化器、replay、环境步数/更新计数、随机数状态及输入哈希；`last_model.pt` 仅在显式设置 `--checkpoint-interval-steps` 时写入。

后验路由 MoE 使用 actor-only residual 结构：router 仅接收真实地图初始化后冻结的 6 维物理描述符以及 `stop_gradient(mu, log_var)`；route 在一个 episode 内固定，只在 posterior version 更新后重算。actor 使用共享主干、加权残差 experts 和统一 Gaussian head，双 critic 与 context encoder 保持 dense。实现与审计入口见 [`POSTERIOR_ROUTED_MOE_IMPLEMENTATION.md`](../docs/pearl/POSTERIOR_ROUTED_MOE_IMPLEMENTATION.md)。

可选的 `--disentangled-representation` 将 5 维后验划为几何/交互/规则三块（默认 `2/2/1`），并加入辅助解码损失。这是独立候选设计，默认 PEARL 不启用；其审计输出仅是后验语义诊断，不能替代性能比较或因果解耦证明。

## 4. K-shot 无梯度适应与 query 评估

`scripts/evaluate_fewshot.py` 调用 `evaluator.evaluate_fewshot`，流程固定如下：

1. 从 `q(z)=N(0,I)` 开始。第 1 个 support 用先验采样的 `z` 执行，之后每个 support 用当前后验采样的 `z` 执行。
2. 每得到一个 support episode，按固定随机种子采样其 32 条转移，重新计算 `q(z | support_1:K)`。默认报告 `K=0, 1, 2, 5, 10`；K 表示已实际执行的 support episode 数，而不是梯度更新次数。
3. 在每个 K 下，以后验均值 `mu` 进行确定性动作，执行独立的 query case。query case 不参与 support case 选择、后验计算或任何梯度更新。
4. 评估前后对 context encoder、actor、双 critic、target critic 和温度参数计算哈希；若哈希变化立即报错。输出中的 `no_gradient_adaptation: true` 与相同的前后哈希共同证明该次适应没有微调模型。

编码器一次 context 最多使用 256 条转移，即至多 8 个每集 32 条转移的 episode。因此 K≤8 时所有已收集 support 都可进入 context；K=10 时仍执行 10 个 support episode 以计算真实交互成本，但后验从其中按固定种子抽取至多 8 个 episode 构建。K=10 的结果不应被解释为“编码器同时使用了全部 10 集证据”。

默认 support 选择为 `fixed`。代码还提供 `random`、初始条件多样性和 posterior-action-disagreement 选择策略，后两者不读取 query case、隐藏规则或未执行 rollout 的结果；它们只用于另行声明的候选实验，不能与当前固定 support 的报告混为一谈。

## 5. 基线与可比较结论

`src/baselines.py` 定义并由 `scripts/run_baselines.py` 执行以下基线：

- `scratch_sac`：针对新任务从零开始训练；
- `pooled_finetune_sac`：先训练 pooled 策略，再针对新任务微调；
- `per_task_sac`、`cross_task_policy_matrix`、`topology_conditioned_pooled_sac`：用于确认任务差异与 pooled 参考；
- `oracle_task_conditioned_sac`：显式输入几何 one-hot 的特权上界，不应与无特权方法直接比较；
- `pearl_no_context`：PEARL actor 固定使用单位高斯先验。

少样本公平比较必须使用 `run_equal_budget_analysis.py`：对每个任务，将 scratch/pooled-fine-tune SAC 的环境步数设置为 PEARL 前 K 个 support episode 的实际累计步数，并使用相同冻结 query case。较长预算的 SAC 只能回答“充分在线训练后的参考性能”，不能证明或否定 PEARL 的低交互样本效率。

当前保留结果所支持的范围与具体数值见 [`../results/pearl_learning/README.md`](../results/pearl_learning/README.md)。它们是固定 taskbook、单个选定 PEARL checkpoint 和既定预算下的比较，不应扩展为任意任务、任意预算或统计显著性的普遍结论。

## 6. 当前主程序验证模式

以下命令只验证训练、checkpoint 保存/加载、后验适应、独立 query 与无梯度约束能够闭环。它限制为 2 个 meta-train 任务、1 个验证任务、1 个未见测试任务、约 1,000 环境步和每任务 2 个 query；任何输出都不是性能结论。

```powershell
conda run -n metadrive python -m pearl_learning.scripts.train_pearl `
  --config pearl_learning/configs/dense_pearl_baseline.yaml `
  --taskbook results/pearl_learning/taskbooks `
  --casebook-root results/pearl_learning `
  --seed 0 --run-name main_flow_smoke --smoke --max-env-steps 1000 `
  --checkpoint-interval-steps 500 `
  --output-root results/pearl_learning/verification

conda run -n metadrive python -m pearl_learning.scripts.evaluate_fewshot `
  --config pearl_learning/configs/dense_pearl_baseline.yaml `
  --checkpoint results/pearl_learning/verification/smoke/main_flow_smoke/best_model.pt `
  --taskbook results/pearl_learning/taskbooks `
  --casebook-root results/pearl_learning `
  --split meta_test_logical --run-name main_flow_smoke --smoke --query-cases 2 `
  --output-root results/pearl_learning/verification
```

正式实验必须另行确认任务、随机种子、总步数、显存/内存预算和输出目录。其命令至少应同时包含 `--formal-run`，并对 PEARL 提供与 taskbook 匹配的 `--formal-validation`；省略任一条件会拒绝运行。例如：

```powershell
conda run -n metadrive python -m pearl_learning.scripts.train_pearl `
  --config <config> --taskbook <taskbook-dir> --casebook-root <casebook-root> `
  --formal-validation <formal-validation.json> --seed <seed> --run-name <run-name> `
  --max-env-steps <steps> --output-root <new-output-root> --formal-run
```

## 7. 结果、审计与测试

- few-shot 输出：`<output-root>/evaluations/<split>/<run-name>/metrics.json`；smoke 输出位于 `<output-root>/smoke/`。输出保留逐任务/逐 K 汇总、support 环境步数、选用的 support case 与 provenance，不保留逐轨迹。
- `audit_integrity.py`、`audit_topologies.py` 检查冻结输入、真实地图/路线与 split 隔离；`audit_task_heterogeneity.py` 检查正式基线的任务异质性和预算对称性。这些 audit 是前提与数据质量检查，不是关键性能结果本身。
- `audit_support.py posterior` 只产生 support 后验和参数哈希；`audit_support.py representation` 审计可选解耦表示。`audit_transferability.py`、`calibrate_transferability.py`、`decide_transferability.py` 组成候选的迁移诊断/保守拒绝接口。它们不读取测试 query 来拟合阈值，且在验证证据不足时应输出 `defer`。
- 契约测试：`conda run -n metadrive python -m pytest pearl_learning/tests -q`。当前 64 项测试同时覆盖 dense 回归以及 MoE 数值、梯度、生命周期、泄漏与 checkpoint 契约。

## 8. 目录

- `configs/dense_pearl_baseline.yaml`、`posterior_adaptation_protocol.yaml`、`posterior_routed_moe.yaml`：dense 基线、后验适应与路由 MoE 配置。
- `configs/posterior_routed_moe_pilot.yaml`、`posterior_routed_moe_specialization_pilot.yaml`：2×2 工程 pilot 与 4-expert 路由专门化诊断；两者均不是正式性能实验。
- `src/taskbook.py`、`src/casebook.py`、`src/task_env.py`：冻结任务、case 和 MetaDrive 环境。
- `src/observation.py`、`src/reward.py`、`src/metrics.py`：观测、奖励与严格安全指标。
- `src/context_encoder.py`、`src/moe.py`、`src/pearl_agent.py`、`src/pearl_trainer.py`：PEARL-SAC 与 posterior-routed MoE 核心。
- `src/evaluator.py`、`src/baselines.py`：few-shot 评估、等预算比较与基线注册。
- `scripts/`：构建、训练、评估、审计与正式实验入口。
- `tests/test_contract.py`、`tests/test_routed_moe.py`、`tests/test_routed_moe_mechanisms.py`：dense/协议回归、路由 MoE 与机制干预契约。
