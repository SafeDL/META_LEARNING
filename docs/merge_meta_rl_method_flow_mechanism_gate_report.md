# Merge Meta-RL 机制优先 Gate 报告

状态：2026-08-17；当前机制链**停在 Gate 1B**。本报告不是论文性能结论。

## 已证实的内容

- 原始物理 Task（lane-drop 与 bottleneck）没有产生稳定的 Task-dependent 最优脚本策略；因此没有直接进入 PEARL。
- 新增 `logical_order_spatiotemporal_near_miss_v3`：严格 VCSR = v2 的无碰撞时空联合条件，加上冻结的 conflict-entry order。目标顺序不进入 24D Actor/Critic/Context observation。
- 派生的两个 Task 物理地图、路线、冲突哈希完全相同；只相反地要求 `adversary_first` 或 `sut_first`。
- mechanism Casebook 现在强制同一 condition 的 `case_seed`、速度、spawn、初始 gap 与距离全部相同，避免 MetaDrive/IDM 外生随机性混淆。
- `order_boundary` Casebook 将八个初始 signed arrival gap 限制在 `[-0.10, 0.10] s`、相对速度为 0，仍不使用任何 calibration threshold。
- strict metric 下 target collision 不再获得成功奖励；新增的 potential shaping 同时依赖 arrival gap、joint conflict distance、pair distance 和目标预期进入顺序，但它不能终止 episode 或计入 VCSR。

## Gate 1：通过

在 order-boundary Casebook 上，九个固定纵向探针的结果为：

- `adversary_first`：`P7_adversary_first_feedback` 的严格 VCSR = 0.625；
- `sut_first`：`P6_arrival_gap_heuristic` 的严格 VCSR = 0.625；
- 最优探针不同，且 100% matched conditions 的赢家改变。

证据：

- `results/pearl_learning/merge_method_flow_logical_order_boundary_mechanism/gate_1_policy_conflict/policy_conflict_gate.json`
- `results/pearl_learning/merge_method_flow_logical_order_boundary_mechanism/gate_1_policy_conflict/scripted_policy_matrix.json`

因此物理可达性与 Task-policy conflict 均存在；这不是“所有策略都一样”的问题。

## Gate 1B：失败

两个独立 SAC 各以一个 seed 训练，在相同八个 matched cases 上构造 2×2 transfer matrix。SAC 仅用于机制诊断，使用记录在 manifest 的轻量更新协议（每 8 env steps 一次梯度更新、batch 64），不代表正式 baseline。

| Casebook / budget | 严格 VCSR | Gate 1B |
| --- | ---: | --- |
| absolute grid，10k/Task，collision-aligned reward | 四格均 0.0 | fail |
| absolute grid，10k/Task，strict potential shaping | 四格均 0.0 | fail |
| order-boundary，10k/Task | 四格均 0.0 | fail |
| order-boundary，20k/Task | 四格均 0.0 | fail |

最后一轮中，SAC 会压低 arrival gap（中位 0.028--0.078 s），但 joint conflict distance 仍是 3.80--5.64 m，远高于冻结的 0.58 m 严格阈值；因此没有对角优势可报告。

权威证据：

- `results/pearl_learning/merge_method_flow_logical_order_boundary_mechanism/gate_1b_single_task_sac_20k/single_task_sac_transfer_gate.json`
- `results/pearl_learning/merge_method_flow_logical_order_boundary_mechanism/gate_1b_single_task_sac_20k/single_task_sac_transfer_matrix.json`

## 结论与约束

当前结论不是“PEARL 网络容量不够”，也不能据此声称问题定义在一般意义上不成立。更精确地说：在固定 IDM、当前 1D longitudinal control、冻结 v3 strict metric 与这些 casebooks 下，**Task-specific 脚本策略存在，但单任务 SAC 在 20k 步内仍不能学会 collision-free strict near-miss**。因此尚无资格测试 support-to-posterior 或 latent-to-action 机制。

禁止进行：Gate 2、20k PEARL、structure-aware prior、150k+ 正式训练或论文性能比较。

下一轮若继续，必须修改 Task 的可学习性而不是扩充 PEARL：例如将动作时序/可控性显式化，或在不改变 VCSR 的前提下构造经过单任务 SAC 可达性筛选的 mechanism cases；修改后从 Gate 1 重新开始。
