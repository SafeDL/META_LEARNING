# 第二阶段目标：PEARL 用于相关逻辑场景的少样本适配

## 1. 文档定位与当前状态

本文是 `pearl_learning/` 的**当前实现说明和目标边界**，以代码为准，而不是未来功能清单。它描述在不修改 `sac_scenario_mining/` 的前提下，如何在 MetaDrive 中实现面向 merge-like 场景的 PEARL-SAC 原型。

当前代码已完成：逻辑任务/案例的冻结、四种场景入口、统一双车环境、37 维无标签观测、PEARL-SAC 更新、任务隔离回放、少样本无梯度评估、关键场景保存与动作回放，以及 Stage 1 兼容性检查。

当前代码尚未给出正式的大规模训练结果，也没有实现 per-task SAC、pooled SAC、scratch SAC 或 MoE。因此，本阶段的研究目标是建立可复现、可验证的元强化学习实验基础；尚不能据此宣称 PEARL 已经优于其他基线。

## 2. 研究目标与范围

目标是在固定 SUT、固定动作语义和统一奖励下，让对抗车策略从少量完整 support episode 推断任务后验，并在 query case 上无梯度地适配：

\[
z_i \sim q_\phi(z\mid C_i),\qquad
a_t \sim \pi_\theta(a_t\mid o_t,z_i)
\]

其中，任务差异来自相近的道路连接和汇入交互结构，而不是被测车内部参数。当前的 scenario family 为 `merge_like`；仅有一辆受策略控制的 adversary 和一辆固定 IDM SUT，不引入背景交通流。

不在本阶段范围内：修改 Stage 1 SAC、SUT 参数随机化作为 task、MoE、RSS、外部 ADS、视觉输入、环岛/信号交叉口/行人场景，以及大规模背景车辆协同。

## 3. Task、Case 与固定 SUT

### 3.1 Task 是逻辑场景模板

`LogicalScenarioTaskSpec` 位于 `pearl_learning/src/task_spec.py`，只保存以下可序列化字段：

```python
task_id: str
split: str
logical_type: str
map_config: dict
conflict_spec: dict
case_seed: int
```

`logical_type` 只用于选择环境适配器、日志和分组；不会被拼接进 Actor、Critic 或 Context Encoder。任务中的道路信息通过环境构造和连续观测中的拓扑描述符出现，而不是通过 one-hot 场景标签泄漏给网络。

当前固定的任务划分为：

| 划分 | 数量 | 逻辑类型 |
| --- | ---: | --- |
| `meta_train` | 12 | `on_ramp_merge`、`lane_drop_merge`、`bottleneck_merge` 各 4 个 |
| `meta_validation` | 3 | 三种已见类型各 1 个 |
| `meta_test_template` | 3 | 三种已见类型各 1 个 |
| `meta_test_logical` | 4 | 仅 `y_merge` |

因此，`meta_test_logical` 的 `y_merge` 不会进入元训练；它用于检验对未见逻辑类型的少样本适配。`taskbook.py` 会检查 task ID 唯一、split 一致以及 held-out 类型未泄漏到训练集，并为完整 taskbook 写入 SHA-256。

### 3.2 Case 是同一模板下的初始条件实例

每个 task 的 casebook 独立生成并保存。一个 case 只包含：

```json
{
  "case_id": "...",
  "case_seed": 123,
  "adversary_speed_mps": 12.5,
  "arrival_offset_m": -1.2
}
```

它改变对抗车初速度和 SUT 的纵向到达偏置，不改变 task 的逻辑类型。五个互不重叠的 case 集为 `train_pool`、`validation_support`、`validation_query`、`test_support` 和 `test_query`；当前默认数量分别为 32、10、20、10、20。

### 3.3 SUT 固定

环境只接收 adversary 的二维连续动作。SUT 由 `IDMPolicy` 控制，速度和换道开关固定于配置：

```json
"sut": {"enable_lane_change": false, "target_speed_mps": 12.0}
```

`MetaDriveAdapterBase.establish_roles()` 从与 adversary 不同、且有后续连接的实际车道中选择 SUT 出生车道，要求初始距离至少为 10 m。它随后将 SUT 的目标速度和禁止换道设置到 IDM policy；case 和 task 都不会随机化这些 SUT 参数。

## 4. 当前场景实现

适配器接口定义在 `pearl_learning/src/adapters/base.py`，隔离 MetaDrive 的版本相关访问，负责建图、建立双车角色、冲突框架、拓扑特征、目标碰撞判定和角色校验。

| 逻辑类型 | 当前环境实现 |
| --- | --- |
| `on_ramp_merge` | MetaDrive `MetaDriveEnv`，程序化地图 `SrS` |
| `lane_drop_merge` | 自定义 `LogicalBottleneckEnv`，瓶颈段从 3 车道缩至 2 车道 |
| `bottleneck_merge` | 同一自定义环境，瓶颈段从 3 车道缩至 1 车道 |
| `y_merge` | MetaDrive `MetaDriveEnv`，程序化地图 `r` |

所有环境均设置 `traffic_density=0`、`random_traffic=false`，因此没有背景车。`audit_topologies.py` 会在真实 MetaDrive 中对这四类代表任务逐一 reset、单步运行，并记录 lane graph、实际角色车道、冲突框架、角色间距和观测有限性；任一失败则退出，不应开始训练。

任务中的 `template_index` 和 `merge_length_m` 会被冻结、哈希并记录，其中 `merge_length_m` 进入冲突规格和观测描述符。当前 on-ramp/y-merge 适配器并未将该长度进一步传入 MetaDrive 的建图参数；所以“不同模板”在这些类型中目前不等同于已证明的不同物理道路几何。正式使用 `meta_test_template` 前，应先以 topology audit 的 lane graph 证实每个模板的物理差异。

## 5. 统一环境、观测与奖励

### 5.1 冲突参考框架

冲突原点不是全局地图端点的平均值。代码从 adversary 与 SUT 的当前/后续导航参考车道中，各采样 64 个点，选择两条实际角色路线距离最近的一对点的中点作为冲突原点。这一做法避免无关道路分支改变冲突坐标。

### 5.2 `logical_merge_obs`：37 维、无显式任务标签

观测契约定义在 `observation.py`，维度固定为 37，全部裁剪并归一化至 `[-1, 1]`。它由以下部分组成：

| 部分 | 维数 | 内容 |
| --- | ---: | --- |
| adversary 路线状态 | 8 | 到冲突区距离、速度、加速度、横向偏移、朝向误差、到达时间、路线进度、在路线标志 |
| SUT 路线状态 | 8 | 与 adversary 对称的 8 项 |
| 双车交互 | 8 | 到达时间差、欧氏距离、相对速度、闭合速度、TTC、冲突角，以及当前固定的优先级编码 |
| 路网描述符 | 13 | 入/出分支数、车道数、汇入长度、两车路线余量、冲突半径、曲率、限速、冲突区数量 |

这不是 conflict-centric 重训练版本；当前训练继续使用这一已实现的 `logical_merge_obs`。配置、环境与 checkpoint 都强制校验 schema 和 37 维长度，旧的 56 维 checkpoint 已删除且不能加载。

### 5.3 统一奖励和严格有效性

奖励在所有任务上相同，包括低 TTC 的稠密奖励、距离邻近奖励、目标双车碰撞奖励，以及非目标碰撞、出界、错误路线、动作幅值和动作突变的惩罚。

目标碰撞需同时满足两车 `crash_vehicle` 状态和车身包围盒距离阈值，避免把与第三方的碰撞误记为目标碰撞。一个 episode 的严格有效关键场景为：

\[
\texttt{valid\_critical\_strict} =
(\texttt{target\_collision} \lor \min\texttt{TTC}\leq\tau)
\land \neg(\texttt{non-target collision} \lor \texttt{out-of-road} \lor \texttt{wrong-route})
\]

汇总指标包含目标碰撞率、严格有效关键场景率、无效率、最小 TTC 中位数和首次有效关键场景所需 episode 数。

## 6. 当前 PEARL-SAC 实现

`PEARLAgent` 由一个 context encoder、Gaussian Actor、双 Q critic、双目标 Q critic 和温度参数组成。配置默认使用 5 维潜变量；context encoder 隐层为 `[200, 200]`，Actor/Critic 为 `[256, 256]`。

每条 context transition 为：

\[
(o_t,a_t,r_t/200,o_{t+1},\texttt{terminated},\texttt{truncated})
\]

context encoder 用每个 task 的 transition 集输出对角高斯后验 `(mu, log_var)`；更新时，Q 损失的梯度经过潜变量回传至 encoder，并加上到单位高斯先验的 KL 项。Actor 更新使用从后验潜变量中分离梯度的 `z`，随后更新温度和软更新目标 Q 网络。

Stage 1 SAC 权重不会被读取、初始化或微调。PEARL 从随机初始化开始学习；Stage 1 仅作为不可修改的兼容性基线。

## 7. 训练、评估与数据边界

训练脚本 `train_pearl.py` 首先构造冻结 taskbook。每个训练 task 具有独立 `TaskReplayBuffer`，缓冲区拒绝 task ID 不一致的 transition。bootstrap 和后续采集的 episode 都来自 `train_pool`，再分别采样 context batch 和 RL batch 进行更新。

正式配置每 task bootstrap 5 个 episode；`--smoke` 时降为 1 个，并缩小 batch/context 和每轮更新次数，仅用于连通性检查。训练定期在 `meta_validation` 调用 few-shot 评估，以以下字典序选择 `best_model.pt`：目标碰撞率、严格有效关键场景率、负无效率、负最小 TTC 中位数。无论是否出现验证点，都会保存 `final_model.pt`。

few-shot 评估按 shots `[0, 1, 2, 5, 10]` 顺序进行。每增加一个 shot，只从 support case 收集 episode 并更新后验；query case 使用 posterior mean 和确定性动作。评估开始与结束会计算网络权重哈希，不一致即报错，因此 query 不写入 context，也绝不更新网络权重。

checkpoint 同时保存 config hash、taskbook hash、步骤数、观测 schema/维数和动作维数。加载时若 schema、37 维观测或 2 维动作不匹配会拒绝加载，防止旧模型静默混用。

## 8. 可复现产物与操作入口

主要产物位于 `results/pearl_learning/`：

- `taskbooks/`：冻结的 task 列表和哈希；
- `casebooks/`：每个 task 的 case 列表和哈希；
- `topology_audit/`：真实 MetaDrive 的四类场景审计；
- `<run>/`：解析后的配置、版本、训练任务、训练进度、验证结果、`best_model.pt` 与 `final_model.pt`；
- `final_eval/<split>/`：few-shot 结果及 top-K 严格有效关键场景；
- 每个保存场景：`manifest.json`、`actions.npy`、`metrics.json` 和回放后的审计结果。

在仓库根目录、使用 `metadrive` conda 环境运行：

```powershell
conda run -n metadrive python -m pearl_learning.scripts.build_taskbook --config pearl_learning/configs/merge_family_pearl.yaml
conda run -n metadrive python -m pearl_learning.scripts.audit_topologies --config pearl_learning/configs/merge_family_pearl.yaml
conda run -n metadrive python -m pearl_learning.scripts.verify_stage1_compatibility

conda run -n metadrive python -m pearl_learning.scripts.train_pearl --config pearl_learning/configs/merge_family_pearl.yaml --seed 0 --max-env-steps 1000 --run-name pearl_smoke --smoke
conda run -n metadrive python -m pearl_learning.scripts.evaluate_fewshot --config pearl_learning/configs/merge_family_pearl.yaml --checkpoint <run>/best_model.pt --split meta_test_logical --query-cases 20
conda run -n metadrive python -m pearl_learning.scripts.replay --manifest <evaluation>/<task_id>/shot_5/critical_scenarios/rank_001
```

回放脚本不加载策略权重：它依据 manifest 重建 task/case，执行保存的动作序列，并检查目标碰撞一致且最小 TTC 误差不超过配置中的 `0.1` 秒。

## 9. 与 Stage 1 的隔离

`sac_scenario_mining/` 和 `results/sac_scenario_mining/` 不属于本阶段的修改范围。`verify_stage1_compatibility.py` 以 `stage1-on-ramp-sac-v1` 为基线，检查 Stage 1 验证 case `validation_000` 可以 reset、执行零动作，且仍产生 38 维 `on_ramp_merge_obs_v2` 观测。其报告保存到 `results/pearl_learning/stage1_compatibility.json`。

因此，Stage 1 的 38 维观测与 PEARL 的 37 维 `logical_merge_obs` 是两个独立契约，不能交换 checkpoint 或直接比较网络输入。

## 10. 当前结论边界与后续验收

当前实现可验证以下工程结论：双车角色可建立、四个入口可经真实 MetaDrive 审计、Task/Case 可冻结、训练/评估数据边界明确、元测试不更新权重、关键场景可离线回放，并且 Stage 1 接口未被改动。

在报告“PEARL 的少样本优势”之前，至少应完成：正式多随机种子训练、对 `meta_test_template` 与 `meta_test_logical` 的 0/1/2/5/10-shot 评估、每个 top-K 场景的回放审计，以及公平实现并比较 pooled SAC 与等环境预算 scratch SAC。若 topology audit 不能显示所称模板的真实几何差异，或基线尚未完成，就只能报告该实现为 PEARL 工程原型，不能声称跨逻辑场景的泛化效果。
