# 第一阶段实现目标：MetaDrive + SAC 危险场景生成基线

> **文档用途**：直接交给 Codex，作为第一阶段代码实现、测试和验收的唯一目标说明。  
> **目标仓库**：`SafeDL/META_LEARNING`  
> **核心问题**：先证明“在一个固定逻辑场景族内，SAC 能否学习到比随机策略更有效的危险交互策略”。  
> **默认逻辑场景**：`merge`  
> **默认算法**：Stable-Baselines3 SAC  
> **参考代码**：本地 `ref_code/metadrive-scenario`，只读借鉴，不直接修改。
> **GPU环境**：本地: conda activate metadrive
> **metadrive仿真软件环境**：本地: F:\PyCharm 2024.3.2\work\metadrive
---

## 0. Codex 执行约束

在修改代码前，必须先阅读并理解以下文件：

- `README.md`
- `configs/default.yaml`
- `src/env_wrapper.py`
- `src/adapt_sac.py`
- `src/networks.py`
- `scripts/run_sanity.py`
- `ref_code/metadrive-scenario/metadrive_scenario/examples/run_scenarios.py`
- `ref_code/metadrive-scenario/metadrive_scenario/utils/env_create_utils.py`

实施时必须遵守：

1. **本阶段不实现 PEARL、MAML、MoE、RSS、跨拓扑元学习或外部自动驾驶栈。**
2. 不删除现有 FOMAML、PPO、SAC、TD3 和结果分析代码；新实现应尽量隔离在 `src/stage1/`、`scripts/stage1_*` 和独立配置中。
3. `ref_code/metadrive-scenario` 仅作为只读参考：
   - 可借鉴其环境创建、场景索引、随机种子、场景回放和配置组织方式；
   - 不允许修改该目录；
   - 不允许把大型数据集复制进主仓库；
   - 不要照搬其旧版 `gym` 四元组接口，最终接口必须符合当前项目使用的 Gymnasium 五元组接口。
4. **主实验不得静默回退到 toy environment。**
   - 缺少 MetaDrive 时，主训练和主评估必须给出明确错误并退出；
   - toy environment 只允许通过显式 `--use-toy-env` 用于无仿真器的快速单元测试；
   - toy environment 的结果不得计入第一阶段验收。
5. 先保证环境语义、车辆角色、动作含义和指标正确，再开始长时间训练。
6. 首版只使用结构化状态，不使用 RGB、激光点云神经网络或视觉模型。
7. 所有随机性必须由显式种子控制；训练、评估和场景回放均需记录种子。
8. 首选复用成熟的 Stable-Baselines3 SAC，避免本阶段同时调试环境和自研 SAC 算法。
9. 代码应具有类型标注、清晰错误信息、最小必要注释和可自动运行的测试。
10. Codex 不得为了“让测试通过”而伪造性能结果、硬编码成功场景或跳过真实 MetaDrive 集成测试。

---

## 1. 第一阶段的科学目标

第一阶段只验证下面一个假设：

> 在同一类逻辑场景（比如， `merge` ）及其多个随机变体中，一个控制对抗车辆的 SAC 策略，能够比随机动作策略更高效地生成与固定被测驾驶策略发生安全关键交互的轨迹场景。

这里的“危险场景生成”在第一阶段采用最小可实现定义：

- 固定一个被测车辆，即 SUT；
- SUT 使用固定、不可训练的规则策略，例如 IDM；
- SAC 控制一辆对抗车辆；
- 对抗车辆在每个仿真步输出控制动作；
- 一整个闭环交互轨迹构成一个生成的测试场景；
- 根据目标车辆碰撞、最小 TTC、最小距离和无效行为判定场景质量。

第一阶段的成果是后续 PEARL 的可靠工程底座，而不是最终论文方法。

---

## 2. 当前代码可以复用的部分与必须修复的问题

### 2.1 可复用部分

当前仓库已经包含：

- Gymnasium 风格的 MetaDrive 环境封装思路；
- 38 维结构化观测的初始设计；
- TTC 和碰撞奖励的初始实现；
- SAC 的 Actor、Critic 和 replay buffer 代码；
- 训练结果 JSON 和模型保存逻辑；
- sanity 脚本和 YAML 配置组织方式。

可复用这些设计思想，但不能默认其行为已经正确。

### 2.2 本阶段必须解决的已知问题

#### A. 动作语义必须纠正并固定

MetaDrive 连续外部动作的语义应按当前安装版本核验。首版统一规定：

```text
action[0] = steering
action[1] = throttle / brake
```

两维均处于 `[-1, 1]`，直接按 MetaDrive 的标准外部动作语义传入，不再把动作错误映射成超出范围的物理加速度和转角。

必须提供测试证明：

- 改变 `action[0]` 主要改变转向；
- 改变 `action[1]` 主要改变加速或制动；
- 动作只影响对抗车辆，不直接控制 SUT。

#### B. 对抗车辆和 SUT 的角色必须稳定

当前实现中“default agent”“NPC”“ego”“最近交通车辆”的语义容易混淆。本阶段必须明确：

- `adversary_vehicle`：唯一接受 SAC 动作的车辆；
- `sut_vehicle`：固定规则策略控制的目标车辆；
- 两个角色在一个 episode 内不得交换；
- reset 后必须记录两者的对象 ID、策略类型和初始状态；
- 如果无法找到合法 SUT，应重新 reset，超过有限次数后明确报错；
- 不得把“与任意车辆碰撞”等同于“与 SUT 碰撞”。

建议将所有 MetaDrive 版本相关的车辆访问逻辑隔离在一个兼容层中，例如：

```text
src/stage1/metadrive_compat.py
```

若必须访问 MetaDrive 内部字段，应集中封装并写清兼容性说明，不要把内部 API 散布在训练代码中。

#### C. 碰撞指标必须按 episode 计数

当前训练循环可能在碰撞持续多个仿真帧时重复累加碰撞次数。新实现必须：

- 每个 episode 的 `target_collision` 只能计为 0 或 1；
- `collision_rate` 必须是“发生目标碰撞的 episode 数 / 总 episode 数”；
- 同时区分：
  - `target_collision`
  - `non_target_vehicle_collision`
  - `object_collision`
  - `adversary_out_of_road`

#### D. 必须有独立评估集

训练过程中的 return 不能代替最终性能。

需要固定：

- 训练场景种子集合；
- 验证场景种子集合；
- 测试场景种子集合；
- 随机基线与 SAC 使用完全相同的测试种子和 episode 数。

#### E. 观测归一化必须可解释且确定

不得使用未初始化或全局可变的均值方差作为隐式归一化。

首版使用每个字段的固定物理尺度归一化，并将结果裁剪到 `[-1, 1]`。

#### F. 主结果必须来自真实 MetaDrive

toy environment 只做快速代码检查，不参与任何性能比较。

---

## 3. 本阶段范围

### 3.1 必须完成

- 单一 `merge` 逻辑场景族；
- 多个程序化 seed 形成同一逻辑场景的变体；
- 固定 SUT；
- 一辆可控对抗车辆；
- Gymnasium 环境；
- Stable-Baselines3 SAC；
- Random policy 基线；
- 独立训练、验证和测试流程；
- episode 级危险性与有效性指标；
- 危险场景清单、动作轨迹保存和确定性回放；
- 最小单元测试与真实 MetaDrive 集成测试；
- 可重复执行的命令和 README。

### 3.2 明确不做

- PEARL、MAML、RL² 或其他元强化学习；
- MoE、Router 或多个 Experts；
- RSS 责任模型；
- Waymo、nuScenes 等大规模真实数据；
- 多拓扑联合训练；
- held-out topology few-shot；
- RGB、视觉生成和 SimGen；
- CARLA、TCP、Pylot、OpenPilot；
- 多辆同时由 RL 控制的对抗车辆；
- 从零生成完整地图或整组轨迹；
- 与论文相关的完整消融实验。

### 3.3 可选扩展，不得阻塞验收

若本地已经有 `metadrive-scenario` 数据，可额外实现：

```text
--scenario-source dataset
```

但默认和验收路径必须使用无需下载数据的程序化 MetaDrive 场景：

```text
--scenario-source procedural
```

---

## 4. 环境与 MDP 定义

建议实现：

```python
class Stage1AdversarialMergeEnv(gymnasium.Env):
    ...
```

### 4.1 车辆角色

- **Adversary**：SAC 控制的车辆。
- **SUT**：固定 IDM 或同等级规则策略控制的车辆。
- **Background traffic**：MetaDrive 默认交通策略控制，不参与学习。

每次 reset 后，`info` 至少包含：

```python
{
    "scenario_seed": int,
    "adversary_id": str,
    "sut_id": str,
    "adversary_policy": str,
    "sut_policy": str,
    "topology": "merge",
}
```

### 4.2 动作空间

```python
spaces.Box(
    low=np.array([-1.0, -1.0], dtype=np.float32),
    high=np.array([1.0, 1.0], dtype=np.float32),
)
```

语义固定为：

```text
action[0] = steering
action[1] = throttle / brake
```

不允许在不同模块中交换动作顺序。

### 4.3 观测空间

沿用并修正 38 维结构化状态，定义显式的 `OBS_FIELDS` 常量。

建议版本 `stage1_obs_v1`：

| 区间 | 内容 | 说明 |
|---|---|---|
| `[0:6]` | 对抗车辆状态 | `vx, vy, heading, yaw_rate, delta_x_from_spawn, delta_y_from_spawn` |
| `[6:12]` | SUT 相对状态 | `dx, dy, dvx, dvy, heading_rel, distance` |
| `[12:30]` | 其余 3 辆最近交通车 | 每辆 6 维，与 SUT 相对状态字段一致 |
| `[30:38]` | 道路结构 | 4 个后续路线方向、lane width、curvature、num lanes、merge flag |

要求：

1. SUT 始终放在第一个相对车辆槽位 `[6:12]`；
2. 不使用世界坐标绝对位置，使用相对出生点位移；
3. 每个字段使用固定物理尺度归一化；
4. 缺失车辆使用零填充，并可选附加到 `info` 的 mask；
5. 返回 `np.float32`；
6. 所有值必须有限并裁剪到 `[-1, 1]`；
7. 观测维度必须由环境空间校验，训练代码不得重复硬编码；
8. 将观测 schema 版本写入输出配置和场景 manifest。

建议物理尺度：

```yaml
normalization:
  velocity: 30.0
  lateral_velocity: 10.0
  heading: 3.1415926
  yaw_rate: 2.0
  longitudinal_distance: 100.0
  lateral_distance: 20.0
  relative_distance: 50.0
  lane_width: 5.0
  curvature: 0.2
  num_lanes: 4.0
```

具体字段尺度允许按 MetaDrive 实测修正，但必须写入配置和测试。

### 4.4 TTC

至少实现相对运动近似 TTC：

\[
\dot d =
\frac{\Delta p^\top \Delta v}{\|\Delta p\|}
\]

当 \(\dot d < 0\) 时：

\[
TTC =
\frac{\|\Delta p\|}{-\dot d}
\]

要求：

- 与 SUT 单独计算；
- 无接近趋势时返回配置中的 `ttc_cap`；
- 重叠时返回 0；
- 对缺失车辆和数值异常有明确处理；
- `info` 中记录当前 TTC 和 episode 内最小 TTC。

### 4.5 奖励函数

奖励必须组件化实现，例如：

```python
@dataclass
class RewardBreakdown:
    ttc_reward: float
    proximity_reward: float
    target_collision_bonus: float
    non_target_collision_penalty: float
    out_of_road_penalty: float
    reverse_penalty: float
    action_penalty: float
    action_smoothness_penalty: float
    total: float
```

建议形式：

\[
r_t =
w_{\mathrm{ttc}} r_{\mathrm{ttc}}
+
w_{\mathrm{prox}} r_{\mathrm{prox}}
+
w_{\mathrm{col}} I_{\mathrm{target\ collision}}
-
w_{\mathrm{other}} I_{\mathrm{other\ collision}}
-
w_{\mathrm{road}} I_{\mathrm{offroad}}
-
w_{\mathrm{reverse}} I_{\mathrm{reverse}}
-
w_a \|a_t\|_2^2
-
w_{\Delta a}\|a_t-a_{t-1}\|_2^2
\]

其中：

\[
r_{\mathrm{ttc}}
=
\mathrm{clip}
\left(
\frac{T_{\mathrm{dense}}-TTC}{T_{\mathrm{dense}}},
0,
1
\right)
\]

\[
r_{\mathrm{prox}} = \exp(-d/d_0)
\]

首版默认配置可使用：

```yaml
reward:
  ttc_dense_threshold: 3.0
  ttc_weight: 1.0
  proximity_scale: 10.0
  proximity_weight: 0.1
  target_collision_bonus: 10.0
  non_target_collision_penalty: 3.0
  out_of_road_penalty: 5.0
  reverse_penalty: 1.0
  action_l2_weight: 0.01
  action_smoothness_weight: 0.02
```

要求：

- 目标碰撞 bonus 每个 episode 只触发一次；
- 每个 reward component 写入 `info["reward_components"]`；
- 奖励参数全部来自 YAML；
- RSS 不在本阶段实现；
- 奖励不能把“驶出道路后碰撞”视为有效成功。

### 4.6 终止条件

`terminated=True`：

- 与 SUT 发生目标碰撞；
- 对抗车辆驶出道路；
- 对抗车辆发生严重非目标碰撞；
- SUT 丢失且无法恢复；
- 出现不可继续的仿真异常。

`truncated=True`：

- 达到 `horizon`。

必须区分 `terminated` 和 `truncated`，不得全部压成一个 `done` 后丢失原因。

### 4.7 关键场景与有效场景定义

配置：

```yaml
evaluation:
  critical_ttc_threshold: 1.5
```

定义：

```python
critical = target_collision or min_ttc <= critical_ttc_threshold
```

首版有效性定义：

```python
valid = (
    sut_was_present
    and not adversary_out_of_road_before_critical_event
    and not wrong_way_before_critical_event
    and not non_target_collision_before_critical_event
)
```

主指标：

```python
valid_critical = critical and valid
```

第一阶段不把该指标表述为“责任归因”或“ego fault”，只称为：

```text
valid critical scenario
```

---

## 5. 场景划分

只使用 `merge` 拓扑，但划分不同程序化种子。

建议默认：

```yaml
scenario_split:
  train:
    start_seed: 0
    num_scenarios: 40
  validation:
    start_seed: 1000
    num_scenarios: 10
  test:
    start_seed: 2000
    num_scenarios: 20
```

要求：

- 三个集合不得重叠；
- split 写入最终结果；
- 训练时只能采样 train seeds；
- 模型选择只能使用 validation seeds；
- 最终表格只能使用 test seeds；
- Random 与 SAC 使用相同 test seeds；
- 每个 test seed 可重复若干 episode，但重复次数必须配置化。

---

## 6. 推荐目录结构

在不破坏现有代码的前提下，新增：

```text
configs/
└── stage1_merge_sac.yaml

src/
└── stage1/
    ├── __init__.py
    ├── env.py
    ├── metadrive_compat.py
    ├── observation.py
    ├── reward.py
    ├── metrics.py
    ├── scenario_manifest.py
    ├── sb3_callbacks.py
    └── utils.py

scripts/
├── stage1_sanity.py
├── stage1_train_sac.py
├── stage1_evaluate.py
├── stage1_compare.py
└── stage1_replay.py

tests/
└── stage1/
    ├── test_action_contract.py
    ├── test_observation.py
    ├── test_reward.py
    ├── test_metrics.py
    ├── test_manifest.py
    ├── test_env_integration.py
    └── test_determinism.py

docs/
└── STAGE1_USAGE.md
```

如现有项目已经采用其他一致目录约定，可做等价调整，但必须保证：

- 环境、训练、评估、回放相互解耦；
- 训练脚本中不包含 MetaDrive 内部对象访问细节；
- 评估代码不复用训练时的统计状态；
- 场景 manifest 具有版本号。

---

## 7. Stable-Baselines3 SAC 实现要求

主训练使用 Stable-Baselines3：

```python
from stable_baselines3 import SAC
```

默认配置建议：

```yaml
sac:
  policy: MlpPolicy
  learning_rate: 3.0e-4
  buffer_size: 100000
  learning_starts: 5000
  batch_size: 256
  tau: 0.005
  gamma: 0.99
  train_freq: 1
  gradient_steps: 1
  ent_coef: auto
  total_timesteps: 100000
  policy_hidden_sizes: [256, 256]
```

要求：

1. 使用单独 validation env 选择 best model；
2. 保存 final model 和 best model；
3. 支持 CPU 与 CUDA；
4. 支持 `--seed`；
5. 支持 `--total-timesteps` 覆盖 YAML；
6. 记录 TensorBoard 或 CSV 日志；
7. 每次运行保存完整配置副本；
8. 记录 Python、PyTorch、SB3、Gymnasium、MetaDrive 版本；
9. 记录当前 Git commit，无法取得时写 `unknown`；
10. 不使用现有 FOMAML encoder、joint encoder 或 `init_mode`；
11. 不以训练 return 作为唯一模型选择依据；
12. 模型选择优先依据 validation `valid_critical_rate`，其次依据 `median_min_ttc`。

现有自研 `src/adapt_sac.py` 可以保留，允许后续作为 parity check，但不作为本阶段主结果来源。

---

## 8. 基线与评估

### 8.1 必须实现的策略

- `random`：从动作空间均匀采样；
- `zero`：输出 `[0, 0]`，用于检查环境自然行为；
- `sac_best`：validation 选出的最佳 SAC；
- `sac_final`：训练结束的最终 SAC，仅作诊断。

### 8.2 每个 episode 必须记录的指标

```text
scenario_seed
episode_index
episode_return
episode_length
target_collision
any_vehicle_collision
non_target_collision
object_collision
adversary_out_of_road
sut_out_of_road
min_ttc
min_distance
critical
valid
valid_critical
mean_action_l2
mean_action_delta_l2
termination_reason
wall_clock_seconds
```

### 8.3 汇总指标

主指标：

```text
valid_critical_rate
```

次指标：

```text
target_collision_rate
critical_rate
invalid_rate
median_min_ttc
mean_min_ttc
median_min_distance
mean_episode_return
episodes_to_first_valid_critical
```

不得仅报告 collision rate 或 return。

### 8.4 公平比较

- 使用完全相同的 test seeds；
- 使用相同 episode 数；
- SAC 使用 deterministic action；
- 随机策略的每次运行使用已记录的 policy seed；
- 至少训练 3 个 SAC seeds：`0, 1, 2`；
- 汇总平均值、标准差和每个 seed 的原始结果。

---

## 9. 危险场景保存与回放

每个测试 episode 保存轻量记录。对 top-K 危险场景保存完整 manifest。

建议目录：

```text
results/sac_scenario_mining/<run_id>/
├── config_resolved.yaml
├── versions.json
├── train_monitor.csv
├── best_model.zip
├── final_model.zip
├── eval/
│   ├── episodes.csv
│   ├── summary.json
│   ├── comparison.csv
│   └── plots/
└── critical_scenarios/
    ├── rank_001/
    │   ├── manifest.json
    │   ├── actions.npy
    │   └── metrics.json
    └── ...
```

`manifest.json` 至少包含：

```json
{
  "schema_version": "stage1_manifest_v1",
  "observation_schema": "stage1_obs_v1",
  "topology": "merge",
  "scenario_source": "procedural",
  "scenario_seed": 2000,
  "policy_name": "sac_best",
  "policy_seed": 0,
  "adversary_id": "...",
  "sut_id": "...",
  "env_config": {},
  "termination_reason": "...",
  "metrics": {}
}
```

`stage1_replay.py` 应支持：

```bash
python scripts/stage1_replay.py \
  --manifest results/sac_scenario_mining/.../critical_scenarios/rank_001/manifest.json \
  --render topdown
```

回放时：

- 重建相同 seed 和配置；
- 使用保存的 action trace，不重新调用神经网络；
- 输出原始指标与回放指标差异；
- 碰撞标志应一致；
- `min_ttc` 允许小幅数值误差，默认容差 `0.1 s`；
- 若无法完全确定性复现，必须在结果中显式标记，而不是静默通过。

---

## 10. 实现里程碑

### M0：仓库审计与实现计划

Codex 首先输出或写入实现说明：

- 当前 MetaDrive、Gymnasium 和 SB3 版本；
- 当前环境动作语义；
- 当前车辆角色分配方式；
- `ref_code/metadrive-scenario` 中实际借鉴的文件和模式；
- 计划新增和修改的文件。

通过条件：

- 不运行长训练；
- 明确指出当前动作顺序、角色识别、碰撞计数和 toy fallback 风险；
- 给出最小改动路径。

### M1：真实 MetaDrive 环境合同

完成：

- `Stage1AdversarialMergeEnv`；
- 稳定的 adversary/SUT 角色；
- 38 维观测；
- 标准动作顺序；
- terminated/truncated；
- TTC；
- 明确的 info 字段。

通过命令：

```bash
python scripts/stage1_sanity.py \
  --config configs/stage1_merge_sac.yaml \
  --real-env \
  --steps 200
```

通过条件：

- reset 和 200 步执行无异常；
- 观测 shape、dtype、范围正确；
- 动作 space 正确；
- 每步 reward 为有限标量；
- adversary 和 SUT ID 在 episode 内稳定；
- 至少成功完成 5 个真实 MetaDrive episodes。

### M2：奖励、指标与基线

完成：

- 组件化奖励；
- episode 级指标聚合；
- Random 和 Zero 基线；
- test split；
- CSV/JSON 输出。

通过命令：

```bash
python scripts/stage1_evaluate.py \
  --policy random \
  --config configs/stage1_merge_sac.yaml \
  --split test \
  --episodes 20 \
  --seed 0
```

通过条件：

- 输出 `episodes.csv` 和 `summary.json`；
- collision rate 在 `[0, 1]`；
- episode 数量准确；
- 不重复按仿真帧计数碰撞；
- 每个 episode 均有 termination reason；
- 相同 seed 重复运行时环境初始状态一致。

### M3：SAC 训练闭环

先完成 smoke run：

```bash
python scripts/stage1_train_sac.py \
  --config configs/stage1_merge_sac.yaml \
  --seed 0 \
  --total-timesteps 2000 \
  --run-name smoke
```

再完成正式训练：

```bash
python scripts/stage1_train_sac.py \
  --config configs/stage1_merge_sac.yaml \
  --seed 0 \
  --run-name merge_sac_seed0
```

通过条件：

- smoke run 在 CPU 上可完成；
- replay buffer 正常开始学习；
- best/final model 均可加载；
- validation callback 正常写结果；
- 训练中不存在 NaN/Inf；
- 训练关闭后 MetaDrive 资源被正确释放。

### M4：独立测试、比较与回放

依次运行：

```bash
python scripts/stage1_evaluate.py \
  --policy random \
  --config configs/stage1_merge_sac.yaml \
  --split test \
  --episodes 100 \
  --seed 0
```

```bash
python scripts/stage1_evaluate.py \
  --policy-path results/sac_scenario_mining/merge_sac_seed0/best_model.zip \
  --config configs/stage1_merge_sac.yaml \
  --split test \
  --episodes 100 \
  --deterministic
```

```bash
python scripts/stage1_compare.py \
  --results-root results/sac_scenario_mining/final_eval_v2
```

通过条件：

- 生成 Random 与 SAC 对比表；
- 生成学习曲线、min TTC 分布和 valid critical rate 图；
- 保存 top-10 危险场景；
- 至少对 top-10 中的场景执行一次 action-trace 回放；
- 回放报告明确显示复现成功或失败原因。

### M5：测试与文档

必须提供：

```bash
pytest -q tests/stage1
```

测试分层：

- 不需要 MetaDrive 的单元测试：
  - 观测归一化；
  - TTC；
  - reward components；
  - episode metrics；
  - manifest schema。
- 需要 MetaDrive 的集成测试，使用 marker：
  - 环境 reset/step；
  - action contract；
  - vehicle role stability；
  - deterministic reset；
  - action-trace replay。

建议：

```bash
pytest -q tests/stage1 -m "not metadrive"
pytest -q tests/stage1 -m metadrive
```

---

## 11. 第一阶段验收标准

第一阶段只有同时满足以下硬性条件才算完成。

### 11.1 工程正确性

- [ ] 真实 MetaDrive sanity 通过；
- [ ] 无静默 toy fallback；
- [ ] 动作顺序确认为 `[steering, throttle/brake]`；
- [ ] adversary 和 SUT 角色稳定；
- [ ] 目标碰撞与非目标碰撞分开记录；
- [ ] episode 级碰撞率不可能超过 1；
- [ ] 观测维度、字段、归一化方式有版本记录；
- [ ] 训练、验证、测试 seed 不重叠；
- [ ] 同一 manifest 可回放；
- [ ] 所有 Stage 1 测试通过。

### 11.2 学习有效性

使用 3 个训练 seed，在 held-out test 场景上比较 SAC 与 Random。

至少满足以下一项，并且结果在至少 2/3 的训练 seed 上成立：

1. SAC 的 `valid_critical_rate` 比 Random 高至少 **10 个百分点**；
2. SAC 的 `median_min_ttc` 比 Random 低至少 **20%**。

同时要求：

- `invalid_rate <= 25%`；
- 性能提升不能主要来自驶出道路、逆行或非目标碰撞；
- 所有原始 episode 结果均被保存，不只保存汇总均值。

如果 100k steps 不足以达到阈值，允许先诊断和调节奖励，但不得无限增加模型复杂度。需要在结果中明确写出失败原因。

### 11.3 可复现性

- [ ] 记录完整配置、版本和 Git commit；
- [ ] 至少保存 top-10 危险场景；
- [ ] top-10 中至少 8 个场景的 critical 标志可通过 action trace 重现；
- [ ] 碰撞标志重放一致；
- [ ] `min_ttc` 重放误差默认不超过 `0.1 s`，若受仿真器数值差异影响，应提供实测容差和说明。

---

## 12. 进入第二阶段 PEARL 的 Go / No-Go 条件

只有第一阶段通过后才进入 PEARL。

### GO

同时满足：

1. SAC 在单一 `merge` 任务族上显著优于 Random；
2. 环境接口、观测和动作语义稳定；
3. 危险场景可保存和回放；
4. 关键指标可按 episode 正确统计；
5. 生成结果不是以大量无效越界或非目标碰撞为代价；
6. 不同 merge seeds 已经形成可配置的任务变体。

### NO-GO

出现以下任一情况时，不应立即加入 PEARL：

- SAC 与 Random 无显著差异；
- 动作实际上控制了错误车辆；
- SUT 在 episode 中发生角色漂移；
- 碰撞率计算不可靠；
- 奖励主要鼓励驶出道路或自杀式碰撞；
- 同一 seed 和动作轨迹无法基本复现；
- 真实 MetaDrive 训练频繁崩溃或资源泄漏；
- 只能在 toy environment 中得到正结果。

NO-GO 时优先修复环境、奖励、角色或评估，不得通过加入元学习或 MoE 掩盖底座问题。

---

## 13. 建议配置文件骨架

创建 `configs/stage1_merge_sac.yaml`，至少包含：

```yaml
project:
  stage: stage1
  run_name: merge_sac
  output_root: results/sac_scenario_mining

environment:
  topology: merge
  scenario_source: procedural
  horizon: 500
  traffic_density: 0.2
  use_render: false
  action_check: true
  max_reset_retries: 5
  observation_schema: stage1_obs_v1
  manifest_schema: stage1_manifest_v1

scenario_split:
  train:
    start_seed: 0
    num_scenarios: 40
  validation:
    start_seed: 1000
    num_scenarios: 10
  test:
    start_seed: 2000
    num_scenarios: 20

normalization:
  velocity: 30.0
  lateral_velocity: 10.0
  heading: 3.1415926
  yaw_rate: 2.0
  longitudinal_distance: 100.0
  lateral_distance: 20.0
  relative_distance: 50.0
  lane_width: 5.0
  curvature: 0.2
  num_lanes: 4.0

reward:
  ttc_cap: 5.0
  ttc_dense_threshold: 3.0
  ttc_weight: 1.0
  proximity_scale: 10.0
  proximity_weight: 0.1
  target_collision_bonus: 10.0
  non_target_collision_penalty: 3.0
  out_of_road_penalty: 5.0
  reverse_penalty: 1.0
  action_l2_weight: 0.01
  action_smoothness_weight: 0.02

evaluation:
  critical_ttc_threshold: 1.5
  episodes_per_policy: 100
  top_k_scenarios: 10
  replay_ttc_tolerance: 0.1
  deterministic_policy: true

sac:
  policy: MlpPolicy
  learning_rate: 0.0003
  buffer_size: 100000
  learning_starts: 5000
  batch_size: 256
  tau: 0.005
  gamma: 0.99
  train_freq: 1
  gradient_steps: 1
  ent_coef: auto
  total_timesteps: 100000
  policy_hidden_sizes: [256, 256]

experiment:
  training_seeds: [0, 1, 2]
  device: auto
  save_replay_actions: true
  save_topdown_video: false
```

Codex 可以根据本地 MetaDrive 版本调整不兼容的配置键，但必须在变更说明中列出调整原因。

---

## 14. 必须支持的命令

以下命令形式可做等价调整，但功能必须保留。

### 环境检查

```bash
python scripts/stage1_sanity.py \
  --config configs/stage1_merge_sac.yaml \
  --real-env
```

### 随机基线

```bash
python scripts/stage1_evaluate.py \
  --policy random \
  --config configs/stage1_merge_sac.yaml \
  --split test \
  --episodes 100 \
  --seed 0
```

### SAC 训练

```bash
python scripts/stage1_train_sac.py \
  --config configs/stage1_merge_sac.yaml \
  --seed 0 \
  --run-name merge_sac_seed0
```

### SAC 测试

```bash
python scripts/stage1_evaluate.py \
  --policy-path results/sac_scenario_mining/merge_sac_seed0/best_model.zip \
  --config configs/stage1_merge_sac.yaml \
  --split test \
  --episodes 100 \
  --deterministic
```

### 结果比较

```bash
python scripts/stage1_compare.py \
  --results-root results/sac_scenario_mining/final_eval_v2
```

### 危险场景回放

```bash
python scripts/stage1_replay.py \
  --manifest <path-to-manifest.json> \
  --render topdown
```

---

## 15. Codex 完成实现后的最终回复格式

Codex 完成任务后，必须在回复中提供：

1. **实现摘要**
   - 做了什么；
   - 没有做什么；
   - 使用了 `ref_code/metadrive-scenario` 的哪些设计。

2. **文件变更**
   - 新增文件；
   - 修改文件；
   - 每个文件的职责。

3. **环境合同**
   - adversary 和 SUT 如何识别；
   - 动作顺序；
   - 观测字段；
   - reward 公式；
   - termination 条件。

4. **实际执行命令**
   - sanity；
   - 单元测试；
   - MetaDrive 集成测试；
   - smoke training；
   - 正式训练和评估。

5. **实际结果**
   - 测试通过数量；
   - Random 指标；
   - SAC 指标；
   - 3 个训练 seed 的结果；
   - 是否满足验收阈值。

6. **复现产物**
   - 模型路径；
   - summary 路径；
   - episodes.csv 路径；
   - top-K manifest 路径；
   - replay 结果。

7. **剩余风险**
   - MetaDrive 版本依赖；
   - 非确定性；
   - 奖励作弊；
   - 车辆角色识别的内部 API 依赖；
   - 尚未进入 PEARL 的原因或进入条件。

---

## 16. 一句话目标

> **先在真实 MetaDrive 的单一 merge 任务族中，建立一个角色明确、动作正确、指标可信、可回放的 SAC 危险场景生成基线，并证明它在 held-out seeds 上优于随机策略；在此之前不引入 PEARL 或 MoE。**
