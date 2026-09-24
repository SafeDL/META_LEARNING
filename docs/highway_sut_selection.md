# Highway-env 异构被测驾驶算法：筛选结果与 Codex 复现任务书

> 适用项目：`SafeDL/META_LEARNING`  
> 调研与代码核验日期：2026-09-22  
> 项目核验提交：`0e57aadf2a2dde10caefa37c6d76ab82adacfe64`  
> 任务边界：增加真正执行 ego 驾驶的 SUT，不修改本文的场景挖掘算法，不从零开展大规模策略训练。

## 0. 给 Codex 的任务摘要

先复现 **IDM+MOBIL、VI、MCTS、已有权重的 DQN、已有权重的 PPO**，再按需要加入 **已有权重的 Double DQN**。保留项目现有 IDM/FVDM 作为参考，但不要将同一算法的多种参数或交通密度检查点计为不同算法家族。

学习型算法优先使用本次核实的 `paranjaa/ece-rl-highway-driving` 权重。这是公开的社区/课程研究项目，不是 Farama 或 Stable-Baselines3 官方发布的标准驾驶模型；其价值在于同时提供算法代码、具体权重、环境配置和评估入口。算法代表性与该仓库的学术级别是两回事。

**本次已经完成的是源码、配置和权重文件元数据核验；尚未下载二进制并执行模型，也未证明这些策略在本项目中驾驶合格。** 不要把“文件存在”“模型能加载”“原生环境能驾驶”“迁移到项目场景后合格”合并为一个验收状态。

优先执行以下顺序：

1. 保存本地环境与代码快照，修通独立的外部 ego 动作链。
2. 获取并检查 DQN/PPO/DDQN 权重，不运行上游训练脚本。
3. 在权重对应的原生 highway-env 配置中做小规模验证。
4. 将通过验证的冻结策略接入项目共同场景，构建新的响应库。
5. 检查跨 SUT 失效集合是否出现交叉，而不只是碰撞率高低不同。

**禁止默认行为：**覆盖现有响应库、升级原 `metadrive` 环境、遇到加载错误就重新训练、将未训练网络或主动撞车策略充当正常驾驶 SUT、根据本文方法是否获胜挑选 SUT。

---

## 1. 当前项目实际环境与接入约束

### 1.1 已核实的环境声明

来源：[项目 environment.yml](https://github.com/SafeDL/META_LEARNING/blob/0e57aadf2a2dde10caefa37c6d76ab82adacfe64/environment.yml)。这是仓库声明，不是对用户当前机器 `pip freeze` 的实测。

| 项目 | 仓库声明 |
|---|---|
| Conda 环境名 | `metadrive` |
| Python | `3.10` |
| PyTorch | `2.5.0` |
| NumPy | `2.2.6` |
| Gymnasium | `0.29.1` |
| highway-env | `1.9.1` |
| MetaDrive | `0.4.3`，本轮不改动 |
| Stable-Baselines3 | 此环境文件未列出；不代表用户机器绝对未安装 |

Codex 首先记录：Python 可执行文件、包版本、操作系统、`git rev-parse HEAD`、`git status --short`。本地代码若已更新，应以本地实际实现为准，并在报告中列出与上述核验提交的差异。

### 1.2 必须处理的四处接口问题

依据：[CutInEnv 源码](../highway_env_benchmark/envs/cutin_env.py)；[SUTProfile 与车辆实现](../sut_algorithms/highway_env/idm_profiles.py)。

| 当前实现 | 对新增 SUT 的影响 | Codex 处理目标 |
|---|---|---|
| `create_profiled_vehicle()` 只分派 IDM/FVDM | 增加一个 profile 名称并不等于接入 DQN/PPO | 新增独立 SUT registry，保留旧 profile 工厂 |
| `CutInEnv._simulate(action)` 没有执行传入的 `action`，而调用 `road.act()` | 外部策略输出可能完全不生效 | 新增外部 ego 环境路径，确保动作进入 ego 且不被自动驾驶逻辑覆盖 |
| `DiscreteMetaAction` 配置为 `longitudinal=False` | 当前动作空间不是常见的五动作驾驶接口 | 新路径提供横向和纵向均开启的五动作接口，核实动作编号 |
| ego 是自驱动的 `ProfiledIDMVehicle/ProfiledFVDMVehicle` | `road.act()` 会重新生成 ego 加速度和转向 | 外部离散策略使用合适的 `MDPVehicle`/专用车辆适配器；不要复用会覆盖动作的 profile 车辆 |

当前场景主要为两车道、7 秒、20 Hz 物理步、5 Hz 策略步。外部权重并不必然在这些条件下训练。**先原生验证，再做场景迁移；不要通过把输入补零、截断或直接改变动作含义来制造“兼容”。**

此外，当前 `collision` 来自任意相关车辆碰撞触发的终止条件。新增实验应单独保存 `ego_collision` 和 `background_collision`，避免三车场景中背景车辆之间的碰撞被算成目标 SUT 失效。旧指标与旧结果保持冻结，新结果记录独立的 `oracle_version`。

---

## 2. 按三个维度筛选后的正式名单

以下“优先级”是本项目的投入顺序，不是算法性能排名；“代表性”是所覆盖的驾驶决策机制，而非对具体社区模型能力的背书。

| 候选 SUT | 可复现性：实际资源 | 代表性 | highway-env 关系 | 本项目决策 |
|---|---|---|---|---|
| **IDM+MOBIL** | 原生源码，无训练和权重依赖 | 规则跟驰 + 自主换道 | 模拟器原生 `IDMVehicle` | **P0：首轮接入** |
| **VI** | 作者实现与有限 MDP 转换；需要少量版本适配 | 动态规划、显式交通预测 | 官方算法示例 | **P0：首轮接入** |
| **MCTS/UCT** | 作者实现；无需训练，但需要规划模型和计算预算 | 多步动作序列搜索 | 官方算法示例 | **P0：首轮尝试，明确预测模型权限** |
| **DQN-MLP** | 核实到具体 `.zip`、训练代码和配置 | 标准离散价值学习 | 权重配套代码训练于 `highway-fast-v0` | **P0：学习型首选** |
| **PPO-MLP** | 核实到具体 `.zip`、训练代码和配置 | 标准策略梯度/Actor–Critic | 权重配套代码训练于 `highway-fast-v0` | **P0：学习型首选** |
| **Double DQN** | 核实到 `policy_net.pth`、网络定义和模型侧配置 | 价值估计机制的同族对照 | 同一 highway-env 项目 | **P1：便宜的额外对照，不计成全新家族** |
| **Social/Ego-Attention DQN** | 作者 highway-env 配置已确认；本次未确认可直接下载的最终权重 | 显式车辆交互表示 | 原实验是 `intersection-v0` | **P2：保留，但不自动启动训练** |

算法实现依据：[highway-env 官方算法页](https://highway-env.farama.org/content/algorithms/)、[rl-agents](https://github.com/eleurent/rl-agents)、[学习型资源 README](https://github.com/paranjaa/ece-rl-highway-driving/blob/17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e/README.md)。

**首轮暂缓：**SAC/TD3/RecurrentPPO/CPO/Decision Transformer 的泛化库适配、ROS2 MPC、FRENETIX。不是这些方法不重要，而是本次没有同时确认“与当前任务接近的 highway-env 权重 + 清晰输入契约 + 低接入成本”；不应继续把通用算法仓库当作现成驾驶器。

---

## 3. 无需训练的三个优先候选

### 3.1 IDM+MOBIL：增加完整换道逻辑，而不是再调 IDM 参数

- 模拟器仓库：[Farama-Foundation/HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv)
- 实现：[highway_env/vehicle/behavior.py](https://github.com/Farama-Foundation/HighwayEnv/blob/main/highway_env/vehicle/behavior.py)
- 说明：[行为车辆文档](https://highway-env.farama.org/dynamics/vehicle/behavior/)
- 关键类/方法：`IDMVehicle`、`change_lane_policy()`、`mobil()`。

**复现目标：**使用项目已安装的 highway-env 1.9.1 对应实现，不下载当前主分支来替换整个模拟器。将原生 `IDMVehicle` 作为 ego，启用自主换道，并固定行为参数及随机种子。

**与现有代码的差别：**项目现有 profile 车辆主要沿固定目标车道执行跟驰控制；新增 SUT 要保留原生 MOBIL 的换道决策。二者不是简单改名。

**最低测试：**在前方慢车、相邻车道可用的场景中验证换道逻辑实际被调用；在相邻后车接近的场景中记录换道选择。不要要求任意一个人为示例都必须换道。

**成本判断：**无需 GPU、无需训练；接入成本低。这是首轮最确定的实现增量。

### 3.2 VI：有限 MDP 上的预测规划

- 算法：[value_iteration.py](https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/dynamic_programming/value_iteration.py)
- 有限 MDP 库：[eleurent/finite-mdp](https://github.com/eleurent/finite-mdp)
- highway-env 转换：[finite_mdp.py](https://github.com/Farama-Foundation/HighwayEnv/blob/main/highway_env/envs/common/finite_mdp.py)
- 使用说明：[官方算法页](https://highway-env.farama.org/content/algorithms/)

作者 `ValueIterationAgent` 在非有限 MDP 环境中调用 `env.unwrapped.to_finite_mdp()`，按当前场景重新计算动作价值。交通预测简化为其他车辆保持速度和车道。

**项目适配点：**

1. ego 必须具备有限 MDP 转换所需的速度索引等接口，优先使用 `MDPVehicle`。
2. 新环境提供 `collision_reward`、`right_lane_reward`、`high_speed_reward`、`lane_change_reward` 等配置。当前仅 `float(not crashed)` 的环境奖励不足以直接表达驾驶进展目标。
3. 核验五动作空间：转换代码中存在与五动作对应的奖励结构，不能直接套在当前三动作配置上。
4. 只做 API 兼容、状态转换和奖励配置适配；保留标准 Bellman 更新，不替换成一个自写跟驰器。

**预测模型身份：**命名为 `VI-TTC`，在 manifest 中记录预测时域、时间离散间隔和奖励配置。

### 3.3 MCTS/UCT：多步驾驶搜索

- 算法：[mcts.py](https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/tree_search/mcts.py)
- 搜索框架：[tree_search/abstract.py](https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/tree_search/abstract.py)
- 环境复制等工具：[common/factory.py](https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/common/factory.py)

**复现目标：**先在标准 highway-env 中运行作者 MCTS，随后接入项目。固定规划预算和随机种子，记录每次决策的模型步数与耗时；内部模型调用不是免费计算。

**项目中需区分两个版本：**

- `MCTS-Sim`：使用复制的完整仿真环境。若复制了 `ScheduledCutInVehicle` 的未来制动/换道日程，则它拥有特权预测信息。可作为理想化对照，但不得与只观察当前状态的策略混为相同信息条件。
- `MCTS-CV`：在规划副本中，仅根据当前车辆状态使用固定的恒速/固定预测模型，不携带未来场景日程。保留实际测试环境中的原日程。这个版本是对作者 MCTS 的项目适配，不是未经修改的原实现。

**首轮策略：**先让 `MCTS-Sim` 完成运行验证，再以有限工程投入建立 `MCTS-CV`。若预测模型适配暂时受阻，报告该状态，其他 SUT 继续执行，不阻塞整个实验。

MCTS 的 reward 应是冻结的驾驶奖励，而不是场景挖掘器的“碰撞越多越好”的奖励。不要把测试生成优化目标传给 ego 驾驶器。

---

## 4. 学习型首选：同一项目提供的 DQN、PPO、Double DQN 权重

### 4.1 资源来源与固定版本

- 项目：[paranjaa/ece-rl-highway-driving](https://github.com/paranjaa/ece-rl-highway-driving)
- 本次核验提交：[`17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e`](https://github.com/paranjaa/ece-rl-highway-driving/commit/17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e)
- 配置：[config.json](https://github.com/paranjaa/ece-rl-highway-driving/blob/17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e/config.json)
- 依赖：[requirements.txt](https://github.com/paranjaa/ece-rl-highway-driving/blob/17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e/requirements.txt)
- 评估：[benchmark.py](https://github.com/paranjaa/ece-rl-highway-driving/blob/17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e/benchmark.py)

仓库包含不同交通密度的模型目录。本轮先使用默认密度 1.5 对应的资源；其他密度权重是训练分布变化的后续对照，不是新的算法。

### 4.2 逐项核实到的权重

文件大小与 Git blob SHA 来自 GitHub Contents API；这里的 SHA 是 **Git 对象 SHA-1，不是文件 SHA-256**。附录下载程序会根据下载字节重新核验，并生成 SHA-256。

| SUT ID | 权重路径 | 文件大小（字节） | Git blob SHA |
|---|---|---:|---|
| `dqn_ece` | `models/DQN/checkpoints/dqn_model_10000000_steps.zip` | 1,232,360 | `99711d37d30f137bed3ddf7d9346ae9241fb910a` |
| `ppo_ece` | `models/PPO/vd_1_5_trial_1.zip` | 1,820,517 | `7b5da3c0b400a42f949054c904d27822416668fb` |
| `ddqn_ece` | `models/DDQN/policy_net.pth` | 303,225 | `0235026ededddc4b20384f87ad3e6f832ca2a634` |

证据目录：[DQN](https://api.github.com/repos/paranjaa/ece-rl-highway-driving/contents/models/DQN/checkpoints?ref=17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e)、[PPO](https://api.github.com/repos/paranjaa/ece-rl-highway-driving/contents/models/PPO?ref=17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e)、[DDQN](https://api.github.com/repos/paranjaa/ece-rl-highway-driving/contents/models/DDQN?ref=17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e)。三个权重合计约 3.36 MB，不包含软件依赖。

### 4.3 DQN：标准离散价值学习代表

- 训练实现：[dqn_agent.py](https://github.com/paranjaa/ece-rl-highway-driving/blob/17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e/dqn_agent.py)
- 权重页面：[DQN 检查点](https://github.com/paranjaa/ece-rl-highway-driving/blob/17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e/models/DQN/checkpoints/dqn_model_10000000_steps.zip)
- 直接下载：[DQN ZIP](https://raw.githubusercontent.com/paranjaa/ece-rl-highway-driving/17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e/models/DQN/checkpoints/dqn_model_10000000_steps.zip)
- 底层算法：[Stable-Baselines3 DQN](https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/dqn)

使用 `stable_baselines3.DQN.load(..., device="cpu")` 载入并采用 `deterministic=True` 评估。文件名和作者说明对应 10M-step 检查点；Codex 必须读取 ZIP 元数据确认，不能只凭文件名报告训练步数。

**选择理由：**有实际模型、原生 highway-env 配套配置、加载路径标准化，适合作为学习型基准。它不是 Double DQN，不能因 SB3 名称而混淆。

### 4.4 PPO：与 DQN 配对的策略梯度代表

- 训练实现：[ppo_agent_v2.py](https://github.com/paranjaa/ece-rl-highway-driving/blob/17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e/ppo_agent_v2.py)
- 权重页面：[PPO 检查点](https://github.com/paranjaa/ece-rl-highway-driving/blob/17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e/models/PPO/vd_1_5_trial_1.zip)
- 直接下载：[PPO ZIP](https://raw.githubusercontent.com/paranjaa/ece-rl-highway-driving/17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e/models/PPO/vd_1_5_trial_1.zip)
- 底层算法：[Stable-Baselines3 PPO](https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/ppo)

使用 `stable_baselines3.PPO.load(..., device="cpu")` 载入并冻结。作者脚本中声明两层 256 单元网络，但实际载入以权重元数据为准。

**重要细节：**当前训练脚本默认选择 density=2.0，并设置 `train=True`；本轮选的是默认目录下 density=1.5 命名的模型。不能直接运行脚本来“测试”，也不能把当前脚本的所有参数无条件当作这个历史模型的准确训练配置。将模型元数据、模型侧配置、评估配置和训练脚本覆盖项一并记录；存在差异时标为待确认。

**选择理由：**在共享资源体系下提供与 DQN 不同的学习机制，且无须新增训练。不能预设它必然比 DQN 更安全或失效区域更不同。

### 4.5 Double DQN：低额外成本的补充

- 训练及网络定义：[ddqn_agent.py](https://github.com/paranjaa/ece-rl-highway-driving/blob/17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e/ddqn_agent.py)
- 模型侧配置：[models/DDQN/config.json](https://github.com/paranjaa/ece-rl-highway-driving/blob/17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e/models/DDQN/config.json)
- 权重页面：[policy_net.pth](https://github.com/paranjaa/ece-rl-highway-driving/blob/17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e/models/DDQN/policy_net.pth)
- 直接下载：[DDQN policy 权重](https://raw.githubusercontent.com/paranjaa/ece-rl-highway-driving/17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e/models/DDQN/policy_net.pth)

作者 `QNetwork` 为 `obs_dim → 256 → 256 → n_actions`，隐藏激活 ReLU，模块名为 `net`。推理只需要 policy network；不要误用 target network。复用或按许可证提取网络定义，严格检查 state-dict 键与维度，冻结后选择 Q 值最大动作。

这可以是最轻的 PyTorch-only 推理接入之一：优先在项目现有 PyTorch 中尝试 `torch.load(..., weights_only=True, map_location="cpu")`；若格式或版本不兼容，再放到隔离进程。不要因为网络简单就换成随机初始化的同结构网络。

### 4.6 这组模型最关键的输入契约

配套配置声明的是 **5×6 Kinematics**，而非通常被口头描述的 5×5：

```json
{
  "observation": {
    "type": "Kinematics",
    "vehicles_count": 5,
    "features": ["x", "y", "vx", "vy", "sin_h", "cos_h"],
    "features_range": {
      "x": [-50, 50],
      "y": [-50, 50],
      "vx": [-40, 40],
      "vy": [-40, 40]
    },
    "absolute": false,
    "order": "sorted",
    "normalize": true,
    "see_behind": true
  },
  "action": {"type": "DiscreteMetaAction"}
}
```

这只是已核实的配置内容；权重实际空间仍由 Codex 加载核验。要恢复 ego 行、邻车顺序、归一化、空行填充、角度特征、后方车辆可见性和有效感知范围的语义，不能仅检查数组维度。

作者完整配置还有四车道、30 辆车、40 秒、`simulation_frequency=5`、`policy_frequency=3`。其中 5/3 不是整数，需记录所用 highway-env 版本中每次决策实际执行的物理子步数及累计时间。**不要悄悄把它“修正”后仍称原生复现。**

### 4.7 依赖不兼容：不要把作者 requirements 安装进现有主环境

| 包 | 当前项目声明 | 该资源仓库 requirements 声明 |
|---|---|---|
| highway-env | 1.9.1 | 1.10.2 |
| Gymnasium | 0.29.1 | 1.2.1 |
| PyTorch | 2.5.0 | 2.9.0 |
| NumPy | 2.2.6 | 2.2.6 |
| Stable-Baselines3 | 环境文件未列出 | 2.7.0 |

这些是源码仓库声明；每个检查点保存时的版本还应读取 ZIP 的 `_stable_baselines3_version`、`system_info.txt`。

采用两步策略：

- **原生验证环境**：独立创建 `hw-sut-native`，根据权重元数据和作者 requirements 安装依赖，CPU 推理即可。保存全部解析后的包版本。
- **项目共同场景**：保持主模拟器 highway-env 1.9.1。可先尝试轻量推理适配；遇到依赖冲突则使用独立常驻策略进程，通过标准输入/输出 JSON 传递观测和动作。不要每个仿真步重新启动 Python。

跨进程只解决软件依赖，**不会自动解决观测/动作/动力学语义差异**。这些必须由适配器显式处理。

---

## 5. 两个已核实有权重的备份来源

这两项不是追加的新算法家族，而是在首选模型不可用或普通驾驶不合格时，提供独立的 DQN/PPO 来源。

### 5.1 ABZ 2025 Case Study 的 DQN

- 项目：[hhu-stups/abz2025_casestudy_autonomous_driving](https://github.com/hhu-stups/abz2025_casestudy_autonomous_driving)
- 配套多车道脚本：[HighwayEnvironment_Base.py](https://github.com/hhu-stups/abz2025_casestudy_autonomous_driving/blob/main/HighwayEnvironment_Base.py)
- 单车道脚本：[HighwayEnvironment_Single_Base.py](https://github.com/hhu-stups/abz2025_casestudy_autonomous_driving/blob/main/HighwayEnvironment_Single_Base.py)
- 多车道正常驾驶权重：[base/new/trained_model.zip](https://github.com/hhu-stups/abz2025_casestudy_autonomous_driving/blob/main/base/new/trained_model.zip)
- 直接下载：[正常驾驶 DQN](https://raw.githubusercontent.com/hhu-stups/abz2025_casestudy_autonomous_driving/main/base/new/trained_model.zip)
- 依赖：[requirements.txt](https://github.com/hhu-stups/abz2025_casestudy_autonomous_driving/blob/main/requirements.txt)

核实到的文件：1,206,798 字节，Git blob SHA `73fb9f0b9892ddee67b6537446f3e58b35963c96`。

配套依赖：`highway_env==1.8.2`、`gymnasium==0.28.1`、`numpy==1.25.0`、`stable_baselines3==2.0.0`。脚本使用 `highway-fast-v0`，多车道 base 为 3 车道，`target_speeds=[0,5,10,15,20,25,30,35,40]`。

**适用性：**来源于命名明确的安全控制案例，正常模型与环境配置可追踪；但较旧依赖及不同目标速度网格要求独立适配。

**不要选择错误资源：**本轮只使用 `base`；`adversarial` 不能仅凭有权重就当作正常安全驾驶器。`single_base` 的单车道/禁横向动作契约与多车道版本不同，不能互换。README 中通用的 `highway_agent.py` 命令也不能替代实际存在的上述脚本。

### 5.2 独立社区项目的 SB3 PPO

- 当前项目地址：[saibhargavpokala/Reinforcement-learning-Validating-Safety-Autonomous-vehicles-Highway-env-D2RL](https://github.com/saibhargavpokala/Reinforcement-learning-Validating-Safety-Autonomous-vehicles-Highway-env-D2RL)
- 训练脚本：[1_Environment_steup.py](https://github.com/saibhargavpokala/Reinforcement-learning-Validating-Safety-Autonomous-vehicles-Highway-env-D2RL/blob/main/1_Environment_steup.py)
- 权重：[highway_ppo/model.zip](https://github.com/saibhargavpokala/Reinforcement-learning-Validating-Safety-Autonomous-vehicles-Highway-env-D2RL/blob/main/highway_ppo/model.zip)
- 直接下载：[PPO ZIP](https://raw.githubusercontent.com/saibhargavpokala/Reinforcement-learning-Validating-Safety-Autonomous-vehicles-Highway-env-D2RL/main/highway_ppo/model.zip)
- 已展开的环境元数据：[system_info.txt](https://github.com/saibhargavpokala/Reinforcement-learning-Validating-Safety-Autonomous-vehicles-Highway-env-D2RL/blob/main/highway_ppo/model/system_info.txt)
- 已展开的模型元数据：[data](https://github.com/saibhargavpokala/Reinforcement-learning-Validating-Safety-Autonomous-vehicles-Highway-env-D2RL/blob/main/highway_ppo/model/data)

核实到的 ZIP：1,786,975 字节，Git blob SHA `aaa883c7184c9d8d7ff38cf8ae9aa3417ef7f318`。

作者展开的模型信息声明：Python 3.9.0、SB3 2.1.0、PyTorch 2.1.0+cpu、NumPy 1.26.1、Gymnasium 0.28.1；未在这个文件中列出 highway-env 的精确版本。模型数据声明训练步数 20,000、网络 `[256,256]`；训练脚本创建默认 `highway-fast-v0`。

**选择地位：**备用，不优先于有模型侧配置/统一评估体系的首选项目。训练步数少不自动证明策略无效，但不能假设其已经具有可靠驾驶能力。

**直接运行风险：**原脚本 `TRAIN_MODE=True`，评估循环没有正确合并 `terminated/truncated`，且依赖显式环境注册。应写纯评估包装器，不直接运行原 `main()`。本轮只复用 PPO，不复现该仓库整套 D2RL/模型扰动流程；也不将它宣称为原始 D2RL 论文的官方实现。

---

## 6. 暂缓清单：保留地址，但不让 Codex 自动投入

| 候选 | 参考地址 | 本轮暂缓理由 |
|---|---|---|
| Social Attention DQN | [作者项目](https://eleurent.github.io/social-attention/)；[配置](https://github.com/eleurent/rl-agents/blob/master/scripts/configs/IntersectionEnv/agents/DQNAgent/ego_attention_2h.json) | 实验与配置明确，但未核实最终权重；且是交叉口，不是当前高速切入 |
| SAC/TD3 | [SB3 SAC](https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/sac)；[TD3](https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/td3) | 仅有通用实现不足以当作现成 highway-env 驾驶器；连续动作还增加一套控制契约 |
| RecurrentPPO | [SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/tree/master/sb3_contrib/ppo_recurrent) | 尚未确认匹配当前任务的公开驾驶权重；不能默认重新训练 |
| MPC/FRENETIX | [Batch-Opt-Highway-Driving](https://github.com/vivek-uka/Batch-Opt-Highway-Driving)；[FRENETIX](https://github.com/TUM-AVS/Frenetix-Motion-Planner) | 前者是 ROS2 自定义 highway simulator，后者使用 CommonRoad；都不是即插即用的 highway-env |
| CPO | [SafetyRL_HighwayEnv](https://github.com/varunjain3/SafetyRL_HighwayEnv)；[OmniSafe](https://github.com/PKU-Alignment/omnisafe) | 检索到旧 TensorFlow 检查点和混合任务目录，但未核实满足本任务的完整低成本契约；通用库仍需训练 |
| DT/离线 RL | [d3rlpy](https://github.com/takuseno/d3rlpy) | 本次未确认能直接使用的高速驾驶最终权重；场景级响应矩阵不是离线 RL 的逐步轨迹数据 |

不作“这些算法没有公开权重”的绝对判断，只说明**本次筛选没有核实足够条件，故不纳入默认执行目标**。

---

## 7. Codex 实现规格：保持旧实验不变，新增一条 SUT 复现链

### 7.1 实现目录

最终实现遵循项目统一目录约定：环境归入共享 highway-env 底座，SUT 独立归入根目录 `sut_algorithms/`，实验入口归入 `replications/`，正式结果归入 `results/highway_replications/`。

```text
highway_env_benchmark/
  envs/external_cutin.py

sut_algorithms/highway_env/
  base.py
  idm_profiles.py
  idm_mobil.py
  value_iteration.py
  mcts_cv.py
  ppo_ece.py
  registry.py

replications/highway_sut_selection/
  README.md
  assets.py
  runner.py
  cli.py
  tests/

results/highway_replications/sut_selection/
  provenance/
  native_validation/
  common_validation/
  response_banks/
  risk_structure/
  report.md

external_assets/highway_sut/     # 按需生成的权重缓存；不提交二进制至仓库
```

不改 `replications/*` 下已经实现的测试/挖掘方法，不覆盖 `results/highway_replications/` 和已有 `method_chains` 结果。

### 7.2 每个 SUT 都必须保存的身份信息

```yaml
sut_id: ppo_ece
algorithm_family: policy_gradient
algorithm_name: PPO
source_repository: paranjaa/ece-rl-highway-driving
source_commit: 17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e
checkpoint_path: models/PPO/vd_1_5_trial_1.zip
checkpoint_git_blob_sha: 7b5da3c0b400a42f949054c904d27822416668fb
checkpoint_sha256: null  # 下载后生成，不能编造
weights_status: remote_file_metadata_verified
runtime_status: not_tested
training_env_id: highway-fast-v0
training_env_version: null  # 1.10.2 是配套 requirements 声明；模型保存版本另行核查
runtime_env_version: null
observation_contract: ece_kinematics_5x6
native_action_contract: discrete_meta_5
native_target_speeds: null  # 恢复原配置/默认值后记录
policy_frequency_hz: 3
physical_clock_audit: pending
deterministic_eval: true
checkpoint_modified: false
fine_tuned: false
license_status: to_record
```

状态至少区分 `remote_file_metadata_verified`、`download_hash_verified`、`loaded`、`native_evaluated`、`common_evaluated`、`rejected`。原生通过不代表共同场景也通过。

### 7.3 外部 ego 动作链的实现约束

新环境可继承当前 `CutInEnv` 或复用其场景/背景车辆逻辑，但外部 ego 创建与动作执行必须独立。最小可行结构如下：

```text
共享场景参数与背景车辆
        ↓
构造 ego 当前观测（按各自训练契约）
        ↓
冻结的 SUT / 规划器给出动作
        ↓
按该 SUT 动作字典解释动作、更新目标车道/速度
        ↓
固定物理步推进；背景车辆继续执行相同的场景脚本
        ↓
记录 ego 动作、轨迹、碰撞与连续安全响应
```

必须有以下小测试：

- 无障碍场景下，外部 `FASTER` 与 `SLOWER` 导致不同 ego 速度轨迹。
- 合法条件下，`LANE_LEFT/LANE_RIGHT` 能改变目标车道，且不存在后续 `road.act()` 覆盖。
- 对同一初始种子，不同 SUT 使用相同外部交通初始条件；脚本式背景交通不因模型推理随机数而改变。
- 不同 SUT 的全零/填充观测规则符合各自原契约；5×5 与 5×6 不能共用错误编码器。
- 每个 episode 重置模型/规划树/观测缓存；SUT 不在测试中学习或调参。
- 不涉及 ego 的碰撞不计为 `ego_collision`。

### 7.4 控制能力与频率：先保留身份，再明确共同协议

不要为凑齐统一数组而改变 ABZ 的九档目标速度或首选模型的原动作意义。`target_speeds`、可用动作、底层跟踪器都是 SUT 契约的一部分。

原生阶段严格保存原时钟。共同场景阶段建议另建清晰的时间配置，例如 30 Hz 物理步、3 Hz 策略步，并使每个新 SUT 使用相同外部场景和物理参数。该配置是**建议的新协议，不是原生复现事实**，必须先做普通驾驶验证；使用旧 IDM/FVDM 时也在这个新协议下重新执行，不能复用旧 20/5 Hz 响应。

若原生时钟审计表明 checkpoint 实际物理动作保持时间与声明值不一致，单独记录并采用清晰的 native/common 两协议；不要将时钟改变造成的能力崩溃解释成算法固有漏洞。过度复杂时，首轮可只选择时序适配成功的候选，其他标记 `adaptation_failed`。

### 7.5 依赖隔离与安全加载

- 原 `metadrive` 环境保持不变；先 `conda list`/`pip freeze` 存档。
- 首选模型的原生验证放独立环境；PyTorch 采用 CPU 安装即可，不要求新增 GPU。
- 上游 `requirements.txt` 并不一定等同于每个检查点保存时版本，先读元数据，再选择可复现组合。
- `rl-agents` 涉及历史接口，优先小范围兼容层，不直接升级主环境或导入整套训练依赖。
- 第三方 ZIP 可能含 pickle/cloudpickle；下载检查阶段只读 ZIP 中的文本/JSON，不反序列化可执行对象。真正加载放到隔离的复现环境，DDQN 优先 `weights_only=True`。
- 记录各仓库的 LICENSE/权重许可；公开可访问不等于任意再分发。保留来源，不自动将权重重新发布到用户仓库。

---

## 8. 分阶段运行与验收：默认零新增策略训练

### 阶段 A：环境与权重检查

目标：完成 snapshot、具体文件下载、哈希核验、检查点文本信息抽取、上游源码固定版本。

不执行 `.learn()`；不直接运行可能带 `train=True` 的上游脚本。网络下载失败记录为 `asset_unavailable`，不能用随机模型填补。

### 阶段 B：原生运行验证

DQN、PPO 各先运行 20 个固定种子的原生 episode；DDQN 为可选。评估 wrapper 必须使用 `terminated or truncated` 结束，设最大步数，关闭 GUI，允许按需保存少量视频。

记录：观测和动作空间、累计 reward、ego 碰撞、实际行驶距离、平均速度、动作直方图、episode 实际物理时长、推理时间及异常。

这里不是严格复现上游论文数值；目的在于确认 checkpoint 与配套环境形成一个真实可运行的驾驶器。不能只看 reward，因为不同资源的奖励尺度不同。

### 阶段 C：共同场景上的资格与风险结构试验

默认新增 SUT：IDM+MOBIL、VI、MCTS、DQN、PPO；另选项目现有 IDM、FVDM 各一个作为参考，共 7 个。

每个 SUT 先执行 20 个普通驾驶资格场景，再执行相同的 96 个挑战场景。优先复用现有六种场景生成器，各 16 个场景；确保场景池包含可行驶区域和中等挑战区域，而不全是明显必撞条件。不要根据本文挖掘方法的结果筛选场景或 SUT。

预算：`7 × (20 + 96) = 812` 次共同环境 episode；加 DQN/PPO 原生验证 40 次，约 **852 次**。加入 DDQN 增加原生 20 次与共同 116 次，总计约 **988 次**。这些是建议预算，不包含模型内部规划子步、调试重跑和单元测试；日志应另行累计真实成本。

资格判据以接口正确、能完成无障碍驾驶、没有持续原地不动/无效动作等明显退化为主。普通交通表现同时对比固定随机和简单基准，但不要为了凑齐算法而反复微调 checkpoint。20 个样本只适合初筛，不足以宣称安全性已充分验证。

### 阶段 D：回应 source→target 风险结构是否仍过于一致

输出下列结果，不继续训练场景搜索器：

1. 每个 SUT 的 ego collision、near-miss、进展与失效时间分布。
2. 两两风险 Spearman 相关性；对常数风险向量返回 `NA`，不强行生成相关系数。
3. 两两失效集合 Jaccard，以及 `|F_a \ F_b|`、`|F_b \ F_a|`。双方都存在独有失效比单纯碰撞率不同更有诊断价值。
4. 按场景功能分别统计，而不是只看总体相关性。
5. 将全部安全、全部失效、部分 SUT 失效的场景分开描述；“部分失效”分组只供事后结构审计，不能把完整目标标签交给挖掘器。

所有方法必须沿相同场景列构建响应矩阵。两两失效集合差异是经验行为证据，不直接等同于不同软件缺陷或不同根因。

### 阶段 E：只有通过后才进入论文方法实验

输出冻结的 `response_bank`、SUT manifest 和共同协议。此时才重新运行已有 AdaTE、Mining、Posterior Search/CoRe-Mine 等选择器。原生验证结果和本轮开发场景已经用于筛选，应与后续独立确认场景分开。

---

## 9. Codex 运行入口与交付物

以下为 **Codex 要创建并实现的命令契约**，不是现有项目已支持的命令：

```bash
# 1. 记录环境；后续命令均不应隐式训练
python -m replications.highway_sut_selection.cli snapshot

# 2. 下载并静态检查首选三个权重
python -m replications.highway_sut_selection.cli fetch
python -m replications.highway_sut_selection.cli inspect

# 3. 在独立原生环境验证 DQN/PPO；DDQN 按需要追加
python -m replications.highway_sut_selection.cli native --suts ppo_ece --episodes 20

# 4. 修通新环境后，检查接口与共同场景
python -m pytest replications/highway_sut_selection/tests -q
python -m replications.highway_sut_selection.cli qualify --episodes 20
python -m replications.highway_sut_selection.cli common --scenarios 96
python -m replications.highway_sut_selection.cli audit
```

对于跨环境 worker，CLI 支持显式 `--policy-python`，由对应 Python 可执行文件启动常驻推理进程；日志走 stderr，协议消息走 stdout，避免污染 JSON 通信。

必须交付：

| 文件 | 内容 |
|---|---|
| `provenance/environment_snapshot.json` | 实际安装环境与代码版本，不是只复制 requirements |
| `provenance/assets.json` | URL、提交、原路径、大小、Git blob SHA、下载 SHA-256、许可记录 |
| `provenance/compatibility.md` | 每个 SUT 的 native/common 环境、输入/动作/时钟差异与处理方式 |
| `native_validation/episodes.csv` | 原生运行证据；含失败模型，不只保存成功模型 |
| `common_validation/episodes.csv` | 共同场景逐次结果与成本 |
| `response_banks/*.npz` + manifest | 新 SUT×场景响应矩阵；不覆盖旧库 |
| `risk_structure/pairwise.csv` | 风险相关性、集合交集及双方独有失效 |
| `risk_structure/by_function.csv` | 按功能拆分的结构分析 |
| `report.md` | 实际接入了谁、谁未通过、原因、计算成本、是否形成有价值的风险差异 |

**投入限制：**默认不训练任何新策略；每个候选进行一次依赖/接口兼容处理和一次原生验证。失败后保留问题记录，转下一个或指定备用来源，而不是无限尝试版本。一个候选失败不能阻塞其他已具备资源的候选。

---

## 附录 A：可直接交给 Codex 的权重下载与静态检查代码

将下列代码保存为 `fetch_highway_sut_assets.py`。它只下载、核验与读取普通文本，不加载策略，不执行上游代码。代码需要 Codex 所在环境能访问 GitHub；本次对话环境未实际运行网络下载部分。

```python
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

PRIMARY_REPO = "paranjaa/ece-rl-highway-driving"
PRIMARY_REF = "17eff0e0809ea7f7dbf8cc33bef7c3dbeba89b8e"
BACKUP_PPO_REPO = (
    "saibhargavpokala/"
    "Reinforcement-learning-Validating-Safety-Autonomous-vehicles-Highway-env-D2RL"
)
ASSETS = [
    dict(id="dqn_ece", group="primary", repo=PRIMARY_REPO, ref=PRIMARY_REF,
         path="models/DQN/checkpoints/dqn_model_10000000_steps.zip",
         size=1232360, blob_sha="99711d37d30f137bed3ddf7d9346ae9241fb910a"),
    dict(id="ppo_ece", group="primary", repo=PRIMARY_REPO, ref=PRIMARY_REF,
         path="models/PPO/vd_1_5_trial_1.zip",
         size=1820517, blob_sha="7b5da3c0b400a42f949054c904d27822416668fb"),
    dict(id="ddqn_ece", group="primary", repo=PRIMARY_REPO, ref=PRIMARY_REF,
         path="models/DDQN/policy_net.pth",
         size=303225, blob_sha="0235026ededddc4b20384f87ad3e6f832ca2a634"),
    dict(id="dqn_abz", group="backup",
         repo="hhu-stups/abz2025_casestudy_autonomous_driving", ref="main",
         path="base/new/trained_model.zip",
         size=1206798, blob_sha="73fb9f0b9892ddee67b6537446f3e58b35963c96"),
    dict(id="ppo_community", group="backup", repo=BACKUP_PPO_REPO, ref="main",
         path="highway_ppo/model.zip",
         size=1786975, blob_sha="aaa883c7184c9d8d7ff38cf8ae9aa3417ef7f318"),
]


def inspect_zip(payload: bytes) -> dict:
    """Inspect SB3 text metadata only. Never unpickle serialized values."""
    result = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        result["archive_members"] = archive.namelist()
        for name in ("_stable_baselines3_version", "system_info.txt", "data"):
            if name not in archive.namelist():
                continue
            if archive.getinfo(name).file_size > 2_000_000:
                raise ValueError(f"Unexpected metadata size: {name}")
            text = archive.read(name).decode("utf-8")
            if name == "data":
                data = json.loads(text)
                # Keep plain fields; do not deserialize :serialized: objects.
                result["plain_training_fields"] = {
                    key: data.get(key) for key in (
                        "num_timesteps", "seed", "policy_kwargs", "gamma",
                        "learning_rate", "n_steps", "batch_size"
                    )
                }
            else:
                result[name] = text
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("external_assets/highway_sut"))
    parser.add_argument("--group", choices=("primary", "backup", "all"),
                        default="primary")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    failed = False

    for asset in ASSETS:
        if args.group != "all" and asset["group"] != args.group:
            continue
        url = (f"https://raw.githubusercontent.com/{asset['repo']}/"
               f"{asset['ref']}/{asset['path']}")
        record = {**asset, "url": url, "runtime_status": "not_tested"}
        destination = args.out / asset["id"] / Path(asset["path"]).name
        try:
            if destination.exists():
                payload = destination.read_bytes()
            else:
                request = Request(url, headers={"User-Agent": "Highway-SUT-Reproduction"})
                with urlopen(request, timeout=60) as response:
                    payload = response.read(int(asset["size"]) + 1)
            if len(payload) != asset["size"]:
                raise ValueError("File size differs from verified metadata")
            header = f"blob {len(payload)}\0".encode("ascii")
            git_sha = hashlib.sha1(header + payload).hexdigest()
            if git_sha != asset["blob_sha"]:
                raise ValueError("Git blob SHA mismatch; do not silently accept changed weights")
            record["sha256"] = hashlib.sha256(payload).hexdigest()
            if destination.suffix == ".zip":
                record["static_metadata"] = inspect_zip(payload)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            record["local_path"] = str(destination.resolve())
            record["weights_status"] = "download_hash_verified"
            print(f"OK: {asset['id']} ({len(payload)} bytes)")
        except Exception as exc:
            failed = True
            record["weights_status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"FAILED: {asset['id']}: {exc}")
        records.append(record)

    manifest_path = args.out / f"manifest_{args.group}.json"
    manifest_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Manifest: {manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

运行：

```bash
python fetch_highway_sut_assets.py --group primary
# 仅在首选资源不能使用时追加：
python fetch_highway_sut_assets.py --group backup
```

备用资源暂用 `main`，但以已核验 blob SHA 限定具体权重；Codex 还需固定其配套源码提交。任一文件更新导致哈希不匹配时，记录并重新核验来源，不能直接删除检查逻辑。

---

## 附录 B：必须形成的最终筛选结论

本轮完成后，报告不要只写“成功复现 N 个模型”。应填入下表：

| SUT | 下载/源码可用 | 原生运行合格 | 共同场景运行合格 | 是否新增独有失效区域 | 是否纳入正式实验 | 原因 |
|---|---|---|---|---|---|---|
| IDM+MOBIL | 待执行 | 待执行 | 待执行 | 待测 | 待定 | — |
| VI-TTC | 待执行 | 待执行 | 待执行 | 待测 | 待定 | — |
| MCTS-CV / MCTS-Sim | 待执行 | 待执行 | 待执行 | 待测 | 待定 | 必须区分预测信息权限 |
| DQN-ECE | 已确认远端权重元数据 | 待执行 | 待执行 | 待测 | 待定 | — |
| PPO-ECE | 已确认远端权重元数据 | 待执行 | 待执行 | 待测 | 待定 | — |
| DDQN-ECE（可选） | 已确认远端权重元数据 | 待执行 | 待执行 | 待测 | 待定 | 与 DQN 同属价值学习家族 |

**最终投入原则：优先复用真实权重和成熟控制代码；用共同场景上的行为与风险差异决定其研究价值，不用训练量、仓库名称或模型文件数量代替证据。**
