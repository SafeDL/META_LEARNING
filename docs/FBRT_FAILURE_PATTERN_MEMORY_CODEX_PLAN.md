# FBRT：面向版本回归与多驾驶策略测试的失效模式记忆
## Codex 详细实施方案、最小实验集与验收目标（V2：场景目录增补）

**编制日期：2026-09-26**  
**核对的代码基线：** `SafeDL/META_LEARNING@6af8a502c90235063e593d2d36c8aede3aa027f2`  
**仓库：** https://github.com/SafeDL/META_LEARNING  
**本轮工作名：** `FBRT-Memory`。保留 FBRT 研究主线，不再另起一个不相连的研究项目。  
**执行方式：** 优先复用现有实测数据；CPU 上建模；不训练驾驶策略；只执行选定的五类场景，完整候选目录见第 23—31 节。  
**文档状态：** 已核对现有实现及报告；下文标为“新增／本轮”的设计是待 Codex 实施的目标，不是已取得的实验结果。

> **交付目标：把“历史失败点附近排序”升级为“失效模式的存储、跨系统利用、目标反馈更新与再次复用”。**
>
> 直接完成工程和约定的两个实验。不以额外查新、所有指标同时提升、统计显著性、每个修改覆盖多个功能、或者某个模块必须胜出为前置门槛。结果不理想也必须交付完整实现、逐任务结果与明确结论；不再自动增加实验直到得到正结果。

---

## V2 增补说明：文献与规范支持的候选场景目录

**增补日期：2026-09-26。**本文件是在原实施方案基础上的完整增补版；第 0—22 节保留方法、实现、成本和验收设计，第 23—31 节新增场景与参数依据。不是已经执行的新实验报告。

**阅读优先级：**第 23—31 节覆盖第 6 节中尚未执行的新场景范围、实验 B 的场景配方以及“所有 SUT 都适用所有场景”的假设；不修改已归档实测结果，不改变第 8—10 节核心模型，不增加 NoMemory 之外的消融，不提高 **400 次新增物理执行总上限**。旧配置作为 `legacy_exact` 保留，不把旧结果改写成新范围的结果。

本次将候选目录扩充为 **14 类场景**，但一次小银行仍最多执行 **5 类 × 16 个样本 × 4 个系统配置**。核心变化是：先明确被测功能和动作能力，再选择交互与范围；不是把所有候选全部跑一遍，也不是不断加大对抗强度。

**三个证据层级：**`SOURCE_VALUE` 为已核对来源中的值；`SOURCE_STRUCTURE` 为来源支持的交互结构；`RESEARCH_RANGE` 为本项目为了连续场景测试而提出的范围。国标、地方标准、Euro NCAP 协议和论文实验配置分别标记，不能相互冒充。

**特别修正：**当前 PPO 的外部动作链可能只有 20/25/30 m/s 三档目标速度。这样的策略不是停车控制器；不能通过向它强塞一车道刹停任务来评价细微回归。第 27 节给出无需重新训练的能力检查和替代配方。

**目录导航：**23 来源与选场景原则；24 单位、TTC 与事件时刻；25 十四类场景卡；26 参数依赖与轨迹；27 SUT 匹配与默认配方；28 有限预算下的组织；29 机器可读目录；30 增补验收与 Codex 指令；31 参考资料。

---

## 0. 给 Codex 的一页任务摘要

### 0.1 核心问题

同一场景在不同驾驶系统或软件版本上可能产生不同结果。如何把过去的失败、附近的成功案例和各系统的响应差异，组织成一个可积累的失效模式库，用更少的新系统测试发现回归或危险场景？

本轮处理两种使用模式：

- **版本回归 `regression`：** 最近参考版本通过，新版本失败。候选依照最近参考版本的执行结果确定，历史记忆允许包含更早的失败。
- **多驾驶策略 `cross_agent`：** 分别测试多个独立驾驶策略，将前一个策略已经付费获得的结果用于下一个策略。不是多辆学习型车辆在同一次仿真中协同训练。

### 0.2 必须完成的五项改动

| 改动 | 本轮具体决定 |
|---|---|
| 统一执行入口 | FBRT 不再硬绑定 `LocalFaultIDMVehicle`；支持现有 Profiled-IDM、原生 IDM+MOBIL 和 PPO-ECE |
| 失效模式记忆 | 保存失败证据、通过对照、局部范围、交互描述、各历史系统的出现记录；不是只保存一个中点与法向 |
| 可表达系统差异的模型 | 用失效模式中心构造 RBF 特征，以层次化贝叶斯逻辑回归实现目标适配；替代强制一维边界平移 |
| 新失败可以入库 | 无旧边界时也能预测和探索；发现新失败后能够建立新区域，并持久化供下一个系统使用 |
| 必要实验 | A：现有银行零新增仿真重放；B：一个小型异构系统银行，同时检验回归与跨 agent 复用 |

### 0.3 固定实施范围

| 项目 | 本轮上限／约定 |
|---|---|
| 主方法 | 一个：`FBRT-Memory`；不再开发 Static/Adaptive/RegionBandit 等多个新变体 |
| 核心模型 | 每功能一个小型贝叶斯逻辑模型，历史 RBF 模式特征＋覆盖型 RBF 特征；不使用深度网络 |
| 主场景 | 旧银行保留四类；新银行由第 27 节能力匹配规则选择五类，完整候选目录不全部运行 |
| 现有数据实验 A | 使用现有 480 次参考与 960 次目标实测记录；不为它重新运行仿真 |
| 新数据实验 B | 最多 5 场景 × 16 样本 × 4 个系统配置 = 320 次 episode；配方与不适用处理见第 27 节 |
| 其他物理开销 | 功能 smoke ≤24，配对回放 ≤8，修复／必要复测预留 ≤48 |
| **全部新增物理上限** | **400 次 episode，硬上限；不含已经存在的缓存** |
| 查询预算 | 主预算 B=20；同一轨迹报告 @1、@5、@10、@20；旧银行可附带 @50，不新增实验 |
| 必要消融 | 仅一项：`FBRT-NoMemory`，用于回答历史记忆是否真正有用；复用同一银行，零新增物理 |
| 调整次数 | 一次有记录的实现／参数修订；不得开启超参数搜索、反复换种子或不断加强故障 |
| 交付原则 | 工程完成与算法获胜分开。完整报告不是只输出 `gate_failed` 后结束 |

### 0.4 不做的事

不切换 MetaDrive/CARLA；不新增 SAC/PPO 训练；不重现所有候选驾驶算法；不增加视觉传感器；不做整套国标认证；不要求真实生产版本；不开展消融矩阵、多个核函数比较、十几组种子、完整无偏事故率估计、故障根因证明或模型校准专项研究。

本文件覆盖此前任务书中与本轮冲突的执行顺序、验收门槛和实验规模；保留历史结果，不覆盖旧报告。

---

## 1. 当前证据与本轮改动的对应关系

### 1.1 已确认的问题，不再重新做“是否存在问题”的实验

| 当前实现／结果 | 已有依据 | 本轮处理 |
|---|---|---|
| FBRT 实验实际只调用一个 IDM 参考和三个局部修改 | `fbrt_env.py`、`experiment.py` | 接通原生 IDM+MOBIL 与已存在的 PPO；不再重新找权重 |
| 四个模板主要通过同一纵向控制链产生差异 | `idm_profiles.py` 与四模板执行器 | 保留四模板；只增加一个真正考察换道后车检查的场景 |
| `slow_front_brake2` 对部分模板大面积失败 | `summary_by_template.csv` | 保留为明显退化对照，汇总按任务等权，不能由其碰撞总量支配结论 |
| 停—保持—起步的参考记录没有失败，因此无旧 patch | 当前 core 报告 | 用所有合法历史系统的失败创建模式；并保留不依赖历史失败中心的背景特征 |
| patch 是最近异标签点对，并强制沿单一方向平移 | `boundary_memory.py`、`boundary_shift.py` | 保留点对作为边界证据，不再将中点／连线当作精确边界与法向 |
| 新发现失败主要更新分数，没有形成可复用的新模式 | `selectors.py` | 增加当前会话模式、新区域创建、保存／载入与下一个 agent 的记忆更新 |
| ART 首点按 `scenario_id` 排序，可能优先选 `core_boundary` | `selectors.py` 与候选编号 | 所有平局采用独立固定 RNG；编号不参与评分或初始化优先级 |
| 当前 @5 多种简单历史方法已达到 9/9 检测 | 当前 core 报告 | 不再要求在这个饱和指标上取得大增益；输出完整早期发现曲线与逐任务结果 |

源码与结果链接见文末 [C1]—[C10]。以上是已有证据；后续模型、SUT 变更和预算均为本任务的设计决定。

### 1.2 已有数据没有证明和没有否定的内容

已有结果表明历史引导有实用性，但尚未表明精细边界建模优于简单历史排序。它们没有否定跨版本失效知识的价值，也没有验证跨 agent 的持续记忆。本轮不再用“一个新的几何评分没胜出”代替对整个研究问题的判断。

---

## 2. Failure-Based Testing 如何进入方法，而不是只进入名称

陈宗岳教授的 Failure-Based Testing 强调利用失效模式的形状、大小、方向、数量等信息；Search for Boundary 研究从已知失败出发识别失效区域。[R1][R2]

本轮只吸收其中可落地的部分：

| 思想 | 本轮实际数据结构／计算 |
|---|---|
| 失败不是孤立点 | 将同一交互中邻近、表现相近的已观察失败组织成区域 |
| 区域与通过区之间的关系重要 | 保存同一历史系统内的失败—通过对照边；不把不同系统的相反结果误连为同一系统边界 |
| 区域可能保留、消失、变形、出现新的部分 | 目标模型允许局部非线性形状变化；通过目标执行更新，而不是强制一维平移 |
| 知识可继续使用 | 为每个区域保留各版本／agent 的实际证据；会话结束后提交，下一会话可以加载 |

**候选方法贡献是“交互条件化的失效模式记忆及其跨系统更新”，不是声称 RBF、逻辑回归、贝叶斯更新或 Chen 的方法是本研究首次发明。**

旧项目材料关于保存 failure signature、使用实际失败而非只优化预测误差、避免重型在线训练的方向继续保留。[P1][P2] 其早期低秩、K=1/2/4 和层层停止门槛不作为本轮要求。

---

## 3. 问题与结果定义

### 3.1 最小统一定义

令 `E(u, x)` 表示系统 `u` 在场景 `x` 上发生 ego 碰撞。首轮主事件仅使用这个二值量。

回归事件：

\[
R(u_{new},x)=\mathbf 1[\operatorname{completed}(u_{ref},x)\land \neg E(u_{ref},x)\land E(u_{new},x)].
\]

跨 agent 事件：

\[
A(u_*,x)=\mathbf 1[E(u_*,x)].
\]

背景车辆自行碰撞导致提前终止、运行异常等记为 `inconclusive`，不是目标通过，也不是目标失败；它仍消耗一次测试调用。不得因为系统触发 ego 碰撞而把该次执行当作“语义未完成”删除。

### 3.2 历史库与候选池分开

`history` 可包含所有已允许使用的历史通过／失败记录；`candidate_pool` 规定当前能测试哪些场景。

- 回归候选只要求**最近参考版本通过**，不要求所有更早版本都通过。
- 跨 agent 候选是同一个合法场景池，不按目标结果过滤。
- 更早版本失败、最近版本修复通过的场景可以进入回归候选。
- 每条历史记录保留系统／版本和场景身份；不同系统在同一场景上的相反标签都保留。
- 同一目标的未查询结果只能留在 evaluator 的实测银行，不能用于模式构建、特征选择或参数拟合。

### 3.3 “失效模式”不冒充根因

首轮模式是**可观察交互＋事件表现＋局部输入区域**，例如“切出交互中与静止目标发生纵向碰撞的一片场景”。它不是已经证明的软件根因。

`merge_blind06`、源代码行号、`fault_active_steps` 等信息只用于解释和验收，不作为选择器输入，不用于给目标提前指定所属模式。

---

## 4. 工程架构：保留场景，解除 SUT 硬绑定

### 4.1 路径与职责

优先在现有 `method_chains/failure_memory_regression/` 内增加 v2 文件，不再创建第三个平行研究目录。

```text
sut_algorithms/highway_env/
    registry.py                    # 保留现有工厂，补充 BuildSpec -> adapter 路由
    fbrt_adapters.py                # 新增：原生／外部策略／旧 Profiled 控制器接口
    regression_builds.py            # 新增：两种明确的集成／功能修改

highway_env_benchmark/envs/
    fbrt_env.py                     # 保留 legacy 入口；增加 build_spec 方式
    fbrt_scenarios.py               # 保留四模板；新增换道后车模板及有单位字段
    fbrt_scripted_vehicle.py        # 保留既有状态机

method_chains/failure_memory_regression/
    boundary_memory.py             # 原 v1 冻结，可用作已有对照
    boundary_shift.py              # 原 v1 冻结，不继续调参
    selectors.py                   # 原 v1 保留；修复后的对照另设名称
    schema_v2.py                   # BuildSpec / EpisodeRecord / PatternCard / Session
    archive_v2.py                  # 导入、可见性、去重、增量提交
    pattern_memory.py              # 失效区域、通过对照、模式中心
    bayes_model.py                 # RBF 特征与层次化贝叶斯逻辑模型
    selector_v2.py                 # FBRT-Memory、唯一 NoMemory 消融
    experiment_v2.py               # cache + compact 两个实验
    report_v2.py                   # 表格、曲线、模式生命周期、回放
    tests/test_memory_v2_*.py

results/method_chains/failure_memory_regression/memory_v2/
    protocol.json
    cache_replay/
    compact_bank/
    compact_regression/
    compact_cross_agent/
    acceptance.json
    report.md
```

旧 `standard_aligned/core/` 内容保持只读。新增代码不得把旧结果重新标记为新实验。

### 4.2 统一被测系统描述

```python
@dataclass(frozen=True)
class BuildSpec:
    build_id: str
    family: str
    parent_build_id: str | None
    adapter_kind: str      # legacy_profile / native_vehicle / external_meta_policy
    policy_name: str
    control_hz: float
    profile: dict | None
    checkpoint_sha256: str | None
    mutation: dict | None
```

`build_id` 用于记录和缓存，不作为目标模型的特征。`family` 用于对历史源权重去重；目标不需要通过名称被猜出内部故障。明确的父版本关系可以用于回归先验，这是任务给定的信息，不是黑盒泄漏。

### 4.3 每个物理 tick 的控制顺序

仿真底层固定 20 Hz。原生规则控制器每个 tick 更新。PPO 的决策周期与底层跟踪周期分开记录。

```text
1. 根据当前状态收集观测快照（包括供延迟版本使用的快照）。
2. 到外部策略决策时刻时，调用模型一次并设置新的目标速度／目标车道。
3. 原生车辆或 MDPVehicle 执行当前低层跟踪控制。
4. 每个脚本背景车执行一次自己的动作。
5. road.step(0.05)。
6. 更新轨迹、碰撞、事件阶段和时间戳。
```

不能重复调用 `road.act()` 导致 ego 第二次覆盖动作。`FASTER/SLOWER/LANE_LEFT` 等高层动作只在决策时应用一次，不能在每个物理 tick 重复增加速度档位。`MDPVehicle.act(None)` 的低层跟踪与再次应用相同高层动作是不同操作。

保留旧路径的执行行为以复用旧缓存；新 `build_spec` 路径使用单独的 `execution_contract_version`。不得在修复新接口后静默沿用行为已经不同的旧 episode。

### 4.4 PPO 接入的具体约定

现有工厂已支持 `ppo_ece`，权重路径为 `assets_root/ppo_ece/vd_1_5_trial_1.zip`；现有 wrapper 使用 CPU 与确定性预测。[C7][C8]

- 优先使用已有已核验检查点及观测配置，不重新找模型、不重训。
- 使用现有 `ExternalCutInEnv` 的 5×6 Kinematics 特征顺序、范围、相对坐标、排序与 padding 规则。
- 新统一执行器默认 PPO **5 Hz 决策、20 Hz 低层执行**，参考与修改版本完全一致；这是本轮集成测试契约，不称为原训练频率的逐项复现。
- 若现有资格工件明确冻结了不同、且已正常运行的频率，使用该值并写入 `protocol.json`，不自动修正原生训练设置。
- 不直接安装外部仓库整份 requirements 覆盖当前环境。
- 模型缺失时调用已有 fetch 流程一次。仍无法加载时写明 `PPO_UNAVAILABLE`，继续完成缓存实验和规则系统实验；不能伪装成已经完成异构学习型验证，也不转为从零训练。

---

## 5. 被测系统与版本：只增加必要的异质性

### 5.1 旧配置的定位

保留已有 `idm_ref`、`merge_blind06`、`merge_brake2`、`slow_front_brake2`，用于缓存实验 A 与后续历史知识。

不要再次加强这三个故障以追求更好的方法差距。`slow_front_brake2` 作为广泛退化对照单列；不得用其大量碰撞制造整体获胜结论。

### 5.2 新银行仅使用四个配置

| build_id | 实际实现 | 用途 |
|---|---|---|
| `mobil_ref_v2` | highway-env 原生 IDMVehicle，正常 MOBIL 规则 | 不同决策机制的目标 agent，也是换道修改的父版本 |
| `mobil_rear_guard_off_v2` | 同一 IDM+MOBIL，仅移除目标车道后车过度制动的拒绝条件 | 横向交互回归，不改背景车或车辆物理能力 |
| `ppo_ref_v2` | 已有 PPO-ECE＋正常观测与动作适配器 | 学习型目标 agent，也是观测修改的父版本 |
| `ppo_obs_age020_v2` | 同一权重、同一控制频率，输入整体观测滞后 0.20 s | 学习型驾驶系统的集成回归，不称权重退化 |

### 5.3 `mobil_rear_guard_off_v2` 的实现边界

从本地安装版本的 `IDMVehicle.mobil()` 提取目标车道后车安全检查，固定该代码版本。只旁路类似下述拒绝条件：

```python
if predicted_rear_braking < -allowed_imposed_braking:
    return False
```

保留其它换道收益、路线、频率和冲突逻辑。不要在场景外强制 ego 换道，不要读取新场景的编号触发故障，也不要关闭碰撞检测。

保存 `mutation_diff`，记录该条件本来会拒绝、但修改版没有拒绝的次数作为调试字段。它不是给选择器的输入。这个版本是明确的受控功能修改，不要求找到真实量产事故。

### 5.4 `ppo_obs_age020_v2` 的实现边界

每个物理 tick 收集 `observation_type.observe()` 的完整拷贝。模型决策时取最近一个满足 `timestamp <= now - 0.20` 的快照；起始不足 0.20 s 时使用初始观测。其余接口与参考完全一致。

版本差异发生在观测缓存，不能偷偷改变 normalization、padding、动作编号、仿真速度或背景脚本。reset 清空缓存。

---

## 6. 原场景实施基线：历史保留，新银行按 V2 配方选择

> 本节保留原设计以解释旧接口；尚未执行的新银行以第 23—29 节为准。已经生成的物理银行不因此自动重建。

### 6.1 保留的四类

| template_id | 保留的功能 | 本轮处理 |
|---|---|---|
| `fbrt_cutin` | 相邻前车切入 | 不改变已运行的脚本；按实际进入方向记录交互 |
| `fbrt_cutout_static` | 前车切出后自车面对静止车辆 | 保留三车结构；不声称有真实视觉遮挡模型 |
| `fbrt_lead_emergency_brake` | 单车道前车刹停 | 保留现有扩大后的间距范围；不再按目标结果调范围 |
| `fbrt_stop_hold_go` | 前车刹停、保持、重新起步 | 保留状态机；没有旧失败也必须可搜索 |

原版建议每类 16 个 Sobol 样本，采用现有 active parameters 和 bounds；V2 新银行使用第 27—29 节明确的配方、范围与依赖约束，每类仍为 16 个样本。旧场景背景行为、道路和监测尽量保持一致；不同 agent 可以使用不同避险动作，这是被测行为，不应通过删车道消除。

规范资料除原 `standard_mapping.md` 外，补充第 23、31 节已核对资料。具体数值来源分级记录；无需追加标准认证或标准审计实验。

### 6.2 唯一新增：`fbrt_lane_change_rear`

**交通结构：** 两车道；ego 位于车道 0，前方慢车使其具有换道动机，车道 1 后方车辆保持直行。自车是否换道由 SUT 决定，不由场景脚本决定。

| 项目 | 本轮默认研究值 |
|---|---:|
| ego 初速 | 25 m/s |
| ego 前方慢车速度 | 18 m/s |
| ego 与慢车初始保险杠净距 | 25 m |
| 目标车道后车初始保险杠净距 | **8～45 m，active parameter 1** |
| 后车相对 ego 的接近速度 | **2～12 m/s，active parameter 2** |
| 后车行为 | 持续直行，保持给定速度 |
| 持续时间 | 10 s |
| 道路 | 2 车道，复用既有宽度和物理模型 |
| 主事件 | ego 与任意车发生碰撞 |

后车位置由保险杠净距和车辆长度计算，不能把正净距误当中心距离。所有 SUT 使用同一个背景场景。

如果正常与修改 MOBIL 都未换道，先检查原生 MOBIL 是否真正接入、有无目标车道、前车查询是否正确。只允许在 smoke 阶段、尚未生成完整新银行前，进行一次不读取方法比较结果的场景修正，例如固定慢车间距由 25 m 改为 20 m；该修正及 smoke 调用写入日志，随后四系统统一使用最终配置。完整新银行开始后不再改场景范围。若本池仍未触发换道或没有回归，报告为当前任务未观察到对应行为，不继续调到必撞。

### 6.3 交互描述与功能名称分离

场景名称用于道路与脚本隔离；模式描述使用可观察属性：

```text
lateral_motion: entering / leaving / ego_changing / none
target_motion: moving / braking / stopped / restarting
interaction_role: lead / revealed_static / adjacent_rear
lane_relation: same_lane / crossing_lane / target_lane_rear
```

这些属性来自场景脚本或已执行轨迹，而不是 `merge_*` 故障名。

跨模板物理坐标暂不直接混合：不同模板的 2D active parameters 没有同一含义。语义相近的模式可以显示关联，但首轮预测模型按模板分开；本轮主张是**同一功能语法内的跨版本／跨 agent 复用**，不是任意跨场景零样本迁移。

---

## 7. 历史记录与失效模式数据结构

### 7.1 EpisodeRecord

```python
@dataclass
class EpisodeRecord:
    execution_id: str
    build_id: str
    build_fingerprint: str
    scenario_fingerprint: str
    template_id: str
    scenario: dict
    simulator_seed: int
    execution_contract_version: str
    completed: bool
    ego_collision: bool
    inconclusive: bool
    collision_partner_role: str | None
    collision_time_s: float | None
    min_ttc: float | None
    min_clearance: float | None
    public_signature: dict
    visibility: str    # historical / revealed / evaluator_only
    episode_cost: int  # 实际执行为 1；访问已有缓存的新增物理成本为 0
    trajectory_path: str | None
```

旧 CSV 缺少字段时用 `null` 或 `unknown`，不能补造轨迹。已有 `collision_partner`、事件时刻等可直接导入；不足以判断 entering/leaving 时只用场景脚本类别，不推断根因。主算法须能在旧数据缺乏完整 signature 时退化为几何模式，而不是强制重跑全部历史。

### 7.2 PatternCard

```python
@dataclass
class PatternCard:
    pattern_id: str
    template_id: str
    semantic_key: tuple
    center: list[float]             # 标准化的可执行输入坐标
    radius: float
    failure_record_ids: list[str]
    pass_contrast_record_ids: list[str]
    boundary_edges: list[tuple[str, str]]
    occurrence_by_build: dict       # 各系统上的已观察通过/失败/未观察
    created_in_session: str
    parent_pattern_ids: list[str]
    evidence_status: str            # observed_region / singleton / contrast_available
```

**同一场景在多个系统上重复执行，不是新的独立几何点。**中心与空间采样去重，系统结果分别保留。

### 7.3 PatternCard 的用途必须是计算性的

模式卡不能只写进报告。其中心、范围和对照证据必须实际进入 RBF 字典、历史拟合以及目标候选预测；查询日志记录当前场景受哪些模式影响。否则不算完成本轮方法。

### 7.4 状态按系统记录，不做全局“已修复”删除

允许的状态：

- `confirmed_present`：该系统至少一条已执行失败证据。
- `local_pass_evidence`：已执行的邻近代表点通过，只是局部通过证据。
- `regression_witness`：同一场景最近父版本通过，当前系统失败。
- `new_region_observed`：失败落在当前模式支持之外，建立新卡。
- `unknown`：没有当前系统执行证据。

少数通过不能证明整片区域已修复；不能因为一个 agent 通过，就把其它 agent 的历史失败从库中抹掉。

---

## 8. 模式构建：从点对扩展到区域和系统响应

### 8.1 坐标与距离

每个模板按已固定的物理 bounds 将 active parameters 线性缩放到 `[0,1]`。所有方法共用这套坐标。不同模板不计算直接欧氏距离；同模板内的对照才构成边界证据。

**禁止评分使用 `scenario_id`、`initial/probe/core_boundary` 等生成阶段标记。**这些只用于追溯。

### 8.2 每个历史系统内部建失效邻接图

对一个系统、一个模板内的实际失败点：

1. 按可获得的粗粒度 `semantic_key` 分组；未知字段允许独立的 `unknown` 分组，不额外补数据。
2. 建立 mutual 3-nearest-neighbor 图，连接距离不超过 `0.25` 的失败点。
3. 如果连线内部附近已有该系统的通过点，距离小于 `0.05`，不使用这条边合并区域，避免跨过已知通过区。
4. 每个连通分量形成一个候选区域；单个失败点保留为 singleton。
5. 对区域内最多四个最分散的失败代表，分别找同一系统的最近通过点，记录 `boundary_edges`。不把其它系统的通过标签冒充该系统边界。

这是一种有限数据的区域近似，不是严格拓扑恢复。只使用已有实际标签；不把连线或凸包内部全部填为失败。

### 8.3 跨历史系统对齐

模板相同、语义标签相容、中心距离不超过 `0.15` 的候选区域可以建立 `related_pattern` 关联。是否合并卡片只影响存储去重，**不合并各系统的标签**。

首轮允许保留两个相关卡而不强制合并。关键是一个模式附近能查询到哪些历史系统失败、哪些通过，而不是从聚类中自动得到根因。

### 8.4 模式中心压缩

全部证据存储，但每模板最多选六个历史中心作为主模型特征：先保留不同可观察事件的代表，再用 farthest-point 从其余中心补齐。不能仅取距离最短的六个点对。

另选四个**不依赖标签**的覆盖中心：从当前公共场景池标准化坐标做固定 RNG 初始化的 farthest-point。它们提供无历史失败区的表达能力。

原始卡数量可以超过六个；六个只是小模型的表示预算。每个选中的特征中心保留来源卡 ID。

---

## 9. 主模型：模式字典上的层次化贝叶斯逻辑回归

### 9.1 为什么采用这个模型

它满足本轮四个必要条件：小数据、CPU 可运行；源系统有各自响应；目标可以改变局部区域形状；历史没失败的位置也能在目标反馈后提高失败概率。

不再要求每个区域只能沿失败—通过连线平移，也不采用 `sum(weight * historical_binary_label)` 这种封闭历史标签混合。

### 9.2 模式特征

在模板 `m` 中，标准化输入为 `z(x)`，构造：

\[
\phi_m(x)=\left[1,\ z_1,z_2,\ k(x,c_1),\ldots,k(x,c_J)\right]^\top,
\]

\[
k(x,c_j)=\exp\left(-\frac{\|z(x)-c_j\|^2}{2h_j^2}\right).
\]

- 历史模式中心最多 6 个，`h_j=clip(radius_j, 0.10, 0.35)`。
- 覆盖中心固定 4 个，`h=0.30`。
- 初始维度最多 `1+2+6+4=13`。
- 每个目标会话每模板最多追加 4 个新失败中心，最终最多 17 维。
- 未测试目标的轨迹、TTC、故障状态不可进入 `phi`；这里全部是场景输入和历史字典。

RBF 让等概率线能弯曲，并可组合多个局部区域；这比一维平移表达范围更宽，但不保证任意高维失效函数都能恢复。

### 9.3 对每个历史系统独立拟合

\[
p_u(x)=\sigma\left(\theta_{u,m}^{\top}\phi_m(x)\right),\qquad
\sigma(a)=\frac1{1+e^{-a}}.
\]

历史系统在相同场景上的相反结果不平均成一个真标签。每个 `u,m` 分别用其实际已知二值碰撞记录拟合带高斯正则的逻辑模型：

\[
\mathcal L_u(\theta)=
\frac{\|\theta\|^2}{2\cdot 2^2}
-\sum_{i\in D_u}\big[y_i\log p_i+(1-y_i)\log(1-p_i)\big].
\]

输出 MAP 参数 `theta_u` 和 Hessian 逆的对角近似 `V_u`。允许只有全通过或全失败的历史系统：高斯正则避免分离问题，不补造相反标签。

各系统只使用实际存在的记录；不要求新场景上补齐所有旧系统的完整响应矩阵。

同一源系统没有本模板有效记录时，不为它拟合一条假“全安全”曲线，也不把它纳入该模板的先验均值。覆盖中心由当前公共候选输入构造，因此同一任务内所有源系统共享完全相同的特征顺序。

### 9.4 历史先验构造

**回归任务：**

父版本有足够记录时，以父版本拟合系数作为目标先验均值；其它历史系统提供模式中心与合理变化尺度。

\[
\mu_0=\hat\theta_{ref,m},\qquad
\Sigma_0=\operatorname{diag}(\operatorname{diag}(V_{ref,m})+d).
\]

**跨 agent 任务：**

\[
\mu_0=\sum_u\omega_u\hat\theta_{u,m},
\]

\[
\Sigma_0=\operatorname{diag}\left[
\sum_u\omega_u\left(\operatorname{diag}(V_{u,m})+
(\hat\theta_{u,m}-\mu_0)^2\right)+d\right].
\]

`omega` 对历史家族先等权、家族内部版本再等权。记录很多次的同一版本不能被重复视作很多个独立 source。

本轮 `d` 默认：截距为 `2.0^2`，线性坐标为 `1.0^2`，RBF 系数为 `1.5^2`。这是防止少量历史导致先验过硬的工程配置，不是经证明的最佳参数。

无历史或无父版本局部记录时，使用 `mu0=0`、`Sigma0=4I`；仍有覆盖中心，能够在目标结果到来后形成区域。

### 9.5 目标后验更新

对当前会话**所有已查询且结果有效的目标记录**，求解：

\[
\mathcal L_*(\theta)=
\frac12(\theta-\mu_0)^\top\Sigma_0^{-1}(\theta-\mu_0)
-\sum_{i\in D_*^{obs}}\big[y_i\log\sigma(\theta^\top\phi_i)
+(1-y_i)\log(1-\sigma(\theta^\top\phi_i))\big].
\]

梯度与 Hessian：

\[
g=\Sigma_0^{-1}(\theta-\mu_0)+\Phi^\top(p-y),
\]

\[
H=\Sigma_0^{-1}+\Phi^\top\operatorname{diag}(p_i(1-p_i))\Phi.
\]

用 Newton 或 `scipy.optimize.minimize`，warm-start 上一次 MAP；每次最多 25 次迭代。后验取 Laplace 近似：

\[
q(\theta)\approx\mathcal N(\hat\theta,H^{-1}).
\]

注意：以**原始会话先验＋全部已观察目标数据**重新拟合；不能把上一轮后验当新先验后又重放同一批数据，造成重复计数。数值实现使用 `expit`、`logaddexp`、Cholesky 与 `1e-6` jitter。

这是参数不确定性的近似，不据此宣称概率已严格校准或保证不漏测。

### 9.6 预测与选择保持简单

从当前后验固定 RNG 抽取 32 组参数，计算候选平均失败概率。每个目标调用后更新所有本模板相关系数，不局限于一个最近 patch。

- 正常查询：选择当前允许候选中后验平均碰撞概率最高的场景。
- 第 10、20 次查询：执行一次公共 ART-maximin 探索，防止旧记忆未覆盖区域完全不被检查。
- 评分相同或全部平坦时：使用共同固定 RNG 破平局；不按场景编号。
- 所有已查询有效结果，包括 ART 得到的结果，都进入后验更新。
- 不再增加历史裕度＋几何距离＋信息增益＋覆盖惩罚的多项评分。模型负责利用信息，选择器只做风险优先和少量全局探索。

固定 B=20 的所有测试均计入预算，没有额外免费 support。

### 9.7 新失败区域的形成

当观察到失败，且其坐标距现有历史模式中心均超过 `0.20`，或该模板根本没有历史失败：

1. 在当前会话创建 `singleton` PatternCard，引用本次真实执行。
2. 若有本目标此前的附近通过记录，立即建立对照边；若无，则保留 singleton，不补造通过点。
3. 若还未达到每模板四个新增中心上限，追加以此失败坐标为中心、带宽 `0.15` 的 RBF 特征。
4. 新系数均值为 0、方差为 4，与原系数初始协方差为 0；保留原先验，并用本会话所有已观察数据重拟合。
5. 下一次查询立即使用新特征，不等到报告阶段才入库。

若达到特征上限，仍保存新模式证据，只不再增加模型维度。旧模式不能被新 agent 的通过记录全局删除。

若几何位置相近但出现不同碰撞对象或不同交互阶段，保留一个新的事件表现卡并关联原区域；不强行称为新的几何区域。相同位置不重复增加等价 RBF 列，避免无意义的共线特征。

---

## 10. 查询、记忆更新与跨 agent 持久化

### 10.1 一次测试会话

```python
memory = load_snapshot(history_snapshot)
view = make_allowed_history(memory, task)
pool = candidate_pool_for_task(task)          # 不读取目标标签
model = fit_source_prior(view, task.reference_id)

for query in range(1, min(20, len(pool)) + 1):
    x = choose_with_memory(model, pool, query, rng)
    y = oracle.execute_or_reveal(x)           # 一次逻辑计费
    ledger.append(x, y, model.snapshot_id)
    if not y.inconclusive:
        session.observe(x, y)
        maybe_add_pattern(session, x, y)
        model.refit_from_session_observations(session)
    pool.remove(x)

commit_observed_records(memory, session)       # 只提交已查询的记录
save_memory_and_session(memory, session)
```

### 10.2 回归和跨 agent 的区别只发生在任务层

- `regression`：父版本结果用于定义候选与先验；发现父通过／新失败的见证。
- `cross_agent`：没有父通过过滤；源模式形成初始先验；新结果提交给下一个 agent。
- 同一实现不得为了某个目标 `build_id` 写入不同的优先功能或不同故障条件。

### 10.3 记忆提交必须可证明

提交产物包含新增的 execution IDs、旧／新 snapshot hash、变更的模式卡、以及哪些记录进入了下一会话的 prior。

“调用 `save()` 成功”不够。必须有一个自动化测试证明：重载 snapshot 后，无需手工传参，下一个会话可读到前一会话的真实已查询失败，并在相关候选上得到不同的预测或排序。

对照方法各自维护自己的会话分支。不能把 FBRT 已查询出的有用失败免费塞给对照，也不能在连续测试中把第一 agent 的完整 evaluator 银行提交为历史。

---

## 11. 必要实验 A：现有银行的零新增仿真重放

### 11.1 数据

导入：

```text
results/method_chains/failure_memory_regression/standard_aligned/core/
    reference_archive.csv
    candidate_pool.csv
    target_response_bank.csv
    summary_by_template.csv
```

旧资料共有 480 个参考和 960 个目标记录；若本地与核对版本不同，以实际完整记录为准并记录差异。不为凑齐这些数字重新仿真。

### 11.2 主重放任务

保留原先 3 种子 × 3 修改版本的任务，每任务父版本为旧 `idm_ref`，候选不变，B=20。每项任务的历史主库为**父版本完整记录**；目标未查询记录隐藏。这样新方法与现有方案可直接对比。

这是实现复核与开发效果，不假装是未参与讨论的全新数据。旧任务的选择难度饱和，不把“必须超过 9/9”设为目标。

### 11.3 一个额外的零物理成本持续记忆演示

在第一个种子上固定顺序：

```text
merge_blind06 → merge_brake2 → slow_front_brake2
```

每个系统最多查询 20 个场景；上一个会话只提交实际查询结果。该序列是受控版本的工程演示，不称为真实生产发布顺序。回归候选始终依据已给定父版本，历史模式可以逐步增加。

仅输出一条“模式被发现—提交—下一系统检索”的完整实例；不扫描所有六种排列，不据此单独宣称性能提升。

---

## 12. 必要实验 B：一个新小银行同时完成两项用途

### 12.1 固定银行

**V2：模板由第27节能力匹配配方确定；采样按第28节，主体上限不变。已生成银行不自动重建。**

- 一个新采样种子：`4179901`。
- 五个模板，各 16 个场景；共 80 个。
- 四个系统配置：第 5.2 节的两个基准和两个修改。
- 全部四个系统执行同样的 80 个场景，共 320 次。
- 背景脚本和道路契约一致；参考／修改之间的动作不同是正常被测响应。

不要求每个模板都有失败，不根据 FBRT 是否获胜挑选场景。不补多种子，不因任务中没碰撞而加强修改。实际缺陷是否产生回归由银行结果决定。

### 12.2 B-R：两项配对回归任务

```text
mobil_ref_v2 → mobil_rear_guard_off_v2
ppo_ref_v2   → ppo_obs_age020_v2
```

每项候选是父版本完整通过的场景，B=20 或候选不足时取其全部数量。允许使用旧银行作为更早历史，加上本次父版本完整记录；这部分源执行计入真实物理成本。

若父版本没有通过样本或目标没有新碰撞，记录为 `no_eligible_cases` / `no_observed_regression_in_pool`；不伪造指标，也不阻断另一个任务与报告交付。

### 12.3 B-A：两项独立驾驶策略测试

初始历史为此前完整已允许使用的旧银行，分别考察 `mobil_ref_v2` 与 `ppo_ref_v2`。

主顺序固定：

```text
旧历史 H → mobil_ref_v2（只揭示 20 次）
         → 提交所揭示记录
         → ppo_ref_v2（只揭示 20 次）
```

将完整 `mobil_ref_v2` 银行用于 evaluator 不代表选择器可以看到全部 80 次结果。B-R 与 B-A 使用不同可见性视图；不能因为前一个实验已读取父版本，就让 B-A 免费得到它。

最多附带一次逆序缓存重放，作为顺序说明，不构成新的仿真实验，也不据此调参。

### 12.4 为什么这两个用途不可省略

B-R 是“同一功能系统版本改变”的例子；B-A 是“分别测试规则型和学习型 agent 并持续复用记忆”的例子。它们共用同一个 320 次银行，不需要再安排两套采样或仿真。

这比继续增加纯 IDM 参数版本更直接回应当前研究目标。

---

## 13. 对照方法与唯一必要消融

主表保留以下五行；旧 FBRT v1 仅在实验 A 附带引用其既有结果，不占新实验行数。

| 方法 | 定义 | 回答的问题 |
|---|---|---|
| `Random` | 固定 RNG 无放回随机；离线重复 10 次取均值 | 随机重放水平 |
| `HistoryRank-UCB-v2` | 按模板 UCB 分配；模板内用历史风险顺序 | 简单历史＋在线分配是否已足够 |
| `FailureDistance-v2` | 按同模板到已允许历史失败的最近距离；目标已观察失败可增量入库 | 只记住失败点是否已足够 |
| `FBRT-NoMemory` | 同样 Bayesian logistic 更新和查询日程，但无历史标签先验、无历史失败中心；只用固定覆盖中心 | 历史模式整体是否贡献了实用价值 |
| `FBRT-Memory` | 本文完整模式库与层次化先验、目标更新、新模式形成 | 主方法 |

### 13.1 对照的精确实现

`HistoryRank-UCB-v2`：

- 所有可用历史系统在候选 `x` 有结果时，取各系统内部 TTC/clearance 风险百分位，再按家族等权求均值。实际碰撞排在非碰撞之前。
- 候选没有精确历史记录时，使用同模板最多 5 个历史近邻做距离加权插值；不能回读目标银行。
- UCB reward 为本次真实目标碰撞；回归候选已保证父通过。
- 未试过的模板依照固定 RNG 次序试一次，再用 `mean_reward + sqrt(2*log(t)/n)`；不按模板字母顺序。
- 每个方法的已查询结果只进入自己的记忆分支。

`FailureDistance-v2`：

- 精确历史失败用例距离可为零，这是合法的历史失败重放优势。
- 无历史失败的模板不永久得零分：每第 10 次使用与主方法相同的 ART 探索。
- 当前目标发现失败后，该点也加入本方法自己的失败集合。

`FBRT-NoMemory` 保留相同父版本通过候选过滤，因为这是任务定义；去掉的是额外可用的历史失效模式与预测先验。因此它测量“在同一回归任务上，额外失效记忆是否有用”，不是声称完全不使用任何旧信息。

不再增加 no-geometry、no-signature、no-update、no-prior、no-ART、多个 bandwidth 等消融。没有新增物理预算的旧 v1 结果可以作为背景，不能为了完整性要求逐个重训重测。

---

## 14. 指标：优先关注实际测试收益，而非拟合分数

### 14.1 主结果

对每个任务画累计实际失败／回归数 `C(b)`，报告 `@1/@5/@10/@20`。

- `Detected@b`：是否已发现至少一个实际失败／回归。
- `FirstFailureRank`：首次发现位置；未发现记 `>B`，不要用 0 参与平均。
- `Count@b`：实际发现数。
- `Recall@b = Count@b / F_pool`：完整 evaluator 银行中存在失败时计算；`F_pool=0` 记 NA。
- `EarlyRecallAUC@20 = mean_{b=1..20}(Recall@b)`：按任务等权汇总，不让大面积退化任务支配全部结果。

若候选不足 20，使用实际预算，并单列；不得补重复用例凑满。 AUC 此时按实际轨迹长度计算并标注 `effective_horizon`；不同 horizon 不直接并入同一个 @20 平均。

**不存在失败的任务也要报告。**它们不能进入以 `F_pool` 为分母的宏平均 recall，但不能从总任务数或失败存在性表里消失。

### 14.2 只保留两个辅助输出

1. 模式生命周期：多少历史模式得到当前失败证据、多少仅有局部通过证据、是否出现新区域；这些不是独立软件缺陷数量。
2. 计算成本：真实物理 episode、缓存命中、逻辑查询、模型更新时间、总 wall-clock。

不增加一组 NLL/ECE/AUROC 专项图；不要求每类功能都出现碰撞；不把模式卡数量作为主获胜指标。

### 14.3 如何给出结论

先展示逐任务结果，再给宏平均；明确哪个方法在哪个任务有效、何处并列或退化。

有父通过／新失败的见证才能说发现回归。未测试区域不标“安全”。PPO 没有观察到失败时，不称 FBRT 已证明其安全，只说明本池中未观察到。

---

## 15. 验收目标：硬工程条件与效果观察分开

### 15.1 必须通过的工程验收

| ID | 验收项 | 最小可检查产物 |
|---|---|---|
| E1 | 同一 FBRT runner 能调用旧 Profiled-IDM、原生 IDM+MOBIL、PPO wrapper | 注册表＋动作变化测试；PPO 资源异常时单独记录，而不是冒充通过 |
| E2 | 外部动作实际影响 ego，且未被第二次控制覆盖 | 加速／减速／换道三类短轨迹测试；相同动作只施加一次 |
| E3 | 模式库使用真实失败与同系统通过对照 | 每个 PatternCard 的 execution IDs 可回查；未知字段不补造 |
| E4 | 模式卡实际影响预测或选例，不只是可视化 | 查询日志有 contributing pattern IDs 和对应 RBF 特征 |
| E5 | 无历史失败模板仍可预测、执行；新失败能新增区域 | 一个合成单元测试＋实际执行出现时的日志；不强求实际新银行必出现新岛 |
| E6 | 目标反馈更新所有已观察数据且无重复似然计数 | 数值单元测试；序贯保存载入与一次性拟合同一观测集结果一致 |
| E7 | 新记忆保存后能被下一 agent 加载使用 | old/new snapshot hash、提交 execution IDs、下一会话预测变化 |
| E8 | 相同预算、候选、可见性、有效事件定义 | 每项 ledger；结果缺失明确记 NA，不填标签 |
| E9 | 两个实验执行或明确记录资源阻塞，旧报告不覆盖 | 两份完整结果和总报告；不是单独 `gate_failed` |
| E10 | 新增物理调用不超过 400 | 统一成本 ledger；重复执行与调试也计入 |

工程 smoke 验收不包含“每个系统都在每个场景零碰撞”；需要能完成输入／动作／事件链，实际碰撞本身不是接口失败。

### 15.2 方法功能验收

以下可以用自动化单元测试与真实日志共同完成，不增加大规模仿真：

- 一个历史模式被检索，改变至少一个候选的预测或选择顺序。
- 同一候选在两个已执行系统上可保留相反结果，而不是平均掉系统差异。
- 目标通过证据降低局部预测，失败证据提高相关预测；不要求整个域单调变化。
- 新失败中心能够影响附近未测场景，且不会把它们标成“已验证失败”。
- 在第二个 agent 会话中，能追溯到第一个 agent 真正查询过的记录。

这些条件检查“是不是实现了失效记忆方法”，不是以漂亮结果为前提。

### 15.3 效果目标：明确但不作为无限返工门槛

期望看到：在至少一项**非饱和**回归或跨 agent 任务中，FBRT-Memory 比同预算的简单历史方法或 NoMemory 更早发现失败，或者发现更多实际失败；同时总表不隐藏其它任务的退化。

可接受的实际信号包括：前 10 次至少多发现 1 个失败，或者首次失败提前至少 1 次，并能给出“历史模式→选例→实际结果”的实例。该信号只是局部实用性证据，不等于普遍优势或显著性证明。

**最终交付分级：**

| 状态 | 含义 | Codex 应交付什么 |
|---|---|---|
| `IMPLEMENTED_WITH_GAIN` | 工程完整，至少存在上述局部真实收益 | 实现＋完整对照＋收益来自哪些信息 |
| `IMPLEMENTED_TIED` | 工程完整，与简单方法基本持平 | 实现＋完整对照＋持平原因；不自动扩实验 |
| `IMPLEMENTED_MIXED` | 部分任务改善、部分退化 | 逐任务结果与清楚适用边界 |
| `IMPLEMENTED_NO_GAIN` | 工程完整，本轮没有收益 | 如实报告；保存方法和数据，不再次堆评分项 |
| `PARTIAL_RESOURCE_BLOCK` | 外部模型确实不可用或必要接口未完成 | 已完成部分、具体阻塞、实际调用成本；不冒称全部完成 |

**交付不是必须获胜；获胜也不能替代工程验收。**不要求 p 值、置信区间、全部基线胜出、所有场景有故障或多种子确认后才交付。

---

## 16. 最小自动化测试清单

只增加与已定位问题直接相关的测试，不搭建庞大审核系统。

| 测试 | 内容 | 需要物理仿真？ |
|---|---|---|
| `test_id_order_independence` | 重命名 scenario_id 不改变非平局评分；平局由 RNG 决定 | 否 |
| `test_source_boundary_identity` | 不允许把 source A 的失败与 source B 的通过组成 A 的边界 | 否 |
| `test_all_safe_history` | 全通过历史仍能得到有限概率与可更新模型 | 否 |
| `test_no_history_failure_creation` | 一个目标失败产生新卡／中心，并影响附近候选 | 否，使用合成标签只验证数学实现 |
| `test_no_double_count` | 重复读取同一执行 ID 不重复强化后验 | 否 |
| `test_snapshot_reuse` | 保存载入后预测一致；下一会话可用已提交证据 | 否 |
| `test_visibility_views` | 回归实验中可用的父版本全集不能泄入 cross_agent 实验 | 否 |
| `test_action_owner` | 原生、外部 meta-policy 各有唯一 ego 动作所有者 | 是，包含在 24 次 smoke 内 |
| `test_mutation_locality` | MOBIL 仅改变后车拒绝分支；PPO 仅改变观测时间戳 | 主要静态／快照测试，必要 smoke 已计费 |
| `test_episode_accounting` | bank、smoke、replay、重试调用统一计数 | 否 |

合成单元测试结果不进入论文的自动驾驶失败数；其用途只是检查算法更新机制和代码契约。

---

## 17. 物理成本、复用与迭代限制

### 17.1 统一预算表

| 工作 | 新增 episode 上限 |
|---|---:|
| 旧银行导入、模式拟合、实验 A 所有选择器 | 0 |
| 4 系统 × 5 模板 × 16 场景新银行 | 320 |
| runner、动作、模型加载和少量正常驾驶 smoke | 24 |
| 最多四对父／子或跨 agent 场景回放 | 8 |
| 一次必要修复／复测预留 | 48 |
| **总上限** | **400** |

同一银行上的 Random 重复、全部方法、NoMemory、逆序演示、生成图表均属于离线重放，不重新调用模拟器。

### 17.2 缓存键

```text
hash(
  build implementation fingerprint,
  checkpoint hash,
  mutation config,
  full physical scenario parameters,
  simulator seed,
  physics/control timing,
  observation/action contract,
  event-oracle version
)
```

不能只用 `scenario_id`。也不要因为报告文字或画图代码改动，就使所有物理缓存失效。只对实际影响执行的组件建立 fingerprint。

### 17.3 一次修订规则

首次完整报告前允许修复明确 bug：动作覆盖、时间戳、标识符排序、后验重复计数等。统计所有额外物理调用。

首次看到比较结果后，最多做一次有记录的方案调整；优先修正模式关联／先验强度，不同时改 SUT、场景和指标。不做参数 sweep。调整后的同银行结果仍称开发结果，不自动追加新确认银行。

如果某模块没有帮助，保留最简可运行主线并交付；不因未超过一个强基线而继续扩大预算。

---

## 18. Codex 执行顺序与目标 CLI

以下是**需要 Codex 实现的入口**，不声称当前仓库已经支持这些命令。

### P0：固定基线并导入缓存

1. 记录当前 commit、环境关键版本和旧银行文件 hash。
2. 导入旧记录，保留缺失字段标记。
3. 建立 v2 schema 和 evaluator-only 的可见性隔离。
4. 不执行仿真，先生成 `archive_import_report.json`。

### P1：完成模式库与主模型

实现第 7～10 节，运行所有不需要物理执行的单元测试，随后完整运行实验 A。无论 A 是否优于旧方法，都继续接通已约定的异构 runner；A 不是停止门。

### P2：接通四配置与唯一新场景

保留旧 runner 行为；增加 BuildSpec 路径。执行 smoke，修复动作／观测问题，不以碰撞率作为接口验收条件。

### P3：执行一次 320 episode 新银行

一次生成新银行，保存执行轨迹摘要与事件；后续比较全部从银行重放。PPO 缺失分支写清实际未执行部分，不补其它算法凑数。

### P4：两种任务重放、一次必要消融、报告

回归 B-R 和跨 agent B-A 使用各自可见性视图。输出效果与工程验收，不再建议用户另补诊断实验。

目标命令：

```bash
# 不运行物理仿真
conda run -n metadrive python -m method_chains.failure_memory_regression.experiment_v2 --stage import --physical-limit 0
conda run -n metadrive python -m pytest method_chains/failure_memory_regression/tests -q -p no:cacheprovider
conda run -n metadrive python -m method_chains.failure_memory_regression.experiment_v2 --stage cache --physical-limit 0

# 新物理调用全部进入共享 ledger，总上限 400，不是每个命令单独 400
conda run -n metadrive python -m method_chains.failure_memory_regression.experiment_v2 --stage smoke --physical-limit 400
conda run -n metadrive python -m method_chains.failure_memory_regression.experiment_v2 --stage compact-bank --physical-limit 400

# 必须是零新增仿真重放
conda run -n metadrive python -m method_chains.failure_memory_regression.experiment_v2 --stage evaluate --physical-limit 0
conda run -n metadrive python -m method_chains.failure_memory_regression.report_v2 --reuse-bank
```

`--physical-limit 0` 表示禁止本次新增调用，已存在记录可用；累计上限另由 run-level ledger 控制。`all` 模式必须断点续跑，不因后续报告缺失重测整个银行。

---

## 19. 目标配置

V2 中的场景列表在第29节统一声明，本配置引用目录与能力匹配规则，不再重复写死五个模板。

```yaml
project: FBRT-Memory
baseline_commit: 6af8a502c90235063e593d2d36c8aede3aa027f2
simulator: highway-env
physics_hz: 20
physical_episode_cap_total: 400
policy_training_allowed: false

scope:
  preserve_legacy_results: true
  legacy_cache_replay: true
  compact_bank: true
  multiagent_means: independently_tested_driving_policies
  scenario_catalogue: configs/fbrt/scenario_catalogue_v2.yaml
  execute_all_catalogue_entries: false
  allow_cruise_and_brake_compositions: true

compact:
  scenario_seed: 4179901
  scenarios_per_template: 16
  # Resolve using section 27 before building the bank, not after method comparison.
  recipe_policy: existing_bank_else_static_capability
  default_when_ppo_cannot_stop: highway_policy_compatible_5
  legacy_recipe: legacy_core_5
  max_templates: 5
  builds:
    - mobil_ref_v2
    - mobil_rear_guard_off_v2
    - ppo_ref_v2
    - ppo_obs_age020_v2
  budget: 20
  checkpoints: [1, 5, 10, 20]

pattern_memory:
  mutual_knn: 3
  max_failure_edge_distance: 0.25
  pass_blocking_distance: 0.05
  cross_source_related_center_distance: 0.15
  historical_rbf_centers_per_template: 6
  coverage_rbf_centers_per_template: 4
  new_target_centers_per_template: 4
  new_region_distance: 0.20
  fallback_bandwidth: 0.30
  new_region_bandwidth: 0.15

model:
  type: hierarchical_bayesian_logistic_rbf
  source_prior_std: 2.0
  target_variance_floor_intercept: 4.0
  target_variance_floor_linear: 1.0
  target_variance_floor_rbf: 2.25
  laplace: true
  posterior_mc_samples: 32
  max_map_iterations: 25
  jitter: 0.000001

selection:
  rule: posterior_mean_failure_probability
  art_queries: [10, 20]
  tie_break: seeded_rng
  use_scenario_id_as_feature: false
  use_target_fault_name_as_feature: false

comparators:
  - Random
  - HistoryRank-UCB-v2
  - FailureDistance-v2
  - FBRT-NoMemory
  - FBRT-Memory

report:
  primary: early_failure_recall_auc_20
  include_task_level_results: true
  require_statistical_significance: false
  require_all_baselines_beaten: false
  max_post_result_revision_rounds: 1
  auto_launch_more_experiments: false
```

配置数值均为本轮实现起点，不作为预先验证的最优参数。除一次允许修订外保持固定。

---

## 20. 最终必须交付的文件

| 文件 | 必须包含的内容 |
|---|---|
| `protocol.json` | 实际系统、场景、版本、控制频率、预算、配置；所有资源降级 |
| `archive_import_report.json` | 旧数据完整性、缺失字段、去重及允许使用范围 |
| `patterns.jsonl` | 有真实 execution IDs 的模式卡，包含通过对照和分系统结果 |
| `history_snapshots/` | 连续测试前后的可重载模式库与 hash |
| `source_models.npz` | 源系统独立拟合结果、字典中心与先验参数 |
| `sessions/*/queries.csv` | 每次选例、理由、预算位置、模式来源、真实结果 |
| `sessions/*/updates.jsonl` | 模式新增／关联／局部通过证据与模型更新 |
| `compact_bank/episodes.jsonl` | 四系统的实测结果；不完整时注明缺失，不填充 |
| `summary_by_task.csv` | 各预算与各任务结果、无失败任务、实际查询数 |
| `compute_ledger.json` | 旧缓存、新物理、smoke、重试、回放、模型计算成本 |
| `acceptance.json` | 第 15 节 E1～E10 的逐项状态与总体交付等级 |
| `report.md` | 研究问题、方法、现有证据、实际结果、失败案例、完成边界 |
| `figures/` | 三类必要图即可，见下 |

只要求三类图：

1. 逐任务或分组的累计失败／回归发现曲线，不只画总体平均。
2. 两个有代表性的场景二维实测标签与模式支持图；不要把未测预测点画成实测标签。
3. 一张模式生命周期表／图，展示历史证据如何进入第二个系统；最多四对场景回放。

报告首页先回答：**接通了哪些实际系统？历史模式具体影响了哪些测试？新结果是否真的进入后续记忆？同预算效果如何？总共用了多少新仿真？** 不以长篇“创新性不足”代替这些交付。

---

## 21. 参考依据与源码入口

### 21.1 公开方法依据

[R1] Tsong Yueh Chen. **Failure-Based Testing**, ICST 2026 keynote. 方法思想：利用失效模式的信息；不是一套可以直接下载的完整自动驾驶测试器。  
https://conf.researchr.org/details/icst-2026/icst-2026-keynote/3/Failure-Based-Testing

[R2] Rubing Huang, Weifeng Sun, Tsong Yueh Chen, Sebastian Ng, Jinfu Chen. **Identification of Failure Regions for Programs with Numeric Inputs**. Search for Boundary、FSB、DSB 的原始研究。本方案借鉴失效区域识别思想，不声称逐行复现其完整实现。  
https://arxiv.org/abs/2007.15231

### 21.2 当前仓库事实依据

以下固定至本次核对的提交；Codex 执行时先记录本地 HEAD，不要求强制回退用户之后的有效修改。

[C1] 当前 FBRT 包说明：  
https://github.com/SafeDL/META_LEARNING/blob/6af8a502c90235063e593d2d36c8aede3aa027f2/method_chains/failure_memory_regression/README.md

[C2] 实测结果报告：  
https://github.com/SafeDL/META_LEARNING/blob/6af8a502c90235063e593d2d36c8aede3aa027f2/results/method_chains/failure_memory_regression/standard_aligned/core/report.md

[C3] 现有几何配对：  
https://github.com/SafeDL/META_LEARNING/blob/6af8a502c90235063e593d2d36c8aede3aa027f2/method_chains/failure_memory_regression/boundary_memory.py

[C4] 现有选择器：  
https://github.com/SafeDL/META_LEARNING/blob/6af8a502c90235063e593d2d36c8aede3aa027f2/method_chains/failure_memory_regression/selectors.py

[C5] FBRT 执行器：  
https://github.com/SafeDL/META_LEARNING/blob/6af8a502c90235063e593d2d36c8aede3aa027f2/highway_env_benchmark/envs/fbrt_env.py

[C6] 场景参数与脚本：  
https://github.com/SafeDL/META_LEARNING/blob/6af8a502c90235063e593d2d36c8aede3aa027f2/highway_env_benchmark/envs/fbrt_scenarios.py  
https://github.com/SafeDL/META_LEARNING/blob/6af8a502c90235063e593d2d36c8aede3aa027f2/highway_env_benchmark/envs/fbrt_scripted_vehicle.py

[C7] 已有 SUT 工厂：  
https://github.com/SafeDL/META_LEARNING/blob/6af8a502c90235063e593d2d36c8aede3aa027f2/sut_algorithms/highway_env/registry.py

[C8] PPO wrapper 与原生 IDM+MOBIL wrapper：  
https://github.com/SafeDL/META_LEARNING/blob/6af8a502c90235063e593d2d36c8aede3aa027f2/sut_algorithms/highway_env/ppo_ece.py  
https://github.com/SafeDL/META_LEARNING/blob/6af8a502c90235063e593d2d36c8aede3aa027f2/sut_algorithms/highway_env/idm_mobil.py

[C9] 当前局部修改实现与分模板结果：  
https://github.com/SafeDL/META_LEARNING/blob/6af8a502c90235063e593d2d36c8aede3aa027f2/method_chains/core_mine/local_fault_idm.py  
https://github.com/SafeDL/META_LEARNING/blob/6af8a502c90235063e593d2d36c8aede3aa027f2/results/method_chains/failure_memory_regression/standard_aligned/core/summary_by_template.csv

[C10] 当前离线重放与物理实验入口：  
https://github.com/SafeDL/META_LEARNING/blob/6af8a502c90235063e593d2d36c8aede3aa027f2/method_chains/failure_memory_regression/experiment.py

[C11] 已接通外部策略的旧环境，可复用其观测契约：  
https://github.com/SafeDL/META_LEARNING/blob/6af8a502c90235063e593d2d36c8aede3aa027f2/highway_env_benchmark/envs/external_cutin.py

### 21.3 项目已有材料的继承范围

[P1] 《面向少样本自适应自动驾驶脆弱性场景挖掘的文献调研、问题动机与研究空白》，2026-09-08：保留历史测试知识、系统响应差异与低算力路线；不继承当时已被后续讨论替代的严格 few-shot 门槛或文献空白结论。

[P2] 《Few-Shot Adaptive Vulnerability Mining：技术方案设计稿》：保留第 9.1 节关于 failure signature 的记录要求；本轮方法用本文件的模式卡与小型 Bayesian logistic 实现，不继续扩建旧 DIVA/CoRe 组合。

---

## 22. 可直接粘贴给 Codex 的执行指令

> 以 `docs/FBRT_FAILURE_PATTERN_MEMORY_CODEX_PLAN_V2_SCENARIOS.md` 为本轮实施目标，尤其执行第 27、30 节对新银行配方和能力匹配的修订。先查看现有 FBRT 代码和银行，保留旧结果，增加一个统一 SUT 接口与真正可以增量提交的失效模式库。按文档实现模式 RBF 字典、源系统独立拟合、目标 Bayesian logistic 更新和新失败区域形成，不再只添加边界距离评分。接通已有 IDM+MOBIL 与 PPO-ECE，按能力选择五类场景；其余候选仅登记。默认不训练任何策略。
>
> 完成两个必要实验：现有银行零仿真重放，以及 4 系统 × 5 场景 × 16 样本的小银行。小银行同时用于版本回归和独立 agent 的顺序测试，不重复物理执行。只做一个 NoMemory 消融。新增物理 episode 总上限 400，所有 smoke、回放和重试都进入同一账本。
>
> 验收首先检查方法真的利用、更新并再次使用失效模式，其次如实报告实际发现收益。不要以每个故障跨多个功能、所有方法全面胜出、统计显著性、完整国标认证或额外查新为前置条件。实现完整而无增益也正常交付；不自动扩展实验。最终输出实现、模式库、可追溯查询、逐任务结果、成本与 `acceptance.json`，不要只输出下一轮实验建议。


---

# 场景与参数增补：V2 的正式执行依据

## 23. 从文献和规范到可执行测试，不再只列场景名字

### 23.1 本次任务定位

优先处理场景覆盖、功能触发、参数耦合和 SUT 能力匹配，保留 FBRT-Memory 主方法。方法有效性仍由实际结果说明；不能把“方法必须获胜”写成筛选场景的规则。

适合本研究的场景应让历史信息具有可用的对象：例如目标切换期间的响应延迟、跟驰制动限制、换道前后间隙判断。这不意味着每个候选都要产生碰撞，也不意味着每种版本修改必须影响多类功能。普通通过区、局部变化区和未受影响区都需要保留。

先在一个受控 ODD 中工作：干燥平整道路、白天、状态观测、最多四个背景车辆、正常车辆动力学。暂不研究天气、图像扰动、行人识别准确率或复杂城市导航。已有工作中对 failure signature、共用场景和源—目标差异的要求继续作为信息设计依据，不恢复早期文件中过多的实验门槛。[P1][P2]

### 23.2 核实到的规范和文献：分别能提供什么

| 来源 ID | 资料与版本 | 本次可使用的内容 | 本次不能据此声称的内容 |
|---|---|---|---|
| S1 | **GB/T 41798-2022**《智能网联汽车 自动驾驶功能场地试验方法及要求》 | 国家场地测试框架；官方条目为现行，2023-05-01 实施 | 本次未完整取得正文参数表；不能把下面研究范围标成其规定值 |
| S2 | **GB/T 47025-2026**《智能网联汽车 自动驾驶功能仿真试验方法及要求》 | 国家仿真测试参考；官方条目显示 2026-01-28 发布并实施 | 只核验到官方元数据，不宣称完成其工具链可信性或全部条款验证 |
| S3 | **DB4403/T 359.1-2023**《智能网联汽车自动驾驶系统技术要求 第1部分：高速公路及快速路自动驾驶》 | 官方完整 PDF 的附录 C，提供切入、切出、停走、紧急制动、部分占道、弯道和两轮车等交互 | **这是地方标准，不是 GB/T 41798 的正文或其数值表** |
| S4 | Euro NCAP **Safe Driving—Vehicle Assistance v1.2，July 2026** | 当前版本的 CCRs/CCRm/CCRb、切入和切出结构及 TTC 口径 | 不是中国法规；也不是把协议所有速度直接给现成 PPO 的理由 |
| S5 | Euro NCAP **AEB Car-to-Car v4.3.1，February 2024** | 固定历史版本的前车制动锚点，便于复现 | 不标为 2026 最新协议；不把 AEB 试验等同于整个 ADS 合格评定 |
| S6 | Euro NCAP **Lane Support Systems v4.3，December 2023** | 同向相邻车辆、后方超越和车道交互的数值参考 | 本项目由 ego 自主决定换道，与协议强制给定偏离轨迹不完全相同 |
| L1 | Feng 等，**Testing Scenario Library Generation …: An Adaptive Framework**，T-ITS 2022；公开预印本 v3 | cut-in 中的距离—相对速度参数化；目标控制器差异需要修正 | 原实验是特定 cut-in 案例，不给出横向运动的通用自然分布 |
| L2 | 李鹏辉等，**面向自动驾驶仿真测试的高覆盖切入场景库生成方法**，《中国公路学报》2024 | 期刊摘要明确采用自然驾驶数据、运动学轨迹参数、联合参数分布和覆盖采样 | 本次未读取其完整数值表；不编造该论文的间距分位数和限值 |
| L3 | Karunakaran 等，**Generating Edge Cases for Testing Autonomous Vehicles Using Real-World Data**，Sensors 2024, 24(1):108 | 从实车数据参数化切入/切出，保留参数之间的关系；作者提供代码与数据入口 | 不把本项目独立设定的矩形范围说成其真实数据分布 |
| R1/R2 | Chen 的 Failure-Based Testing 报告与 Search for Boundary 论文 | 历史失效模式及其邻近通过区的信息组织 | 不保证任何候选场景上的任一模型都一定优于基线 |

本次国内具体参数优先取自 **S3 官方文本的已核实叙述条款**；S1、S2 负责国家标准定位。S3 的表 C.4/C.5 按 Vmax 分档，全文可查，但本轮不复制全部速度档表，以免把条款条件丢失后只保留一个 TTC 数字。PDF 截图服务对部分页面不可用；未核对的插图细节不补造。文末保留官方链接、条款和页码。

### 23.3 可直接登记的来源锚点（不是本项目全域搜索范围）

以下为简要摘记，具体试验还包含道路、车辆状态和路径等条件；本项目只借鉴明确列出的部分。

| source_anchor_id | 已核对的值／口径 | 来源位置 |
|---|---|---|
| DB-CUTIN | 完成换道不大于 3 s；开始切入由预设 TTC 区间触发 | S3，C.4.3.3.2.2，印刷页 31 |
| DB-STOPGO | 前车以 2–3 m/s² 减速停车；符合该条款条件时，起步后 2 s 内达到 10 km/h | S3，C.4.3.3.4.2，印刷页 33 |
| DB-BRAKE | 前车 1 s 内达到 6 m/s² 减速度并刹停 | S3，C.4.3.3.6.2，印刷页 34 |
| DB-PARTIAL | 静止车辆侵入自车车道宽度 1–1.2 m；纵轴与车道线夹角不大于 30° | S3，C.4.3.3.1.1，印刷页 30 |
| DB-CURVE | 弯道半径 250–650 m | S3，C.4.3.3.9.1，印刷页 35 |
| NCAP26-CUTIN | 一组高速锚点：自车 120、目标 70 km/h，TTC=1.5 s；TTC 在**换道完成时**定义 | S4，2.1.4，印刷页 19 |
| NCAP26-CUTOUT | 自车/前车 70/50 或 90/70 km/h，TTC=3 s；这是**切出开始时前车到静止目标**的 TTC | S4，2.1.5，印刷页 19 |
| NCAP26-CCRB | 自车/目标 55/50 km/h，目标减速度 4 m/s² | S4，2.1.3，印刷页 18 |
| NCAP24-CCRB | 自车与前车均 50 km/h，间距 12 或 40 m，减速度 2 或 6 m/s² | S5，8.2.2.3，印刷页 18 |
| NCAP23-OVERTAKE | 自车 72 km/h，同向目标 72 或 80 km/h；相对速度为 0 或 8 km/h | S6，7.4，印刷页 18 |
| ATSLG-RANGE | 距离变量 (0,90] m，距离变化率 [-20,10] m/s | L1，V.B，PDF 第 7 页；保留原文变量定义 |

**执行分类：**精确满足原条款所有所需条件的才可标 `source_anchor_reproduced`；仅借鉴部分速度/减速度但道路、路径或触发不同的，标 `source_anchor_adapted`。本轮默认都是 `source_anchor_adapted` 或 `research_range`，不要求完成认证复现。

---

## 24. 统一参数语义：先消除“同一名字、不同物理含义”

### 24.1 单位与变量

仿真输入统一使用 **m、s、m/s、m/s²、rad**。表中 km/h 只用于保留来源的原值，进入代码时除以 3.6。例如 50 km/h=13.8889 m/s，72 km/h=20 m/s，8 km/h=2.2222 m/s。禁止用 72 直接作为 `speed_mps`。

| 字段 | 定义 |
|---|---|
| `ego_speed_mps` | episode 初始 ego 速度；不等于巡航目标速度 |
| `lead_speed_mps` | 初始前方/切入目标速度 |
| `desired_speed_mps` | 驾驶器的巡航目标；若外部模型自行选速度档，应记录档位而非强加目标 |
| `initial_clearance_m` | t=0 时两车辆沿道路纵向的保险杠净距；相邻车道也用纵向投影定义 |
| `relative_speed_mps` | **v_other − v_ego**；前车比自车慢时为负 |
| `closing_speed_mps` | 对前方车辆为 v_ego−v_lead；对后车另用 `rear_closing_speed_mps`=v_rear−v_ego |
| `lane_change_time_scale_s` | 旧 ControlledVehicle 增益配置中的时间尺度；**不是已验证的实际换道完成时间** |
| `lane_change_duration_nominal_s` | 使用显式参考轨迹时的计划换道时间；与上一字段不可混用 |
| `lane_change_duration_measured_s` | 实际轨迹达到完成判据的时间差；完成前碰撞记未完成，不填计划值 |
| `lead_deceleration_mps2` | 正的制动强度 b；执行加速度为 −b |
| `deceleration_ramp_s` | 从巡航到给定减速度的过渡时间；固定值也是完整配置的一部分 |
| `hold_s`、`restart_acceleration_mps2` | 停车保持时间、重新起步加速度 |
| `event_start_s` | 脚本事件的预定起始时刻；不能由当前方法挑出的结果反向调整 |

车辆长度不同也应正确计算：

\[
g_{front}=x_{lead}-x_{ego}-(L_{lead}+L_{ego})/2.
\]

弯道改用道路弧长坐标，不能继续使用全局 x 差。正的投影净距不是二维不相交证明；场景编译时再用旋转车身多边形检查初始重叠。

### 24.2 四种 TTC 必须分开保存

| 字段 | 谁与谁、什么时候 | 用途 |
|---|---|---|
| `ttc_nominal_start_s` | ego 与切入目标，在切入开始时，按声明的匀速名义模型计算 | 国内切入触发参考、场景定位 |
| `ttc_nominal_completion_s` | ego 与切入目标，在计划完成横向运动时，按同一名义模型计算 | Euro NCAP 切入数值的相近参数化 |
| `lead_to_static_ttc_start_s` | 切出前车与静止目标，在前车开始切出时 | 切出静止车场景的核心参数 |
| `ttc_measured_at_intrusion_s` | 当前 SUT 实际运行，在车身首次进入自车车道时 | 结果记录；不能冒充预先固定的候选输入 |

S3 的切入是在开始时触发；S4 的 cut-in TTC 在换道完成时定义；S4 的 cut-out TTC 又指前车与静止目标。**三个“3 s”不是同一个条件。**

名义闭合速度 c>0 时，g=c×TTC；c≤0 时不定义有限追赶 TTC，记录 `inf`，不用除以一个人为很小的正数制造危险。

计划以 t_s 开始、用 T_L 完成、c 保持常数时：

\[
g_0=c(t_s+T_L+\tau_{completion}).
\]

这是**名义几何构造**，不是新 SUT 真实反应后的精确 TTC；ego 会减速时，应另外记录实测值。不能针对不同 SUT 重新移动目标车来“维持同一 TTC”，否则不再是配对的共同场景。

切出场景中的静止目标位置由前车决定：

\[
x_{static}(0)=x_{lead}(0)+v_{lead}t_s+
 (L_{lead}+L_{static})/2+v_{lead}\tau_{lead\rightarrow static,start}.
\]

不能把这里的前车—静止车 TTC 写成 ego—静止车 TTC。

### 24.3 公共物理契约

新银行默认继续沿用安装版本的车辆动力学、**20 Hz 物理频率**、现有道路宽度。旧 highway-env 常用 4 m 车道，Euro NCAP 切入的 3.5 m 是其路径横移值；不为对齐文字而静默缩窄现有车道。精确 3.5 m 路径只能作为新的 `context_id`。

脚本车不读取待测算法、故障名称、FBRT 的分数或未揭示结果。所有车辆从 episode 开始就在场；除明确另立感知模型的任务外，不允许为制造“揭示”而临时生成/隐藏碰撞物体。

同一个候选允许不同 SUT 采取不同制动/换道动作。反事实比较应共享初始状态与脚本，不应共享强制 ego 轨迹。

---

## 25. 十四类候选场景卡：明确范围、功能、适用对象和实现量

**读表规则：**除第 23.3 节已经标为来源值的锚点，下面所有连续范围和默认值均为 **RESEARCH_RANGE（本项目设计值）**，不是国标规定值，也不是从自然驾驶数据估计的百分位数。`主范围`、`扩展范围`是可选配置，不要求同时运行。每个执行 profile 只变化两项，其他上下文固定并入 fingerprint。

### S01｜前方车辆切入（`fbrt_cutin`，已有骨架）

**功能：**新前方目标进入本车道时，目标选择、反应与纵横向避碰是否协调。来源结构：S3 C.4.3.3.2、S4 2.1.4；距离—速度研究参考 L1，轨迹相关性参考 L2/L3。

**角色与过程：**ego 直行；相邻前车以固定纵向速度行驶，在 1 s 发起向 ego 车道的完整切入。两车道，持续 12 s。ego 可以制动或自主换道，不强制只制动。

| 项目 | 主范围／默认 | 可选扩展 |
|---|---|---|
| Active-1 初始纵向净距 | **8–60 m** | 5–90 m |
| Active-2 旧控制器换道时间尺度 | **1.5–3.0 s**，字段用 `lane_change_time_scale_s` | 0.8–4.0 s；真实完成时间另记 |
| ego/切入车速度 | **25/20 m/s** | 固定速度上下文 20/15、30/25 m/s；不是三组都跑 |
| 触发 | 固定 t=1 s | 可选名义 TTC 参数化，替代净距而不是额外增加独立变量 |
| 显式轨迹备选 | `lane_change_duration_nominal_s=2.5–4.0 s` | 仅轨迹跟踪器实现后启用，不与增益时间尺度混称 |

**可选 source-like 锚点：**S4 的 120/70 km/h、完成时 TTC 1.5 s，仅登记为协议参考；未复现路径前不声称完全等价，也不默认在最高速度上加载 PPO。

**可迁移信息假设：**同一切入冲突中，不同系统的失败位置可能随着反应时序或动作选择改变；它既有可复用的共同几何，也有系统差异。历史全失败/全通过不作为删除本模板的条件。

**工程：**复用已有切入车；记录首次车身入侵、完成时刻、ego 首次明显制动/换道、碰撞对象。进入/离开方向必须区分，不能只用 `abs(lateral_offset)>0.25` 命名切入故障。

### S02｜前车切出后存在静止车辆（`fbrt_cutout_static`，已有三车结构）

**功能：**前车对象切换与静止目标响应。来源：S3 C.4.3.3.5、S4 2.1.5。

**过程：**ego 跟随 VT1；静止 VT2 从起始就在前方；VT1 在 1 s 开始切出。ego 对 VT2 的响应由自身控制器决定。在状态观测环境中这是目标交互测试，不称真实视觉遮挡测试。

| 项目 | 主范围／默认 | 可选扩展 |
|---|---|---|
| Active-1 ego—VT1 初始净距 | **8–40 m** | 5–60 m |
| Active-2 VT1→静止 VT2 在切出开始时的 TTC | **2.0–4.5 s** | 1.5–5.0 s |
| ego/VT1 初速 | **20/20 m/s** | 固定为来源相近的 25/19.4444 m/s 等，另设 context |
| 旧横向跟踪时间尺度 | **0.8 s**，保留已验证脚本起点 | 不称实际 0.8 s 完成；实际横移过程必须记录 |
| 车道/持续时间 | **2 车道、14 s** | 只在已有轨迹显示脚本还未完成时按同一规则延长所有版本 |

**约束：**VT1 切出轨迹与 VT2 不应先发生纯背景碰撞；若发生，记 `background_inconclusive`，不能当作 ego 回归或直接填通过。几何预筛只看目标脚本和车身，不查看哪个方法获胜。

**系统匹配：**可停车的纵向控制器适用；只能保持 ≥20 m/s 的 PPO 只有在存在真实横向避让空间时，才能作为“转向避碰”任务参加，不能按停车合格性评价。

### S03｜前方车辆减速至停止（`fbrt_lead_emergency_brake`，已有）

**功能：**对已在本车道内的目标制动响应，不与“切入识别”混淆。来源：S3 C.4.3.3.6；S4 CCRb；S5 8.2.2.3。

| 项目 | 主范围／默认 | 可选扩展 |
|---|---|---|
| Active-1 初始净距 | **12–80 m** | 已有 `legacy_exact` 的 5–120 m 单独保留 |
| Active-2 目标减速度 | **2–6 m/s²** | 6–8 m/s² 为压力配置，不默认启用 |
| ego/前车初速 | **20/20 m/s** | 来源锚点双方 50 km/h=13.8889 m/s |
| 制动触发/建立时间 | **t=1 s；0.5 s 内线性建立减速度** | 旧瞬时制动脚本保留为旧 contract |
| 终态 | 前车速度截到 0，保持停车 | 禁止减速到负速度或自动恢复巡航 |
| 道路/时长 | **1 车道、18 s** | 多车道避险单列 `escape_allowed`，不与一车道合并 |

**来源锚点：**`vE=vL=50 km/h`、`gap∈{12,40}m`、`b∈{2,6}m/s²`。四组合仅在选择 source-anchor profile 时占用已有 16 个名额，不额外再做四次。

**特别说明：**修改减速度建立时间会改变物理结果，不能复用旧银行的同名缓存。首轮未实现 ramp 时直接使用旧 instantaneous profile，标明适配差异；不以新增完整制动执行器为整个方法交付前置条件。

**适用：**IDM/FVDM、具备停车动作的策略。PPO 目标档位均大于零且无独立制动动作时，本停车任务标 `NOT_APPLICABLE_ACTION_CAPABILITY`。

### S04｜前车停车—保持—起步（`fbrt_stop_hold_go`，已有）

**功能：**接近停车目标、停车后的行为与起步响应；不把整个过程全归为低 TTC 跟驰。来源：S3 C.4.3.3.4。

| 项目 | 主范围／默认 | 可选扩展 |
|---|---|---|
| Active-1 初始净距 | **6–35 m** | 4–45 m |
| Active-2 前车减速度 | **1.5–3.5 m/s²** | 3.5–5 m/s² 为压力配置 |
| ego/前车初速 | **15/15 m/s** | 5–10 m/s 低速切片，需单独能力声明 |
| 前车保持/起步加速度 | **2 s / 1.5 m/s²** | 保持 1–4 s、起步 0.8–2.0 m/s²，仅作固定上下文 |
| 道路/时长 | **2 车道、30 s** | 跟驰-only 与允许换道超越的功能口径分别记录 |

**状态机：**巡航→减速至零→保持→起步→恢复巡航，不能用一个固定制动窗口代替。允许换道的 ADS 可以绕行；不把“没有和前车一起停车”判失败。

**和来源的差异：**S3 的起步条件与被测车状态有条件关系；本项目为共享确定背景，默认按前车自身停车时刻保持 2 s 后起步，明确标 `scripted-time adaptation`。不能在不同 SUT 上根据各自反应悄悄改变共同背景，又继续称严格配对。

**能力/历史：**没有源失败时照常保留；这可以检验模式库对“无旧边界”的处理，但不要求本模板必须失败。现有 `slow_front_brake2` 在 15 m/s 初态即能触发低于 18 m/s 的分支，属于明显功能退化对照，不应靠扩大该模板权重支配总结果。

### S05｜自车换道，目标车道后方快速来车（`fbrt_lane_change_rear`，必要新增）

**功能：**换道动机、后车安全判断、等待与执行；是本轮区别于纵向控制的必要交互。来源借鉴 S6 同向超越结构；自车自由决策是本研究适配。

| 项目 | 主范围／默认 | 可选扩展 |
|---|---|---|
| Active-1 目标车道后车初始净距 | **8–60 m** | 5–80 m |
| Active-2 后车接近速度 vR−vE | **0–10 m/s** | 10–12 m/s；上限还需满足固定 ODD 速度约束 |
| ego 速度 | **25 m/s** | 源协议相近上下文 20 m/s |
| 本车道慢前车速度/净距 | **18 m/s / 25 m** | 20 m/s / 35 m 为较缓上下文，非追加实验 |
| 后车速度 | **vE+Active-2** | 由相对速度派生，不再独立采样 |
| 后车行为/道路/时长 | **匀速直行；2 车道；12 s** | 反应式后车是另一场景合同，不能中途更换 |

**相近来源锚点：**20 m/s 的 ego 和 22.2222 m/s 的后车，对应 S6 的 72/80 km/h；本项目 8–60 m 不是协议给定后车距离。

**执行要求：**ego 是否换道由 SUT 决定，不能调用脚本强制换道。没有换道时仍是合法测试，通过数据保留，并将 `ego_lane_change_initiated=false` 作为功能行为描述。不能把未发生侧碰误写成后车检查已经覆盖。

**关键版本：**`mobil_ref_v2` 对 `mobil_rear_guard_off_v2`；PPO 的横向选择也可参加，Profiled-IDM 固定车道只作为不具备该功能的对照。

### S06｜同车道慢车跟驰/绕行（`fbrt_moving_lead`，轻量配置候选）

**功能：**速度差响应及持续跟驰，提供没有突然事件的对照。来源：S4 CCRm；S3 C.4.3.3.8 还包含横向压线，其完整条件不与本卡完全相同。

| 项目 | 主范围／默认 | 可选扩展 |
|---|---|---|
| Active-1 初始净距 | **15–90 m** | 8–100 m |
| Active-2 前车速度 | **20–27 m/s**（现有高速 PPO 配方） | 5–20 m/s 仅适用于低速/停车能力匹配的系统，或明确允许超越 |
| ego 初速 | **25 m/s** | 固定 20 或 30 m/s 上下文 |
| 道路/时长 | **2 车道、15 s** | 1 车道 follow-only 需另立模板上下文 |
| 目标行为 | 全程保持车道和速度 | 无制动事件 |

**价值：**把普通稳定交互也放进同一个测试池，防止所有候选都只考察一个紧急制动分支。20–27 m/s 的选择来自现有动作接口能力，不是对 Euro NCAP 目标车速度的复制。

**实现：**复用 ScriptedVehicle 的巡航阶段/原有 slow-lead 逻辑，不需要新仿真器；不能继续保留一个实际不生效的 `timing` 作为可变参数。

### S07｜本车道静止车辆（`fbrt_stationary_lead`，可选）

**功能：**静止目标处理；是 S02 去掉目标切换后的对照。来源：S4 CCRs、S5 stationary-target 测试结构。

| 项目 | 主范围／默认 | 可选扩展 |
|---|---|---|
| Active-1 到静止车的初始净距 | **15–100 m** | 10–140 m |
| Active-2 ego 初速 | **10–25 m/s** | 25–30 m/s，前提是当前系统支持 |
| 目标速度 | **0 m/s** | 固定 |
| 道路/时长 | **1 车道、20 s**，停车功能配置 | 2 车道 `escape_allowed` 是另一个 context |

**用途：**区分“静止目标就处理不好”与“只有切出揭示后才处理不好”。同一个 `stop` 需求不能给没有停车动作的 PPO。

**成本：**非默认必跑；需要替换一类场景时，使用同样 16 个名额，不新加一轮实验。

### S08｜切入后短时间内制动（`fbrt_cutin_then_brake`，已有动作的组合候选）

**功能：**目标进入之后紧接着状态变化，测试连续反应而非单次切入。来源是 S3/S4 中切入与前车制动要素的**研究组合**，不是声称标准有完全相同的组合条款；轨迹相关性参考 L2/L3。

| 项目 | 主范围／默认 | 可选扩展 |
|---|---|---|
| Active-1 初始净距 | **10–60 m** | 6–80 m |
| Active-2 制动强度 | **1–6 m/s²** | 6–8 m/s² |
| ego/目标初速 | **25/20 m/s** | 与 S01 相同的固定速度上下文 |
| 换道时间尺度 | **1.5 s**（旧跟踪器命令） | 明确轨迹版本时改为 nominal duration |
| 切入完成后制动等待 | **0.3 s** | 1.0 s 作为固定较缓切片，不独立增维 |
| 制动持续/最低速度 | **1.5 s / 5 m/s** | 不允许负速度 |
| 道路/时长 | **2 车道、18 s** | 空余车道可供 ego 真实避让 |

**触发：**以脚本车实际完成横移的几何事件计时；不是用 ego 的事故结果触发，也不是简单“t_start+命令时间”冒充已经完成。未完成时有明确定义的最大场景时长，结果标 `event_incomplete`。

**实现：**把已有 lane-change、brake 两个动作串成背景状态机，不另建新的训练任务。对不能停车的 PPO，仅以允许换道的短暂减速交互参加，不能说它通过了低速跟驰能力测试。

### S09｜前车切出后道路释放（`fbrt_cutout_release`，可选）

**功能：**前车退出后是否合理恢复速度，是否错误继续制动/把相邻车当作前车。来源：S3 C.4.3.3.3。

| 项目 | 主范围／默认 | 可选扩展 |
|---|---|---|
| Active-1 ego—前车净距 | **10–45 m** | 5–60 m |
| Active-2 相邻后车纵向净距 | **0–15 m** | 15–30 m |
| ego/前车/相邻车速度 | **20/20/20 m/s** | 固定上下文，不独立全组合 |
| 前车切出 | **t=1 s，time scale=1.5 s** | 实际完成时刻另存 |
| ego 期望巡航速度 | 由 SUT 正常策略决定；规则驱动固定25 m/s | 不能覆盖 PPO 自选动作 |
| 道路/时长 | **2 车道、12 s** | 无静止目标 |

S3 提供相邻目标接近 ego 后端的几何约束；上表 0–15 m 是研究扩展。S09 和 S02 必须使用不同模板 ID：一个是道路释放，一个是新静止目标出现。

**主失败仍为碰撞**。恢复加速时间只是解释字段，不为让方法胜出临时把不加速也合并计入主失败。

### S10｜目标车道前后夹逼的换道间隙（`fbrt_lane_change_bounded_gap`，可选）

**功能：**同时判断目标车道前车与后车，不让后车检查与前车检查被同一个“最近障碍”分数代替。来源结构：S6 的相邻车交互与 S3 的多个目标元素；这是本项目组合设计。

| 项目 | 主范围／默认 | 可选扩展 |
|---|---|---|
| Active-1 目标车道后车净距 | **8–60 m** | 5–80 m |
| Active-2 目标车道前车净距 | **8–45 m** | 5–60 m |
| ego/本车道慢车/目标后车/目标前车速度 | **25/20/30/23 m/s** | 固定上下文 |
| 本车道慢车间距 | **35 m** | 25 m 为可选上下文 |
| 道路/时长 | **2 车道、12 s** | 四辆车，包含 ego |

两 active 参数都相对 ego 的初始纵向位置定义，前后对象不能颠倒。后车与目标前车从起始就存在；不在 ego 换道后突然刷车。

**适用：**原生 MOBIL/PPO；固定车道的 IDM 不作为换道功能验证对象。不能强制 ego 接受间隙后再把碰撞算其自主换道失败。

### S11｜静止车辆部分占道（`fbrt_partial_lane_obstacle`，可选）

**功能：**车道归属、目标横向位置与车身碰撞几何之间的差异。来源：S3 C.4.3.3.1。

| 项目 | 主范围／默认 | 可选扩展 |
|---|---|---|
| Active-1 纵向净距 | **15–90 m** | 10–120 m |
| Active-2 目标侵入 ego 车道宽度 | **0.2–1.2 m** | 1.2–1.5 m |
| ego/目标速度 | **20/0 m/s** | 固定 |
| 目标纵轴夹角 | **0 rad** | 0–30° 仅固定切片，使用旋转多边形 |
| 道路/时长 | **2 车道、15 s** | 相邻车道保留可通行空间 |

**源锚点：**1.0 与1.2 m 是 S3 支持的占道尺度；0.2–1.0 m 是研究扩展。必须按车身多边形计算实际占道宽度，不按中心属于哪个车道来替代。

这适合研究状态观测驾驶器的几何处理，不需要将其夸大为摄像头漏检。

### S12｜连续两车切入／前车对象切换（`fbrt_successive_cutins`，可选）

**功能：**第一次切入后又遇到第二个目标，检查缓存、目标切换和响应状态是否正确重置。来源要素：S3 C.4.3.2 的多目标与切入条件，以及 L2/L3 的时序参数化；完整组合是研究设计。

| 项目 | 主范围／默认 | 可选扩展 |
|---|---|---|
| Active-1 两次切入开始时间差 | **0.8–3.0 s** | 0.5–4.0 s |
| Active-2 第二辆车在其切入开始时的名义纵向净距 | **10–50 m** | 6–70 m |
| ego/第一目标/第二目标速度 | **25/22/20 m/s** | 固定 |
| 第一次开始/名义开始净距 | **1 s / 35 m** | 固定 |
| 两目标换道 time scale | **1.5 s** | 固定 |
| 道路/时长 | **2 车道、18 s** | 所有车辆初始即在道路上 |

**几何预筛：**第二目标初始位置通过名义速度倒推；不允许目标车初始重叠、车道内顺序自相矛盾或在与 ego 交互前先互撞。预筛使用背景脚本，不读取目标SUT结果。

这类场景只作为候选扩展，首轮不增加新的多智能体策略训练。

### S13｜弯道前车缓行（`fbrt_curved_slow_lead`，P2）

**功能：**曲率、车道投影、跟踪与前方目标识别之间的关系。来源：S3 C.4.3.3.9。

| 项目 | 主范围／默认 | 可选扩展 |
|---|---|---|
| Active-1 曲率半径 | **250–650 m** | 150–250 m 仅研究压力区 |
| Active-2 道路弧长净距 | **15–80 m** | 10–120 m |
| ego/前车速度 | **20/15 m/s** | 固定 |
| 道路 | 直线—圆弧—直线，2车道 | 同一参数配置所有系统共享 |
| 时长 | **20 s**，道路长度应覆盖完整运行 | 不把驶出道路当控制器故障 |

半径范围直接参照 S3；本文不默认复现原条款中的目标压线位置。若要复现压线，应另立 context。

**不在本轮执行：**当前部分脚本硬编码直路和 lane index，已有高速 PPO也未确认弯道适配。先登记能力与几何需求，不为凑场景数强行跑。

### S14｜同车道两轮车目标（`fbrt_same_lane_ptw`，P2）

**功能：**小目标横向位置和跟随/超越决策。来源：S3 C.4.3.3.7；S4 还包含专门的两轮目标测试。

| 项目 | 主范围／默认 | 可选扩展 |
|---|---|---|
| Active-1 初始净距 | **10–60 m** | 6–90 m |
| Active-2 目标相对车道中心横向偏移 | **−1.0 至 +1.0 m** | 按车身宽度与车道宽度进一步约束 |
| ego/目标速度 | **13.8889/6.9444 m/s**，即50/25 km/h | 来源两轮车速度20–30 km/h只作候选上下文 |
| 目标几何 | 首轮候选长2.2、宽0.8 m，**研究设定** | 不冒称官方软目标尺寸 |
| 道路/时长 | **2车道、18 s** | 固定 |

**边界：**不能把汽车改一个 name 就称已测试摩托车；至少要改变几何与运动状态，且说明没有视觉识别模块。当前 PPO 速度能力不匹配，默认不执行。该卡用于未来扩展而非本轮硬验收。

---

## 26. 参数不能各自随便采样：只加必要的工程耦合

### 26.1 不独立采样互相确定的量

同一场景不能把 `gap`、`closing_speed`、`TTC` 三者都当独立自由度。类似地，后车速度应由 ego 速度和后车接近速度派生，停止时间应由初速、减速度及其建立过程决定。

第一版每个模板仍只保留两个 active 参数。其余速度、车道数、动作频率、保持时间、目标尺寸均为固定 `context`。需要换上下文时，另生成一个可追溯 profile，不把多个上下文悄悄投影到同一个二维点。

L2 与 L3 支持重视参数联合关系；本轮用几何等式和脚本预筛即可，不新增 GMM/HMC 或轨迹生成网络。没有真实联合数据时不宣称采样服从自然驾驶分布。

### 26.2 明确轨迹版本时的轻量参考

若使用显式平滑参考横移，可用研究参考曲线：

\[
y(r)=y_0+\Delta y(10r^3-15r^4+6r^5),\qquad r=(t-t_s)/T_L\in[0,1].
\]

该曲线的峰值横向加速度为：

\[
|a_y|_{max}=\frac{10\sqrt3}{3}\frac{|\Delta y|}{T_L^2}.
\]

例如横移3.5 m、T_L=3 s时，名义峰值约2.25 m/s²；T_L=1.5 s时约8.98 m/s²。二者不是同等正常程度的动作。

这是数学构造示例，**不是标准规定，也不是命令时间尺度的实际加速度**。首轮可继续使用已有 ControlledVehicle，不要求重写完整轨迹控制器；但必须记录真实横移时间，不能把 KP 参数硬写为规范中的完成时间。若启用上述轨迹，实际车身仍由动力学执行，禁止直接跳变位置穿过 ego。

### 26.3 时长与可见窗口

停车任务的 episode 应至少覆盖前车刹停和规定保持；停走任务还需覆盖实际起步。使用事先声明的脚本时长公式或本目录默认值，不依赖某个 SUT 是否失败临时缩短。

运行到时长末尾只说明“本测试窗口内没有失败”；已知碰撞发生在窗口外时不能通过缩短窗口变成无回归。旧银行的既有时长不改标签；新时长建立新缓存键。

### 26.4 最小有效性检查

只要求：输入单位正确、初始无车身重叠、目标脚本能够按定义运行、结果区分 ego 碰撞与背景单独碰撞、所有版本共用候选。无碰撞和无回归都是允许结果。

不额外训练可避免性网络；不要求每个压力场景都被证明存在全局最优避险轨迹；不开展全标准认证。几何错误是实现错误，不是可以计入 failure memory 的驾驶缺陷。

---

## 27. SUT—场景匹配：这是本次补充最需要落实的一项

### 27.1 静态能力检查，禁止用“不可能执行的动作”评价回归

当前项目的 `ExternalCutInEnv` 使用 `DiscreteMetaAction`，没有显式设置 `target_speeds`。[C11]

highway-env 官方实现会回退到 `MDPVehicle.DEFAULT_TARGET_SPEEDS`；所核对版本为 `np.linspace(20,30,3)`，即20/25/30 m/s。[E1][E2]

因此，Codex 在建新银行前直接输出：

```text
installed_highway_env_version
policy_observation_shape
policy_action_names
policy_target_speeds_mps
minimum_commandable_speed_mps
can_command_full_stop
can_change_lane
supported_road_topology
```

读取本地实际配置是静态能力声明，不是追加一轮实验。若本地有效配置已经改变，按本地实际契约记录，不强制用网页当前默认值。

**不能做：**为让原 PPO 通过停车场景，悄悄把动作档位改成0/15/30；这会改变原策略动作语义。需要研究这样的适配时，应明示为另一个系统，不作为原权重“直接复现”。

### 27.2 适用矩阵

| 任务组 | Profiled-IDM/FVDM | 原生 IDM+MOBIL | 现有 PPO+高速离散速度档 |
|---|---|---|---|
| S01切入、S02切出静止目标（两车道） | 纵向响应适用 | 纵横向响应适用 | 允许实际换道避险时适用；不是停车功能验收 |
| S03一车道刹停 | 适用 | 适用 | 无停车动作则不适用，不以必撞数量评价方法 |
| S04跟驰停车起步 | 适用 | 跟随/超越按任务声明 | 只能变道绕行版本适用；必须停车版本不适用 |
| S05/S10换道后车/间隙 | 不具有自主换道，作为功能外对照 | 核心适用 | 高速横向决策适用 |
| S06慢车、S08切入后短暂制动 | 适用 | 适用 | 速度范围兼容或允许真实避让时适用 |
| S07一车道静止车停车 | 适用 | 适用 | 不具备停车动作则不适用 |
| S09释放、S11部分占道、S12连续切入 | 有相应纵向子任务 | 适用 | 在既有道路/观测能力内适用 |
| S13弯道、S14两轮车 | 需额外适配 | 可作为后续适配 | 本轮能力未确认，不执行 |

`NOT_APPLICABLE` 必须基于静态动作/任务契约，而不是因为某个算法已经撞了就把该案例删掉。适用任务中的真实碰撞仍是失败，不能免除。

### 27.3 新银行只选择一个配方，不把候选目录全部执行

**配方 A：`legacy_core_5`**

```text
S01 切入
S02 切出静止车
S03 前车紧急制动
S04 停—保持—起步
S05 换道后车
```

用途：保留既有四模板路线，并突出纵向/停车能力。已有银行继续使用此配方；PPO 不适用任务标 NA，不补标签，不自动重建全部数据。

**配方 B：`highway_policy_compatible_5`（新银行尚未运行、且现有 PPO 无停车动作时默认采用）**

```text
S01 切入
S02 切出静止车（允许真实横向避让）
S05 换道后车
S06 同车道慢车（前车20–27 m/s）
S08 切入后短暂制动（两车道）
```

该配方用两种已有动作的轻量配置，替换要求停车的任务；四个系统仍在同一批80个场景上执行，保持320次主体上限。保留横向避让空间，不用临时删车道来提高碰撞率。S06复用巡航，S08组合已有切入/制动状态，不另建新环境。

**不再要求“最多只改一个模板”而把不适用的停车任务硬塞进 PPO。**本增补允许上述必要替换，但不允许两套配方都跑，也不运行全部14类。

配方决策只读取静态能力、已有数据完整性和本地模型可用性；不能运行方法比较后选更有利的配方。开始生成银行后冻结。若原版银行已存在，先重放并标注能力边界，新增部分只使用尚未消耗的总预算。

### 27.4 回归与跨 agent 的组织不变

回归只在对应父版本实际通过的候选上计算；其他历史失败作为模式证据仍可保留。跨 agent 使用共同适用的候选集合，不把“所有历史系统都通过”作为默认过滤。

对于没有适用交集的任务，分别报告而不强行合并。比较 FBRT、NoMemory 与所有基线时，**相同任务内的候选、名义上下文、预算和起始历史视图一致**。

---

## 28. 只做必要实验：参数目录丰富，物理执行不膨胀

### 28.1 每模板16个样本，不做全因子笛卡尔积

默认每个选中模板：

- 4 个 `anchor_like`：依据本目录固定上下文和所选二维范围，在归一化位置 `(0.25,0.25)`、`(0.25,0.75)`、`(0.75,0.25)`、`(0.75,0.75)` 取点；它们是研究代表点，除完全匹配来源定义外不标来源原值。
- 12 个 `space_filling`：从固定种子的二维 Sobol 序列取前12个非重复、满足纯几何约束的点。可先产生16点后截取；不把截取后的12点夸称为完整平衡Sobol网格。

启用第23.3节精确/适配锚点时，用它们替换上述4个名额，不另加样本。几何不合法点在仿真前重新编译，不能根据父/目标碰撞结果重采样。随机种子用于场景采样和平局，不被当作独立软件缺陷。

不同来源的速度档不全跑。每次选择一个支持当前 SUT 的固定上下文。方法比较完全复用已经生成的银行，不另开真实仿真。

### 28.2 不让难度组织再次退化为“全安全或全必撞”筛选

可以利用已有历史记录给候选附加 `ordinary / boundary_neighbour / stress / uncovered` 标签；该标签用于解释覆盖，不作为必须达到某一碰撞比例的验收条件。

如果部分功能无旧失败，保留；如果某个明显修改在整类功能均失败，保留并单列，不继续增强。判断场景值得登记的依据是功能可执行和交互有来源，而不是完整方法是否在该模板获胜。

只用三张零额外仿真的组织检查表即可：

1. `capability_matrix.csv`：哪些构建可以执行哪些功能；
2. `scene_parameter_manifest.csv`：每个候选的独立参数、派生参数、来源与单位；
3. `response_by_task.csv`：已有银行中每个父/子/agent 的通过与失败数，用于说明任务分布，不作为重新筛选获胜场景的理由。

不追加总体风险相关性研究、很多粒度的覆盖统计、多个噪声实验或参数消融矩阵。

### 28.3 旧数据、扩展范围与模型坐标

保持实验 A 的旧数据为 `legacy_exact`。新 profile 改速度、车道、脚本阶段或频率时，不冒充旧 episode；完整参数和执行合同进入缓存键。

仅改变范围上下限但物理变量含义相同时，可以从旧记录的物理值重新编码新 RBF 特征；不能沿用旧归一化坐标，把30 m在两个不同范围中误认为同一点。

改变固定速度/道路拓扑/动作阶段含义后，必须保留 `context_id`。当前模型未显式建模该上下文时，分开建模或采用冷启动覆盖特征；不擅自当作同一二维函数。交互模式可以有关联，但不等于标签能直接搬运。

### 28.4 成本上限保持原版

```text
实验A旧银行：新增0
实验B：最多4×5×16=320
smoke：最多24
配对回放：最多8
必要修复和复测：最多48
本任务全部新增物理调用：不超过400
```

目录注册、来源映射、单位计算、能力检查、全部选择器和唯一NoMemory消融均不产生额外物理配额。若模型不可用或能力不适用而少跑，允许总数小于上限，不能把剩余配额当必须花完的指标。

---

## 29. 机器可读场景目录（复制为 YAML；只编译配方中的条目）

下面 `research_bounds` 统一使用SI单位；每个范围均为本项目设计。`evidence` 表示结构来源，不表示所有数值直接摘自该来源。S13/S14 不要求本轮实现执行器。

```yaml
schema: fbrt_scenario_catalogue_v2
reviewed_repo_sha: 6af8a502c90235063e593d2d36c8aede3aa027f2
canonical_units: SI
default_range_status: RESEARCH_RANGE
reference_values_in_section: '23.3'
physics_hz: 20
lane_width_policy: preserve_installed_and_fingerprint
max_selected_templates: 5
samples_per_template: 16
new_builds:
- mobil_ref_v2
- mobil_rear_guard_off_v2
- ppo_ref_v2
- ppo_obs_age020_v2
max_main_bank_episodes: 320
max_total_new_physical_episodes: 400
new_bank_recipe_policy: use_legacy_if_bank_exists_else_choose_by_static_action_capability
recipes:
  legacy_core_5:
  - S01
  - S02
  - S03
  - S04
  - S05
  highway_policy_compatible_5:
  - S01
  - S02
  - S05
  - S06
  - S08
sampling:
  anchor_like_count: 4
  space_filling_count: 12
  target_outcome_filtering: false
  all_methods_share_candidates: true
scenarios:
- id: S01
  template_id: fbrt_cutin
  implementation: existing
  priority: P0
  evidence:
  - S3
  - S4
  - L1
  - L2
  - L3
  parameterization_version: research_v2
  research_bounds:
    initial_clearance_m:
    - 8
    - 60
    lane_change_time_scale_s:
    - 1.5
    - 3.0
  fixed_context:
    ego_speed_mps: 25
    lead_speed_mps: 20
    event_start_s: 1
    lane_count: 2
    duration_s: 12
  required_capability:
  - longitudinal_or_lateral_avoidance
  execute_all_catalogue_entries: false
- id: S02
  template_id: fbrt_cutout_static
  implementation: existing
  priority: P0
  evidence:
  - S3
  - S4
  parameterization_version: research_v2
  research_bounds:
    initial_clearance_m:
    - 8
    - 40
    lead_to_static_ttc_start_s:
    - 2.0
    - 4.5
  fixed_context:
    ego_speed_mps: 20
    lead_speed_mps: 20
    lane_change_time_scale_s: 0.8
    event_start_s: 1
    lane_count: 2
    duration_s: 14
  required_capability:
  - full_stop_or_lateral_escape
  execute_all_catalogue_entries: false
- id: S03
  template_id: fbrt_lead_emergency_brake
  implementation: existing
  priority: P0
  evidence:
  - S3
  - S4
  - S5
  parameterization_version: research_v2
  research_bounds:
    initial_clearance_m:
    - 12
    - 80
    lead_deceleration_mps2:
    - 2
    - 6
  fixed_context:
    ego_speed_mps: 20
    lead_speed_mps: 20
    deceleration_ramp_s: 0.5
    event_start_s: 1
    lane_count: 1
    duration_s: 18
  required_capability:
  - full_stop
  execute_all_catalogue_entries: false
- id: S04
  template_id: fbrt_stop_hold_go
  implementation: existing
  priority: P0
  evidence:
  - S3
  parameterization_version: research_v2
  research_bounds:
    initial_clearance_m:
    - 6
    - 35
    lead_deceleration_mps2:
    - 1.5
    - 3.5
  fixed_context:
    ego_speed_mps: 15
    lead_speed_mps: 15
    hold_s: 2
    restart_acceleration_mps2: 1.5
    event_start_s: 1
    lane_count: 2
    duration_s: 30
  required_capability:
  - stop_restart_or_declared_overtake
  execute_all_catalogue_entries: false
- id: S05
  template_id: fbrt_lane_change_rear
  implementation: new_structure
  priority: P0
  evidence:
  - S6
  parameterization_version: research_v2
  research_bounds:
    rear_clearance_m:
    - 8
    - 60
    rear_closing_speed_mps:
    - 0
    - 10
  fixed_context:
    ego_speed_mps: 25
    lead_speed_mps: 18
    lead_clearance_m: 25
    lane_count: 2
    duration_s: 12
  required_capability:
  - autonomous_lane_change
  execute_all_catalogue_entries: false
- id: S06
  template_id: fbrt_moving_lead
  implementation: reuse_cruise
  priority: P1
  evidence:
  - S4
  parameterization_version: research_v2
  research_bounds:
    initial_clearance_m:
    - 15
    - 90
    lead_speed_mps:
    - 20
    - 27
  fixed_context:
    ego_speed_mps: 25
    lane_count: 2
    duration_s: 15
  required_capability:
  - speed_tracking_or_lateral_escape
  execute_all_catalogue_entries: false
- id: S07
  template_id: fbrt_stationary_lead
  implementation: reuse_static
  priority: P1
  evidence:
  - S4
  - S5
  parameterization_version: research_v2
  research_bounds:
    initial_clearance_m:
    - 15
    - 100
    ego_speed_mps:
    - 10
    - 25
  fixed_context:
    lead_speed_mps: 0
    lane_count: 1
    duration_s: 20
  required_capability:
  - full_stop
  execute_all_catalogue_entries: false
- id: S08
  template_id: fbrt_cutin_then_brake
  implementation: compose_existing_actions
  priority: P1
  evidence:
  - S3
  - S4
  - L2
  - L3
  parameterization_version: research_v2
  research_bounds:
    initial_clearance_m:
    - 10
    - 60
    lead_deceleration_mps2:
    - 1
    - 6
  fixed_context:
    ego_speed_mps: 25
    lead_speed_mps: 20
    lane_change_time_scale_s: 1.5
    brake_after_measured_merge_s: 0.3
    brake_duration_s: 1.5
    lead_speed_floor_mps: 5
    event_start_s: 1
    lane_count: 2
    duration_s: 18
  required_capability:
  - longitudinal_or_lateral_avoidance
  execute_all_catalogue_entries: false
- id: S09
  template_id: fbrt_cutout_release
  implementation: reuse_cutout
  priority: P1
  evidence:
  - S3
  parameterization_version: research_v2
  research_bounds:
    initial_clearance_m:
    - 10
    - 45
    adjacent_rear_clearance_m:
    - 0
    - 15
  fixed_context:
    ego_speed_mps: 20
    lead_speed_mps: 20
    adjacent_speed_mps: 20
    lane_change_time_scale_s: 1.5
    event_start_s: 1
    lane_count: 2
    duration_s: 12
  required_capability:
  - speed_recovery
  execute_all_catalogue_entries: false
- id: S10
  template_id: fbrt_lane_change_bounded_gap
  implementation: additional_actor
  priority: P1
  evidence:
  - S3
  - S6
  parameterization_version: research_v2
  research_bounds:
    rear_clearance_m:
    - 8
    - 60
    target_lane_front_clearance_m:
    - 8
    - 45
  fixed_context:
    ego_speed_mps: 25
    lead_speed_mps: 20
    lead_clearance_m: 35
    rear_speed_mps: 30
    target_lane_front_speed_mps: 23
    lane_count: 2
    duration_s: 12
  required_capability:
  - autonomous_lane_change
  execute_all_catalogue_entries: false
- id: S11
  template_id: fbrt_partial_lane_obstacle
  implementation: static_geometry_variant
  priority: P1
  evidence:
  - S3
  parameterization_version: research_v2
  research_bounds:
    initial_clearance_m:
    - 15
    - 90
    obstacle_lane_incursion_m:
    - 0.2
    - 1.2
  fixed_context:
    ego_speed_mps: 20
    lead_speed_mps: 0
    obstacle_yaw_rad: 0
    lane_count: 2
    duration_s: 15
  required_capability:
  - full_stop_or_lateral_escape
  execute_all_catalogue_entries: false
- id: S12
  template_id: fbrt_successive_cutins
  implementation: additional_actor_schedule
  priority: P1
  evidence:
  - S3
  - L2
  - L3
  parameterization_version: research_v2
  research_bounds:
    cutin_start_interval_s:
    - 0.8
    - 3
    second_nominal_clearance_at_start_m:
    - 10
    - 50
  fixed_context:
    ego_speed_mps: 25
    first_lead_speed_mps: 22
    second_lead_speed_mps: 20
    first_nominal_clearance_at_start_m: 35
    lane_change_time_scale_s: 1.5
    first_event_start_s: 1
    lane_count: 2
    duration_s: 18
  required_capability:
  - target_switching
  execute_all_catalogue_entries: false
- id: S13
  template_id: fbrt_curved_slow_lead
  implementation: deferred
  priority: P2
  evidence:
  - S3
  parameterization_version: research_v2
  research_bounds:
    curve_radius_m:
    - 250
    - 650
    initial_arc_clearance_m:
    - 15
    - 80
  fixed_context:
    ego_speed_mps: 20
    lead_speed_mps: 15
    lane_count: 2
    duration_s: 20
  required_capability:
  - curved_route_support
  execute_all_catalogue_entries: false
- id: S14
  template_id: fbrt_same_lane_ptw
  implementation: deferred
  priority: P2
  evidence:
  - S3
  - S4
  parameterization_version: research_v2
  research_bounds:
    initial_clearance_m:
    - 10
    - 60
    target_lane_offset_m:
    - -1
    - 1
  fixed_context:
    ego_speed_mps: 13.88888888888889
    lead_speed_mps: 6.944444444444445
    target_length_m: 2.2
    target_width_m: 0.8
    lane_count: 2
    duration_s: 18
  required_capability:
  - declared_ptw_geometry_support
  - low_speed_or_lateral_escape
  execute_all_catalogue_entries: false
```


### 29.1 编译时必须处理的字段映射

- 原 `lane_change_duration_s` 实际用作增益时间尺度时，映射为 `lane_change_time_scale_s`；旧缓存按原语义读取，新日志别名明确，不修改历史运行。
- 原 `static_target_ttc_s` 映射为 `lead_to_static_ttc_start_s`，并检查静止车位置构造是否确实使用同一对车辆。
- 同一 parameterization 内 RBF 归一化按 `research_bounds` 计算；旧历史物理坐标按第28.3节重编码。
- 未实现的 P1/P2 条目可以登记为 `NOT_IMPLEMENTED_NOT_SELECTED`；不影响选定配方的工程交付。被选中的条目必须真实执行，不能用另一个模板结果替代。
- 所选新 profile 改了时长/ramp/固定速度时，建立新的 `context_id`，不能读取相同 scene ID 的旧结果来节省预算。
- 配方B所选 S06/S08 是必要的能力匹配替代；其余 P1/P2 不自动实现和执行。

---

## 30. 增补验收：检查交互是否成立，不添加大量效果门槛

### 30.1 仅增加以下轻量检查

| ID | 检查 | 方式与交付 |
|---|---|---|
| S-E1 | 全部候选有来源、单位、两项active参数、固定上下文与适用能力 | 静态YAML校验；`scenario_catalogue.yaml` |
| S-E2 | 国标框架、地方标准值、NCAP值、论文值、研究范围不混淆 | `scenario_sources.md`，直接使用第23/31节；不新增认证 |
| S-E3 | TTC的对象与定义时刻、净距/中心距和速度单位正确 | 三个代数/几何单元测试；无需仿真 |
| S-E4 | PPO等动作能力被读取，配方在生成银行前确定 | `capability_matrix.csv`、`selected_recipe.json` |
| S-E5 | 切入/切出/刹停/保持/起步等真实阶段可追溯 | 选定模板已有smoke/回放轨迹中提取事件；不额外加每模板几十次 |
| S-E6 | 新银行≤320，整体≤400，未选场景不被自动执行 | 复用统一compute ledger |

这些检查不能要求每个SUT在所有场景都零碰撞，也不要求每类历史数据都同时有通过与失败。新场景的实际效果如实列入第15节既有交付等级。

### 30.2 Codex 执行顺序

1. 保留第8—10节主模型和第11节旧银行实验；不要因补充场景重新设计整个方法。
2. 将第29节目录导出到 `configs/fbrt/scenario_catalogue_v2.yaml`，读本地动作能力并生成 `selected_recipe.json`。这一步不运行驾驶训练。
3. 有新银行则复用对应合同；尚无新银行且PPO不能停车时，采用 `highway_policy_compatible_5`。不要先跑两套配方挑赢家。
4. 只补选中模板的必要执行逻辑和字段。所有测试方法共享参数化、能力适用范围、候选和物理结果。
5. 在既有320+80预算内完成必要执行，输出逐任务结果与来源映射，唯一NoMemory消融离线完成。
6. P1/P2未选条目以“候选已定义、未执行”交付；不在报告中写成实验覆盖。

### 30.3 可直接粘贴的更新指令

> 在 FBRT-Memory 原方案上落实本增补。参考第23节的国家标准定位、深圳地方标准具体交互、Euro NCAP协议和国内外论文，按第24—29节实现候选场景目录、物理参数语义与能力匹配。原始标准值和本项目连续范围分别记录。重点避免：TTC对象/时刻混用、把KP时间尺度当真实换道时长、PPO没有停车动作却强行执行停车回归任务、不同上下文压成同一个二维点、看到方法结果后更换候选池。
>
> 场景候选共14类，不全部执行。已有数据按legacy合同保留；新银行只选一个五类配方，每类16个样本、四个系统，主体最多320次。PPO无法停车时使用目录中的高速策略兼容配方，复用巡航及切入/制动组合，不重新训练。总新增物理调用最多400，所有调试、回放和重试计入。只做既有必要方法比较和唯一NoMemory消融。
>
> 完成可运行实现、实际选用场景的参数和来源、能力矩阵、模式库更新、逐任务发现结果、成本账本与验收报告。是否获胜如实报告；不启动额外实验直到获胜。新场景目录是让任务更合理、更可解释的实施依据，不是强行得到正结果的筛选规则。

---

## 31. 本次增补的来源、定位与获取状态

以下均在2026-09-26检索/核对。PDF“印刷页”与从1开始的文件页不同：S3附录示例印刷31页对应PDF第35页；S4印刷19页对应PDF第21页。部分截图请求失败，本文件不从未看清的图形估计额外数值。

### 国家与地方标准

**[S1] GB/T 41798-2022**《智能网联汽车 自动驾驶功能场地试验方法及要求》。国家标准全文公开系统，现行；2022-10-14发布，2023-05-01实施。获取状态：官方元数据已核对，正文参数表本次未完整取得。  
https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=C3FD7FF23C6D06A9F7459DCD73E68905

**[S2] GB/T 47025-2026**《智能网联汽车 自动驾驶功能仿真试验方法及要求》。国家标准全文公开系统，现行；2026-01-28发布并实施。获取状态：官方元数据已核对，未将未取得的正文要求写成已完成事项。  
https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=F1D96EE9F6E84D109F1AC57BDF7A1412

**[S3] DB4403/T 359.1-2023**《智能网联汽车自动驾驶系统技术要求 第1部分：高速公路及快速路自动驾驶》。深圳市市场监督管理局官方发布与PDF。主要依据：附录C，C.4.3.2、C.4.3.3.1—C.4.3.3.9。获取状态：官方全文文本可读；优先引用明确叙述条件，未完整复刻其中图示路径。  
发布页：https://amr.sz.gov.cn/gkmlpt/content/10/10800/post_10800873.html  
官方PDF：https://amr.sz.gov.cn/attachment/1/1342/1342711/10800873.pdf  
参与编制方解读（辅助，不替代标准）：https://www.castc.net/news/9807.cshtml

### 国外官方测试协议

**[S4] Euro NCAP. Safe Driving—Vehicle Assistance Test & Assessment Protocol, v1.2, July 2026.** 主要依据：Car-to-Car测试矩阵，2.1.3—2.1.5。获取状态：官方全文文本和测试矩阵可读；保留该版本，不与Safety Backup中的另一组TTC混用。  
https://cdn.euroncap.com/cars/assets/Euro_NCAP_Protocol_Safe_Driving_Vehicle_Assistance_v1_2_b6fb486fa6.pdf  
当前目录：https://www.euroncap.com/safe-driving/

**[S5] Euro NCAP. AEB Car-to-Car Test Protocol, v4.3.1, February 2024.** 固定版本复现依据，不称最新。主要依据：8.2.2，尤其8.2.2.3，前车制动速度/间距/减速度组合。获取状态：官方全文及相关页表可读。  
https://cdn.euroncap.com/cars/assets/euro_ncap_aeb_c2c_test_protocol_v431_532926aad1.pdf

**[S6] Euro NCAP. Lane Support Systems Test Protocol, v4.3, December 2023.** 固定版本复现依据。主要依据：7.2.4.4、7.4，同向超越与相对速度；本项目自主换道测试是适配，不复刻强制偏离轨迹。  
https://cdn.euroncap.com/cars/assets/euro_ncap_lss_test_protocol_v43_f2ddd5f6d6.pdf

### 国内外原始论文

**[L1] Feng, S., Feng, Y., Sun, H., Zhang, Y., Liu, H. X.** Testing Scenario Library Generation for Connected and Automated Vehicles: An Adaptive Framework. IEEE T-ITS, 23(2), 1213–1222, 2022. DOI:10.1109/TITS.2020.3023668。本文核对公开预印本v3的V节，不沿用其离散场景总数的算术表述。  
https://arxiv.org/abs/2003.03712  
https://arxiv.org/pdf/2003.03712

**[L2] 李鹏辉、董倩茹、袁赫男、胡文浩、孙巍、谷远利、董春娇。** 面向自动驾驶仿真测试的高覆盖切入场景库生成方法。《中国公路学报》2024,37(7):237–249。DOI:10.19721/j.cnki.1001-7372.2024.07.019。获取状态：期刊目录中的摘要可读，完整数值表未获取；本方案只借鉴运动学参数化和联合参数分布思想。  
https://zgglxb.chd.edu.cn/CN/10.19721/j.cnki.1001-7372.2024.07.019  
摘要目录：https://zgglxb.chd.edu.cn/CN/Y2024/V37/I7

**[L3] Karunakaran, D., Berrio Perez, J. S., Worrall, S.** Generating Edge Cases for Testing Autonomous Vehicles Using Real-World Data. Sensors,2024,24(1):108。DOI:10.3390/s24010108；网页在线发表时间为2023-12-25，卷期为2024。获取状态：出版社全文可读，参考其切入/切出参数相关性，不复制未校验的数据范围。  
https://www.mdpi.com/1424-8220/24/1/108  
作者在文中给出的代码/数据入口：https://github.com/dkarunakaran/concrete_scenario_generation_real_world

陈教授方法依据沿用[R1][R2]，不新增必须复现其所有方法或最新文献的方法比较任务。

### 工程能力依据

**[E1] highway-env `DiscreteMetaAction`。** 未显式配置速度档时，使用`MDPVehicle.DEFAULT_TARGET_SPEEDS`；本地安装版本为运行事实来源。  
https://github.com/Farama-Foundation/HighwayEnv/blob/main/highway_env/envs/common/action.py

**[E2] highway-env `MDPVehicle`。** 本次核对的上游代码默认`np.linspace(20,30,3)`。默认值不是所有驾驶策略的通用能力；Codex应输出本地实际动作空间。  
https://github.com/Farama-Foundation/HighwayEnv/blob/main/highway_env/vehicle/controller.py

**[C11补充] 本项目外部动作配置。** 同一核对提交中的`ExternalCutInEnv`未显式覆盖`target_speeds`，因此必须执行第27节能力读取，不能继续假设PPO具有停车能力。  
https://github.com/SafeDL/META_LEARNING/blob/6af8a502c90235063e593d2d36c8aede3aa027f2/highway_env_benchmark/envs/external_cutin.py

---

**最终执行口径：14类候选已定义；选择5类以内实际执行；范围来自清楚区分的规范锚点与研究设计；优先复用历史失败，接通真实不同驾驶机制；主体320、总量400次上限不变。**
