# FBRT：规范启发的功能场景与历史失效边界回归测试
## 给 Codex 的工程实施与小规模实验任务书

**日期：2026-09-24**  
**代码基线：** `SafeDL/META_LEARNING@bee4fea216d0c0c32371bc17be3f0d958039a3e2`  
**方法工作名：** Failure-Boundary-guided Regression Testing，FBRT  
**本轮范围：** highway-env；已有 IDM 控制器及其可执行局部修改；缓存优先；不训练驾驶策略。  
**状态：** 本文整理当前方法设计与实际实验设置。原 quick、smoke 和分阶段流程不再作为实验配置；正式结果见 `results/method_chains/failure_memory_regression/standard_aligned/core/`，实际入口见第 15 节。

> **本轮直接实现并运行：规范启发的交互场景 → 历史失败与邻近通过样本 → 局部失效边界记忆 → 新版本回归优先级 → 实际回归检测结果。**
>
> 不再把全功能覆盖、完整标准认证、复杂模型胜出或统计显著性设为执行前置条件。先完成一个能运行、能回放、能比较的研究原型。结果局部有效、静态方法最好、某个修改没有回归，均正常交付。

---

## 0. 一页执行摘要

### 0.1 本轮研究问题

同一驾驶系统更新后，如何利用历史失败以及失败附近的成功案例，用较少新版本测试，发现**过去能处理、现在处理不了**的场景？

核心信息不再是“旧版本某场景得分高”，而是：

- 哪些场景构成一片历史失效区域；
- 这片区域与通过区域的交界在哪里；
- 哪些旧通过场景靠近交界；
- 曾经修复过的失败场景是否重新失败。

### 0.2 本轮采用的配置

| 项目 | 决定 |
|---|---|
| 仿真器 | 复用当前 highway-env，不切换 MetaDrive/CARLA |
| 参考系统 | `idm_revision_pilot.py` 中的 `REFERENCE` |
| 目标系统 | `merge_blind06`、`merge_brake2`、`slow_front_brake2` |
| 主功能场景 | 前方车辆切入、单车道前车紧急制动、前车停—保持—起步、前车切出后存在静止车辆 |
| 本轮不包含 | 切入后制动、低速前车跟驰、其他 SUT 与策略训练 |
| FBRT 选择器 | `FBRT-Static`、`FBRT-Adaptive`、`FBRT-RegionBandit` |
| 实际预算 | 每个种子与目标版本 50 个不同场景；报告 @5、@10、@20、@50 |
| 主结果 | 回归检测、首次回归查询位置、回归场景数与碰撞数 |
| 模型与算力 | 几何近邻＋一维离散贝叶斯更新；CPU；不需要 GP 或神经网络 |
| 实际实验 | 3 个种子 × 4 类功能场景；先构建参考档案与目标响应库，再在共同候选上离线比较 |
| 暂不要求 | 完整 FSB/DSB 原论文数值复现、覆盖全部国标场景、每个目标影响多个场景、全面超越所有基线 |

### 0.3 研究范围

本轮只评估 highway-env 上的 IDM 参考版本、三个受控局部修改和四类交互场景。历史 CoRe-Mine、MetaDrive 与其他方法链的结果保留作研究记录，不混入 FBRT 的样本或排名。

---

## 1. 规范怎么用：提供场景骨架，而不是增加一套认证工程

### 1.1 已核验的标准与资料

| 资料 | 本次核验范围 | 本任务的用途 |
|---|---|---|
| **GB/T 41798-2022《智能网联汽车 自动驾驶功能场地试验方法及要求》** | 国家标准全文公开系统显示“现行”，实施日期为 2023-05-01；是 GB/T 推荐性标准，不是“GB 41798”强制性标准 | 提供功能场景分类和场地试验的组织框架 |
| **GB/T 47025-2026《智能网联汽车 自动驾驶功能仿真试验方法及要求》** | 官方记录显示“现行”，发布与实施日期均为 2026-01-28 | 当前任务是仿真，应将其纳入参考清单；本次未核对到全部正文，不编写未验证的条款要求 |
| 中国标准出版社发布的 GB/T 41798 正版试读 | 可核验正式目录：6.4“周边车辆行驶状态识别及响应”、6.5“自动紧急避险”等 | 核对章节框架；不能仅用目录推断未见到的全部子条款 |
| 中汽研对深圳《智能网联汽车自动驾驶系统技术要求 第1部分：高速公路及快速路自动驾驶》的参与编制方解读 | 明确介绍前方切入、切出、目标车辆停—走、切出后静止车辆、单车道前车紧急制动等工况 | 为本轮四类交互场景提供可追溯的具体行为骨架 |
| Chen：Failure-Based Testing；Huang/Chen 等：Search for Boundary | 前者给出失效模式利用思想，后者给出从已知失败寻找区域边界的研究路线 | 将“功能场景”与“失效区域记忆”结合，而不是只重放单个失败点 |

**核验边界：** 本次取得了官方标准状态和正式目录，但没有完整读取 GB/T 41798、GB/T 47025 的所有正文、参数表和图。下面的数值范围是**研究配置**，不是伪装成标准规定的数值。具体交互骨架还参考了上述中汽研参与编制方解读；该解读针对深圳地方标准，不等同于 GB/T 41798 正文。

部分网络转载采用“6.22 前方车辆切入”等编号，与本次看到的正式目录组织不同。本任务不要复制这些未经核对的编号。`standard_mapping.md` 先记录标准名称、已核验大类、工况名称和参考地址；取得正式全文后补充子条款即可，不阻塞开发。

### 1.2 本轮采用两层场景定义

**功能层：** 规定道路、车辆角色、行为阶段和测试功能。例如“前车刹停、停住一段时间、再起步”。

**研究参数层：** 在功能层不变时变化间距、相对速度、制动强度或换道持续时间，形成通过区和失败区。

标准启发我们选取具有工程意义的交互。FBRT 决定这些交互的哪些参数组合值得优先复测。标准本身不提供本任务的回归排序算法。

本轮统一命名为：

> “规范启发／功能对齐的 highway-env 回归测试场景”，而不是“通过 GB/T 41798 或 GB/T 47025 认证”。

不实现视觉交通标志识别、雨雾传感器模型、驾驶人接管、整车认证装备或所有标准试验项目。当前驾驶器不具备的功能无需凑数。

---

## 2. 当前功能场景确有需要修订之处

以下来自 `highway_env_benchmark/envs/cutin_env.py` 及现有外部控制器执行器，不是对仿真实验效果的猜测。

| 当前情况 | 对 FBRT 的影响 | 本轮处理 |
|---|---|---|
| `stop_and_go` 只是有限时长减速，之后恢复速度控制；没有显式刹停与停车保持状态 | 场景名字与实际行为不完全对应；容易只是另一个“前车制动” | 新建完整 `stop_hold_go`，保留旧模式不覆盖 |
| `fast_intrusion`、`cutin_braking` 的 `timing/intensity` 在不同模式含义不同；部分参数同时控制两个物理量 | 几何距离和“边界方向”不好解释 | 新 schema 使用有单位的物理字段；保留 legacy adapter |
| `slow_lead_following` 的 `timing` 不起作用；前车速度还受 `intensity` 改变 | 无效维度会产生虚假距离与多样性 | 新场景只使用实际前车速度，不加入无效参数 |
| 旧环境默认两车道；纵向制动与完整换道驾驶没有明确区分 | 同名功能可能测成不同任务 | 紧急制动固定一车道；其他场景按模板固定两车道，所有版本一致 |
| `passing_cutin` 实际仍是切入，并额外放置一个运动前车 | 不能拿它当作“前车切出、露出静止车辆” | 为 `cutout_static` 新增明确三车逻辑 |
| `cutin_duration` 被用于设置横向控制增益，不保证等于真实完成换道所用时间 | 输入参数与实际触发时刻可能不同 | 同时记录设定值与实际进入／完成车道变化时间 |
| 切入目标初始纵向位置按固定 `CUTIN_START` 作补偿 | `initial_gap` 不总等于真实初始净距或切入净距 | 新模板显式定义初始保险杠净距；不再使用隐含补偿 |
| 原 `episode_result().collision` 来源于任意车辆终止；外部字段 `background_collision` 也包含被 ego 撞到的对方车辆 | 新增三车场景时可能混淆“ego 碰撞”与“仅背景车互撞” | 主事件直接读 ego 碰撞；单独记录背景车互撞，不把“其他车 crashed”一律当无效 |
| 当前最小距离只监视指定 `_cutin_vehicle` | `cutout_static` 的真正危险对象可能是另一辆车 | 新运行器对所有相关目标计算最小净距，并记录碰撞对象 |

**结论：** 不是所有旧场景都无用。切入和制动可复用；最需要修改的是行为阶段不完整、功能重叠和参数语义不清楚。规范可用来修正这些问题，但无需推翻仿真器。

---

## 3. 被测算法：先用能解释版本变化的现有实现

### 3.1 主实验：同一个 IDM 参考系统的三个局部修改

直接复用：

- 参考配置：`method_chains/core_mine/idm_revision_pilot.py::REFERENCE`
- 控制器：`sut_algorithms/highway_env/idm_profiles.py::ProfiledIDMVehicle`
- 局部修改：`method_chains/core_mine/local_fault_idm.py::LocalFaultIDMVehicle`

| 版本 | 已有真实执行逻辑 | 首先观察的功能 |
|---|---|---|
| `idm_ref` | 正常跟驰和制动；维持既定目标车道 | 所有主场景的参考版本 |
| `merge_blind06` | 首次观察到有横向偏移的前车后，短时间按无前车计算加速度 | 切入／切出过程中目标处理 |
| `merge_brake2` | 类似触发后，在短窗口内限制制动指令 | 快速切入、切入后减速 |
| `slow_front_brake2` | 前车速度低于 18 m/s 时限制制动指令 | 低速前车、制动至低速、静止目标 |

注意 `merge_*` 当前触发逻辑检查的是前车横向偏移，并不显式区分“切入”与“切出”；不要在报告中未经验证地写成只影响切入。记录触发轨迹即可。

这些是**受控软件修改版本**。初期可直接用于回归研究，不要求先找到商业产品的真实历史缺陷。修改根据车辆状态触发，不根据 `scenario_id` 或目标真实标签触发。

早期 80 场景 pilot 仅作局部修改的开发背景，不并入本轮历史档案或正式结果；本轮参考标签全部来自当前 480 次参考执行。

### 3.2 不要求每个修改在所有功能场景中失败

同一候选套件提供给所有版本。没有触发故障的场景属于正常对照；一个切入相关修改只在切入场景产生回归，也有研究价值。

主指标按“是否尽早检测到这个修改版本存在回归”统计，不按它必须影响多少功能验收。

### 3.3 当前范围之外

本轮不运行 MOBIL、PPO、FVDM、VI 或 MCTS 版本对，也不训练新驾驶策略。当前结果只支持对 IDM 及这三个受控局部修改的结论。

---

## 4. 新的四类主功能场景

### 4.1 共同设置与参数解释

下列所有数值都是本任务的**首轮研究默认值**，不是标准规定值。

- 道路：直路；长度建议 1000 m，避免长一点的停—走试验驶出道路。
- 物理更新：20 Hz。
- 首轮 IDM 内部控制：20 Hz。
- 不使用降频扩大回归差异；同一版本对共享相同执行周期。
- 首轮使用现有车辆几何和动力学，不修改其制动硬件能力。
- 间距字段统一为指定初始化时刻的**保险杠纵向净距**，单位 m。
- 输入速度统一用 m/s；显示和标准对照时再换算 km/h。
- 对每个模板，首轮只变化两个有效物理参数。高维实验不是前置条件。
- 周围车辆执行固定、可重放的行为阶段；ego 始终闭环运行。配对版本使用相同周围车辆脚本。
- 所有改造单独版本化；不改写旧实验工件。

本轮初始化后短时启动事件属于研究简化，不声称满足规范中所有稳定跟随、触发精度和设备要求。若已有原文可用，可以另保存 nominal anchor；不要把未核实条款硬编码成“国标参数”。

### 4.2 S1：前方车辆切入 `fbrt_cutin`

**功能：** 对进入本车道的目标及时响应。  
**规范来源：** GB/T 41798 的相关功能框架；具体行为骨架见中汽研解读的“前方车辆切入”。  
**车辆：** ego＋相邻车道目标；两车道。

行为：

1. ego 在车道 0；目标在车道 1，且在前方。
2. 目标按固定纵向速度运动。
3. 初始化后 1.0 s 开始向 ego 车道换道。
4. 进入车道后继续保持目标纵向速度。

首轮配置：

| 参数 | 默认／范围 | 是否搜索 |
|---|---|---|
| ego 初速 | 25 m/s | 否 |
| 目标初速 | 20 m/s | 否 |
| 初始净距 | 8–50 m | 是 |
| 设定换道持续时间 | 0.6–3.0 s | 是 |
| 换道开始时刻 | 1.0 s | 否 |
| episode 时长 | 9 s | 否 |

复用旧切入执行器的结构，但直接用物理字段。目标纵向速度使用固定开环控制，不因两次 SUT 运行不同而改变。

记录实际开始换道、车体首次侵入 ego 车道、完全进入车道的时间。`lane_change_duration_s` 是设定值；实测值另存。若末尾尚未换道完成但 ego 已发生碰撞，碰撞仍可被记录，不能以“交互未完成”为由删除。

预期检验：旧通过的短净距／快速切入组合是否因局部修改变成失败。是否真的发生由执行结果决定。

### 4.3 S2：单车道前车紧急制动 `fbrt_lead_emergency_brake`

**功能：** 既有前车突然减速时的跟驰和制动响应。  
**规范来源：** 相关紧急避险功能框架；中汽研原始解读明确给出单车道前车紧急制动工况。  
**车辆：** ego＋同车道前车；一车道。

行为：

1. ego 与前车以相同初始速度行驶。
2. 1.0 s 后前车开始减速。
3. 前车持续减速直至零速，之后保持停车。
4. 无相邻车道可驶入；这一道路拓扑对所有版本固定。

首轮配置：

| 参数 | 默认／范围 | 是否搜索 |
|---|---|---|
| ego 与前车初速 | 20 m/s | 否 |
| 初始净距 | 5–60 m | 是 |
| 前车减速度幅值 | 3–8 m/s² | 是 |
| 制动开始时刻 | 1.0 s | 否 |
| episode 时长 | 12 s | 否 |

不能继续用“短时间减速后恢复巡航”代替该场景。低速到停车的积分必须避免穿越零速后倒车；这是目标脚本数值处理，不是对 ego 额外增加制动保护。

### 4.4 S3：前车停—保持—起步 `fbrt_stop_hold_go`

**功能：** 前车停止、保持停车、重新起步时的连续跟驰响应。  
**规范来源：** 中汽研解读的“目标车辆停—走”。  
**车辆：** ego＋同车道前车；两车道。首轮 IDM 自身没有自主换道，执行纵向跟随。

目标状态机必须明确：

```text
CRUISE → BRAKE_TO_STOP → HOLD_STOP → RESTART → CRUISE
```

首轮配置：

| 参数 | 默认／范围 | 是否搜索 |
|---|---|---|
| ego 与前车初速 | 15 m/s | 否 |
| 初始净距 | 4–35 m | 是 |
| 前车减速度幅值 | 2–5 m/s² | 是 |
| 开始制动 | 1.0 s | 否 |
| 停稳后保持 | 2.0 s | 否 |
| 起步加速度 | 1.5 m/s²，直到恢复初速 | 否 |
| episode 时长 | 24 s | 否 |

时长应覆盖完整背景事件：`event_start + v_lead / deceleration + hold + v_lead / restart_acceleration + 2 s`。当前最慢制动配置对应约 22.5 s，所以采用 24 s，而非在目标尚未重新加速完成时结束。若 ego 提前碰撞，按碰撞正常终止，不要求碰撞后仍继续完成状态机。

首轮 oracle 主看 ego 碰撞；起步延迟与是否恢复跟随只作单独功能记录，不与碰撞合成一个新分数。

接入会换道的驾驶器后，安全换道超越是可记录的正常响应，不能为了让所有方法都跟停而临时禁用它。

**现有 `stop_and_go` 不直接重命名。** 新模板要有真实停车和保持过程；否则输出 `scenario_semantics_error`，修场景，不改评分。

### 4.5 S4：前车切出后存在静止车辆 `fbrt_cutout_static`

**功能：** 跟随对象切出后，对同车道静止目标的处理。  
**规范来源：** 中汽研解读的同名工况。  
**车辆：** ego、行驶前车 VT1、静止前车 VT2；两车道。

行为：

1. ego 和 VT1 同车道行驶，VT2 位于 VT1 前方同车道并保持静止。
2. VT1 在 1.0 s 时切出到相邻空车道。
3. ego 根据自身控制逻辑处理 VT2。
4. VT1 需能按脚本完成避让，不能让它预先撞上 VT2 终止试验。

首轮配置：

| 参数 | 默认／范围 | 是否搜索 |
|---|---|---|
| ego 与 VT1 初速 | 20 m/s | 否 |
| ego—VT1 初始净距 | 8–35 m | 是 |
| VT1 开始切出时到 VT2 的名义 TTC | 1.5–4.0 s | 是 |
| VT1 换道设定持续时间 | 0.8 s | 否 |
| VT1 切出时刻 | 1.0 s | 否 |
| episode 时长 | 10 s | 否 |

根据目标车辆脚本和尺寸一次性计算初始位置；运行过程中不能平移目标来“补足设定 TTC”。保存触发时真实净距和 TTC。

当前 highway-env 使用状态信息，本轮不声称模拟了真实传感器遮挡。VT2 从一开始存在，是否被 controller 选为相关前车由已有邻车／观测逻辑决定，不给目标版本单独隐藏 VT2。

背景轨迹有效性根据脚本和保存的轨迹检查；不能根据目标是否碰撞选择性排除样本。

---

## 5. 保留但不强行“标准化”的两个场景

| 场景 | 本轮定位 | 处理 |
|---|---|---|
| `cutin_braking` | 切入与制动的组合研究扩展 | 保留；后续可增加独立 `brake_after_merge_delay_s`，但不把旧 timing 直接冒充此参数 |
| `slow_lead_following` | 低速前车处理的机制诊断 | 保留；输入为实际前车速度，去除无效 timing 维度 |

它们不属于本轮正式候选池。若某主模板没有历史边界，保留该模板结果并使用已实现的历史裕度回退。

不新增交叉口、环岛、行人、视觉识别场景。需要完整 MOBIL/PPO 的后续扩展另开配置，不让首轮发散。

---

## 6. 工程接口：少改旧代码，明确新增部分

### 6.1 当前文件组织

```text
highway_env_benchmark/envs/
    fbrt_scenarios.py
    fbrt_env.py
    fbrt_scripted_vehicle.py

method_chains/failure_memory_regression/
    boundary_memory.py
    boundary_shift.py
    selectors.py
    experiment.py
    report.py
    tests/test_contract.py

results/method_chains/failure_memory_regression/standard_aligned/core/
```

实验复用 `method_chains/core_mine/idm_revision_pilot.py` 的 IDM 参考配置，以及 `local_fault_idm.py` 的受控修改实现；主链没有复制仿真环境或控制器。

### 6.2 场景数据结构

```python
@dataclass(frozen=True)
class FBRTScenario:
    scenario_id: str
    template_id: str
    initial_clearance_m: float
    lane_change_duration_s: float | None = None
    lead_deceleration_mps2: float | None = None
    static_target_ttc_s: float | None = None
```

车道数、初速度和时长由模板确定，不重复存进每条场景记录。`ACTIVE_PARAMETERS` 与 `BOUNDS` 固定定义可搜索的两个物理参数。

### 6.3 时间、动作和车辆状态

- 原始 `CutInEnv` 的 profile ego 通过 `road.act()` 自主控制，其忽略外部 action 属于该旧执行路径；不要不分路径地“修复”整个旧环境。
- 本轮主实验都是 profile IDM，优先复用其现有稳定执行路径。
- 停车和起步状态机只修改背景车；ego 是否刹停由 SUT 决定。
- 实现所有相关车辆的轨迹记录，而不是只保存一个 lead。
- 同一版本对的背景脚本、参数、种子、物理频率完全一致。

---

## 7. Oracle 与历史档案：只保留几项必要定义

### 7.1 主事件

```text
reference_pass(x):
    参考执行有效，观察时长完成，没有 ego 碰撞

regression(x):
    reference_pass(x) AND 新版本在相同场景发生 ego 碰撞
```

这里的“通过”是**本研究的无碰撞参考通过**，不等于通过全部国标要求。近失效单独存储，不用它把全部边界附近参考样本排除。

保留四类配对结果：

- reference pass / target pass；
- reference pass / target collision；
- reference collision / target collision；
- reference collision / target pass。

后两类用于历史与修复分析，不计入新回归。

### 7.2 多车碰撞的最小处理

`ego_collision` 直接取 ego 状态，并记录时间和对象。

`other_vehicle_crashed` 可以保留为原始字段，但不等于“仅背景车辆互撞”。ego 撞上 VT2 时，VT2 也可能 crashed；不能因此删除这个有效 ego 回归。

最好记录碰撞双方 ID；暂时没有 pair 日志时，至少将“ego 是否碰撞”与“无 ego 碰撞但背景导致提前终止”分开。对后一类记 `inconclusive_background_termination`，不填成安全。

### 7.3 记录格式

正式档案记录场景参数、种子、构建名、执行契约和实测响应，包括 `ego_collision`、碰撞对象与时刻、`completed`、`semantic_valid`、`min_ttc`、`min_clearance` 及事件日志。回放轨迹单独写入结果目录。当前数据契约使用场景 ID 和参数逐项匹配，不依赖输入或代码哈希。

评分器不得读取未查询的目标标签、目标故障名字或触发计数。触发计数可供事后工程诊断。

同一动作空间和同一参数维度不代表缓存可互换。模板、几何、版本、更新频率改变之后，旧缓存只能作为历史背景，不能直接充当新模板的结果。

---

## 8. 应用 Chen 思想：把一次失败变成局部边界记忆

### 8.1 本轮采用的具体转化

Chen 的 Failure-Based Testing 强调利用失效模式信息；Search for Boundary 研究从已知失败寻找靠近失效区域边缘的样本。

本轮实现 **SB-inspired failure/pass bracketing**：

```text
历史失败点
    + 同类交互中的附近通过点
    → 通过／失败交界的局部区间
    → 旧安全侧候选的回归优先级
```

这是借鉴其思想的自动驾驶回归适配，不把下述近邻／二分实现声称为原论文 FSB/DSB 的逐行复现。算法新意候选落在“边界记忆如何跨版本服务回归”，不是给二分搜索换名字。

### 8.2 不再把历史失败随着 eligibility 一起丢掉

历史档案存所有 reference/historical 结果，既有失败也有通过。

新版本可执行候选限制为最近 reference 通过的样本。这个限制不影响模型使用历史失败端点。

若有更早版本失败、reference 修复后通过的记录，则场景进入 `repaired_case` 清单。没有真实的多版本修复记录时，该清单为空，不自动制造“过去修复过”的故事。

本轮参考档案包含 480 次执行，其中 160 次 ego 碰撞；形成 67 组失败—通过边界配对。`stop_hold_go` 模板没有参考碰撞，因此该模板按历史裕度回退。目标修改的碰撞不替代参考版本失败端点。

### 8.3 近邻失败—通过对

每个 `(reference_version, template_id, topology)` 独立处理：

1. 按声明范围将两个有效物理参数归一化到 [0,1]。
2. 失败集合取 `ego_collision=True`；通过集合取实际 reference 通过。
3. 为每个失败寻找最近通过点，按配对距离排序。
4. 每模板最多保留 8 个代表性配对；优先取不同失败端点和不同位置。
5. 局部近似中点只称为 `boundary_estimate`，不声称中点已经执行或是真实边界。
6. 某模板没有异标签对时，继续用历史裕度／ART 完成其任务，记录 `boundary_missing`，不停止整个 experiment。

### 8.4 历史补点

每个种子和功能模板先运行 32 个 Sobol 点与 4 个预设探针，再根据参考结果选 4 个中点。存在失败—通过局部配对时，中点沿该配对插值；没有配对时使用预设范围内的补充点。补点的真实参考结果决定标签，所有实际点都保留在参考档案中。

这些有限补点只增加局部信息，不保证恢复全部失效边界，也不假定边界全局单调或失效区域凸。

---

## 9. 方法一：FBRT-Static

### 9.1 几何表示

对一个配对的归一化端点 `f`（失败）与 `s`（通过）：

```text
b = (f + s)/2
n = (s - f) / ||s - f||
w = ||s - f||
```

`n` 是局部“失败→通过方向”，**不是已经证明的真实法向或梯度**。

对于同模板候选 `z`：

```text
u = nᵀ(z - b)
v = ||(z - b) - u*n||
h_n = max(w, 0.03)
h_t = max(2*w, 0.10)
score_p(z) = exp(-abs(u)/h_n - v²/(2*h_t²))
```

超过局部半径 `max(3*w, 0.25)` 的配对不外推。多个可用配对取最大分数，并固定对应的 `patch_id`。没有邻近配对的候选由回退分支负责。

参考通过以真实记录为准，不因 `u<0` 就擅自改成失败或删掉候选。

### 9.2 选择规则

- 在尚未查询的 reference-pass 候选中，选分数最高者。
- 距离和分数相同按场景 ID 打破平局，不使用未查询的目标结果。
- 第 10、20 次查询留给全局 ART 式 maximin 探索；同样计入总预算。
- 不设置强制前 10 次覆盖全部 mode 的支持集。
- 不加入 4×4 碰撞格子惩罚、复杂 GP、多模型权重或长时域规划。

如果静态几何排序已有收益，它就是本轮可保留的 FBRT 版本，不必须升级才算完成。

---

## 10. 方法二：FBRT-Adaptive

### 10.1 要更新的是局部边界移动，而不是完整风险函数

每个 patch 用一个标量 `delta` 表示边界沿旧安全方向的位置变化。

在局部范围内采用：

```text
P(collision | z, delta)
  = eta + (1 - 2*eta) * sigmoid((delta - u_p(z))/tau_p)
```

研究默认值：

```text
delta_grid = linspace(-0.30, 0.30, 81)
prior ∝ exp(-0.5*(delta/0.10)^2)
eta = 0.02
tau_p = max(w_p/2, 0.02)
```

单位为归一化场景坐标。`eta` 是局部模型失配项，不是测得的传感器噪声。没有历史版本位移数据时使用弱先验，不声称已经元学习到它。

### 10.2 更新与选择

每付费执行一次候选，只更新该候选固定归属的一个 patch：

```python
log_q += y * log(p_grid) + (1-y) * log1p(-p_grid)
log_q -= logsumexp(log_q)
```

其中 `y` 为真实 ego 碰撞标签。不要把一个观测重复更新多个 patch 并当成独立证据。

下一候选的分数：

```text
posterior_collision(z)
    = sum_delta q(delta) * P(collision | z, delta)

adaptive_score(z)
    = posterior_collision(z) * exp(-v_p(z)^2/(2*h_t^2))
```

同样保留第 10、20 次 ART 查询。无局部关联的 ART 结果不强行更新远处边界。

目标执行出现工程异常、背景提前终止时不做通过标签更新；执行消耗如实记入物理成本。修复工程异常后另生成可用记录。

### 10.3 统一回退

某模板没有 reference 失败时，该模板使用历史安全裕度排序；历史裕度也无差异时使用 ART/固定随机顺序。输出选择原因 `boundary / adaptive_boundary / fallback_history / global_art`。

所有任务依然运行完并报告。回退发生过不等于整个方法失败。

### 10.4 简要理论动机

设局部旧、新版本带符号安全裕度满足 `|m_new(x)-m_ref(x)| <= epsilon`，正值代表通过。若 x 是新回归，则：

```text
m_ref(x) > 0, m_new(x) <= 0
⇒ 0 < m_ref(x) <= epsilon.
```

因此，小局部变化引起的回归可能集中于旧安全边缘。这个推导是条件性的动机，不是对当前全部控制器已验证的保证；新失效岛由全局探索保留发现机会。

---

## 11. 对照：固定共同候选，直接看历史失败信息是否有用

第一轮在同一个完整响应缓存上运行：

| 方法 | 作用 |
|---|---|
| `Random` | 不使用历史排序，只共享 reference-pass 候选 |
| `ART-Maximin` | 利用输入空间分散性，但不使用失败边界 |
| `HistoryMargin` | 参考版本 TTC／净距裕度排序 |
| `FailureDistance` | 只计算到历史失败点的距离 |
| `HistoryRank-UCB` | 当前已有“历史排序＋功能级 UCB”思想 |
| `FBRT-Static` | 失败—通过局部关系 |
| `FBRT-Adaptive` | 在同一边界记忆上利用目标反馈 |

当前 `ModeUCB1` 模式内部使用历史均值排序，不能称为完全 target-only。主表明确写 `HistoryRank-UCB`。

`HistoryMargin` 默认参考当前 response/TTC 分数；无穷 TTC 单独处理，平分按净距和场景 ID 排序。不要为使 FBRT 好看而故意削弱这个基线。

`FailureDistance` 与 FBRT 使用同一归一化、同一历史失败集合。每种算法都获得同一场景池以及相同历史档案。所有方法可以使用 reference 边界加密后的通过点；禁止只给 FBRT 提供一套更优候选。

随机方法运行 20 个低成本重放顺序；其他方法各运行一次。所有方法复用同一目标响应库。

---

## 12. 实际实验设置：一次物理运行，共享结果库比较

### 12.1 物理场景与运行数量

- 3 个随机种子：`4179801, 4179802, 4179803`。
- 4 类功能场景；每个种子、每类场景使用 32 个 Sobol 点、4 个预设探针，以及根据参考结果选出的 4 个补点。
- 因此参考版本共运行 `3 × 4 × 40 = 480` 个 episode。
- 其中 320 个场景是参考版本完整通过的共同候选；3 个修改版本各在这 320 个场景上运行，共 960 个目标 episode。
- 主车物理更新和 IDM 控制频率均为 20 Hz；环境、SUT、场景模板与故障定义见对应源码和结果清单。

参考结果决定边界补点，因此这些场景和缓存用于本轮方法开发与离线比较，不是独立留出的确认集。

### 12.2 选例预算与方法比较

- 每项“随机种子 × 目标版本”任务预算为 50 个不同场景，检查 @5、@10、@20、@50。
- 比较 Random（20 次随机顺序）以及其余 7 种方法（各一次）：ART-Maximin、HistoryMargin、FailureDistance、HistoryRank-UCB、FBRT-Static、FBRT-Adaptive、FBRT-RegionBandit。
- 共 9 项任务、243 次 campaign、12,150 次逻辑查询；所有方法使用相同候选池和同一目标响应库。
- 选例比较通过 `TargetOracle.query(scenario_id)` 逐项揭示目标结果。离线重放不增加物理 episode；逻辑查询次数与物理仿真成本分开记录。

正式结果保存在 `results/method_chains/failure_memory_regression/standard_aligned/core/`。候选和逐次查询记录分别见 `candidate_pool.csv`、`queries.csv`；完整输入、响应和成本记录见同目录下的档案、目标响应库与 `compute_ledger.json`。

---

## 13. 工程排障与结论边界

### 13.1 只保留必要的工程条件

1. 车辆实际按场景行为运行，参数影响能在轨迹中看到。
2. 参考／目标比较的是同一物理场景。
3. 背景车互撞、载入错误和缺失结果不冒充 ego 回归。
4. 选择器不提前读取未查询的目标结果。
5. 实际执行、缓存使用和逻辑预算可追溯。

这五项保障结果能用，不是新增论文验收门槛。

### 13.2 明确取消的门槛

- 不要求每个故障在两个或更多功能场景失效。
- 不要求全部模板都存在旧失败或新回归。
- 不要求方法显著优于每个基线。
- 不要求 Bayesian 更新优于静态边界排序。
- 不要求碰撞总数、多样性、覆盖和速度全部同时提升。
- 不要求原始 SUT 完全通过国标或复杂场景才允许测试。
- 不要求重新训练合格 PPO/SAC 才能开始。
- 不要求先证明低秩、完整失效区域或通用跨 SUT 泛化。
- 不要求对每个已缓存 episode 再进行一遍物理确认。

---

## 14. 指标与输出：先回答“多早发现更新出问题”

### 14.1 主指标

- **RegressionDetected@B：** 每个目标版本在前 B 次查询是否发现至少一个 reference-pass / target-collision。
- **FirstRegressionRank：** 首次回归在序列中的位置；未发现记为右删失，不能填 0。
- **RegressionCollisionCount@B：** 前 B 次真实回归碰撞数。

报告 @5、@10、@20、@50；参考通过候选不足 B 时，使用实际数量并标注，不能用补齐的假样本。

在当前完整候选银行中没有回归的版本单列 `no_observed_regression_in_pool`。检测率同时给“全部受测版本”和“银行中确有回归的版本”分母，避免隐藏负例。场景随机种子不算独立的软件缺陷。

### 14.2 辅助信息

近失效、停车／起步行为、最小净距、边界来源、故障实际触发频次、推理时间、物理步数、历史构建开销。

4×4 网格数不再是主验收指标。一个局部回归有多次碰撞，不等于多个独立软件 bug。

### 14.3 最小图表与回放

- 每功能一个二维 reference 通过／失败图，标出边界配对；实测点和估计边界区分。
- 每个修改版本一张首次发现／累计发现曲线。
- 至少挑选一个**实际存在**的 reference 通过、target 碰撞场景做配对回放；没有时如实报告没有，不用示意图代替结果。
- 有停—走模板就输出速度—时间曲线，确认真的停车和保持；有切出模板就输出三车位置轨迹。

颜色非必要；生成图片不是 Codex 开始实现方法的前置条件。

---

## 15. 当前运行入口

离线重放已保存的目标响应库，不运行新仿真：

```powershell
conda run -n metadrive python -m method_chains.failure_memory_regression.experiment --replay-measured-bank
conda run -n metadrive python -m method_chains.failure_memory_regression.report --reuse-replay
```

重新生成物理结果时运行同一个 `experiment` 模块，不带 `--replay-measured-bank`；它会按 `--workers` 执行缺失的 episode，并将结果写到正式结果目录。预算、种子和场景范围固定在脚本常量中。

---

## 16. 正式结果文件

```text
results/method_chains/failure_memory_regression/standard_aligned/
    standard_mapping.md
    baseline_manifest.json
    scenario_contracts.json
    revision.md
    core/
        reference_archive.csv
        boundary_memory.csv
        candidate_pool.csv
        target_response_bank.csv
        queries.csv
        summary_by_revision.csv
        summary_by_template.csv
        validation.json
        compute_ledger.json
        report.md
        figures/
        replay/
```

`standard_mapping.md` 至少包含：

```text
template_id
functional_name
reference_standard
verified_parent_section
specific_clause_status
source_url
borrowed_behavior_skeleton
research_parameter_ranges
deviations
```

`compute_ledger.json` 记录：

```text
new_reference_episodes
new_target_episodes
physical_reference_episodes_used
physical_target_episodes_used
physical_cache_records_total
replay_episodes
cache_hits
logical_queries_per_campaign
wall_clock_seconds
```

`report.md` 用几页即可，回答：

1. 修订后的场景实际测了什么；
2. 历史失败怎样进入边界记忆；
3. 哪些版本存在回归，方法多早发现；
4. 收益来自失败点、边界关系，还是更新；
5. 最终应保留 Static 还是 Adaptive；
6. 是否值得新增 MOBIL/PPO 扩展。

**工程完成不等于宣称研究已经取得普遍胜利。即使只有一两个功能存在明确收益，也完整交付，继续围绕该证据收敛，而不是增加层层验收。**

---

## 17. 参考资料与固定源码地址

### 标准与场景来源

[S1] 国家标准全文公开系统：GB/T 41798-2022，官方状态与基本信息。  
https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=C3FD7FF23C6D06A9F7459DCD73E68905

[S2] 国家标准全文公开系统：GB/T 47025-2026，官方状态与基本信息。  
https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=F1D96EE9F6E84D109F1AC57BDF7A1412

[S3] 中国标准出版社上传的 GB/T 41798 正版试读（用于核验正式目录，不采用网页自行生成的“标准解读”作为条文）。  
https://www.renrendoc.com/paper/240760461.html

[S4] 中汽研科技，参与编制方对深圳《智能网联汽车自动驾驶系统技术要求 第1部分：高速公路及快速路自动驾驶》的原始解读。重点见“周边车辆行驶状态识别与响应”部分。  
https://www.castc.net/news/9807.cshtml

**说明：** 本任务没有复制整份标准或其参数表。取得正式文本后按本任务结构增补来源字段；取不到完整文本也可按研究模板先完成方法。

### Failure-Based Testing 与边界搜索

[F1] Tsong Yueh Chen. *Failure-Based Testing*. ICST 2026 Keynote.  
https://conf.researchr.org/details/icst-2026/icst-2026-keynote/3/Failure-Based-Testing

[F2] Rubing Huang, Weifeng Sun, Tsong Yueh Chen, Sebastian Ng, Jinfu Chen. *Identification of Failure Regions for Programs with Numeric Inputs*. 公开稿 2020。  
https://arxiv.org/abs/2007.15231

F1 提供 failure pattern 信息利用的总思想，F2 提供 SB/FSB/DSB 路线。本任务新增的是功能场景约束下的历史边界记忆与版本回归使用方式；没有核实到可直接运行的原作者 FSB/DSB artifact，不虚构安装命令。

### 当前代码

[C0] 固定基线。  
https://github.com/SafeDL/META_LEARNING/tree/bee4fea216d0c0c32371bc17be3f0d958039a3e2

[C1] 现有场景、目标脚本和评估。  
https://github.com/SafeDL/META_LEARNING/blob/bee4fea216d0c0c32371bc17be3f0d958039a3e2/highway_env_benchmark/envs/cutin_env.py

[C2] 外部控制器环境。  
https://github.com/SafeDL/META_LEARNING/blob/bee4fea216d0c0c32371bc17be3f0d958039a3e2/highway_env_benchmark/envs/external_cutin.py

[C3] 参考控制器与已有 80 场景。  
https://github.com/SafeDL/META_LEARNING/blob/bee4fea216d0c0c32371bc17be3f0d958039a3e2/method_chains/core_mine/idm_revision_pilot.py

[C4] 三种局部修改。  
https://github.com/SafeDL/META_LEARNING/blob/bee4fea216d0c0c32371bc17be3f0d958039a3e2/method_chains/core_mine/local_fault_idm.py

[C5] 已有局部修改结果，不作为本轮新结果。  
https://github.com/SafeDL/META_LEARNING/blob/bee4fea216d0c0c32371bc17be3f0d958039a3e2/results/method_chains/core_mine/studies/local_fault_pilot/summary.json

[C6] 当前历史排序＋UCB。  
https://github.com/SafeDL/META_LEARNING/blob/bee4fea216d0c0c32371bc17be3f0d958039a3e2/method_chains/core_mine/mode_label_fresh_confirmation.py

[C7] 已有 IDM/FVDM 实现。  
https://github.com/SafeDL/META_LEARNING/blob/bee4fea216d0c0c32371bc17be3f0d958039a3e2/sut_algorithms/highway_env/idm_profiles.py

[C8] 已有固定 PPO。  
https://github.com/SafeDL/META_LEARNING/blob/bee4fea216d0c0c32371bc17be3f0d958039a3e2/sut_algorithms/highway_env/ppo_ece.py

### 既有项目材料的定位

用户提供的《文献调研、问题动机与研究空白》《少样本自适应漏洞场景挖掘技术方案》《竞争性定位与 DIVA-Mine 研究建议》是早期跨 SUT 场景挖掘方案，不是标准原文或 Chen 方法原文。本轮沿用其历史档案和低算力动机，但任务改为明确的版本回归，不继续执行其低秩/K-shot/逐门验证路线。

---

## 原始任务请求（已完成，仅留作研究任务来源）

请基于当前工作树核对本文件指定的 `bee4fea` 基线，保留已有变更与旧研究结果。在 highway-env 上实现四类规范启发的车辆交互模板，复用现有 IDM reference 和三个局部修改，构建包含历史碰撞与通过记录的边界记忆，实现 FBRT-Static、FBRT-Adaptive 及共同候选上的对照。先完整运行 quick，报告 @5/@10/@20 的回归检测与实际成本，再做必要修订和 core。不要重训驾驶策略，不要求每个修改跨功能失效，不把统计显著性或 Adaptive 胜过 Static 作为交付条件；也不要只提交设计文档而不运行方法。所有场景、边界与目标标签必须来自实际执行或契约一致的缓存，未运行的内容明确列为未运行。
