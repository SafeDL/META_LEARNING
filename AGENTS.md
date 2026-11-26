# 多源历史响应迁移与联合反馈自动驾驶失效测试

## 一、核心研究想法

自动驾驶系统（ADS）的安全验证需要在有限闭环仿真预算内尽早暴露真实碰撞。对缺乏测试反馈的新 ADS，从零探索候选场景空间面临冷启动成本；同类交通场景中已有多个历史 ADS 的闭环测试记录可以提供高风险交互区域线索，但不同控制架构、参数配置或策略迭代会改变系统的失效边界。直接照搬历史风险排序或将多源数据混合拟合单一模型，容易在目标响应发生偏移的区域引发负迁移。

本项目提出 **多源历史响应迁移与联合反馈适配方法（Historical Response Transfer with Joint Feedback）**，真实失效由仿真引擎原生碰撞事件判定，轨迹派生的归一化责任敏感安全（RSS）距离裕度作为同次执行伴随产出的连续辅助响应。方法围绕四个递进环节构建：

| 核心环节 | 算法设计思路 |
|---|---|
| **1. 分来源联合风险先验构建** | 采用共享–专属双输出深度核高斯过程（`SharedPrivateKernel`）联合建模二值碰撞（Bernoulli）与连续 RSS 裕度（Gaussian），通过坐标严格互斥的逐源留出顺序元训练学习共享特征表征，并独立保留各历史 ADS 的候选内联合响应后验 |
| **2. 少样本目标适配与空间局部可信度选例** | 冻结共享深度表征，每次查询后以六标量十步轻量拟合目标残差核（先 RSS 高斯条件化、后碰撞 Laplace 推断）；结合指数遗忘、核平滑局部似然残差与目标分支保底下限（`Tempered Local Reliability`）动态更新各历史来源与两条目标分支权重，在借力共性危险区加速的同时消除跨架构负迁移 |
| **3. 同谱系策略更新的高召回复测** | 将联合响应迁移与在线适配无缝用于学习型策略多轮修复后的固定套件回归复测，优先定位残留高危场景以更少查询达到 90% 碰撞召回（$Q_{90}$），并区分总体高召回与相对前代新增碰撞的早期发现 |
| **4. 偏差感知归档与门控持续演化** | 任务完成后保留随机初始化前缀、按查询前碰撞概率对主动选例执行倾向去偏加权并锚定基线碰撞率，按有效样本量（ESS）配权归档；通过 L2-SP 锚定轻量微调与零泄漏留一来源门控（LOSO Gate）跨任务更新共享模型，改善后续任务的零反馈先验与早期发现效率并防止表征退化 |

---

## 二、工程目录组成

| 目录 / 文件 | 核心职责 |
|---|---|
| [docs/](docs/) | [正式实验协议](docs/experiment_plan.md)、[代码与目录维护规范](docs/style.md)及[参考论文](docs/references/) |
| [simulator/](simulator/README.md) | 闭环交通仿真环境。[simulator/highway/](simulator/highway/README.md) 实现 HighwayEnv 原生 20 Hz 物理／5 Hz 控制闭环环境、可见观测遮挡与策略修复课程；[simulator/metadrive/](simulator/metadrive/README.md) 预留独立跨平台场景搜索实现 |
| [controllers/](controllers/README.md) | 被测 ADS 控制器与统一可见状态决策接口，包含 8 个历史来源与 12 个目标 ADS（规则、优化、搜索、离散强化学习 `PPO`/`Continual DQN`，以及跨架构 `OnlineMC PPO`、`Social Attention DDQN`、`Decision Transformer` 与连续动作 `Continuous SAC`） |
| [testing_methods/](testing_methods/README.md) | 有限候选库下的场景选择算法。[testing_methods/history_transfer/](testing_methods/history_transfer/README.md) 实现本文方法各核心模块；[testing_methods/baselines/](testing_methods/baselines/README.md) 实现 `Target-only GP`、`Pooled GP`、`Random`、`ScenarioFuzz`、配对系统主动评估及 `DETOUR`、`Birchler`、`SDC-Scissor`、`SPECTRE` 回归基线 |
| [experiments/](experiments/README.md) | 场景采样、物理执行库采集、受限付费查询账本、五大正式实验（E01–E05）运行调度、独立复算审计与 600 dpi 科研绘图入口 |
| [results/](results/README.md)、[figures/](figures/) | 封存物理执行库、模型权重、付费账本与复算报表（统一 12-ADS 主结果汇总于 [results/lifelong_transfer/](results/lifelong_transfer/README.md)），以及按研究主题归档的正式 PNG 插图与轨迹回放 GIF |
| [tests/](tests/README.md) | 核函数推断、权限隔离、坐标互斥、并行恢复与配对统计的单元测试与回归检查 |

---

## 三、长期执行与维护约定

1. **物理一致性与坐标严格隔离**：HighwayEnv 统一使用官方环境生命周期、原生车辆动力学与 `ego.crashed` 碰撞判定，物理更新 20 Hz、控制决策 5 Hz；训练、能力验收、数据采集与回放共用同一时钟。四类功能场景按声明物理单位在整数矩形域内不放回采样，先排除全部已查看开发坐标，再按数值坐标严格隔离历史库（每场景 8 来源 × 500 点）与目标库（每场景 12 目标 × 2,000 点）。
2. **反馈权限与共同前缀配对**：各功能场景独立建模与测试。完整目标候选库真值仅由评估端持有，严禁参与训练、调参或选例；选择器子进程仅通过 `QueryLedger` 获取公开候选参数与当前已付费反馈。所有对比方法在同一任务上共享相同的候选库与固定种子共同初始化前缀。
3. **固定可比配置与硬件调度**：`Target-only GP`、`Pooled historical GP` 与本文方法在线分支固定采用 float64 精度与六标量十步适配约定（`Target-only GP` 与本文原始特征目标分支共用相同初始化与学习率 `0.01`），不在正式测试坐标上搜参或增加步数。常规在线适配使用 CPU 单线程多进程并行，大规模 `Pooled GP` 使用单卡 CUDA 串行；等效仿真时长、算法耗时、CPU 主存与 GPU 显存分别统计。
4. **实证边界与图表规范**：少反馈风险预测（E01）、固定预算主动发现（E02）、权重与校准消融（E03）、同谱系固定套件回归（E04）与跨任务持续演化（E05）分别依据各自配对统计与 ADS 整块 bootstrap 95% 区间下结论，正负结果与适用边界如实报告。正式图表统一将本文方法标为 `Ours method`，图例置底、保留指标方向标记、不加图内标题。
5. **单一版本命名与工程整洁**：本项目仅维护本文件作为根目录宏观文档，各模块安装与命令维护在对应子目录 `README.md`。文件与文件夹按研究内容或用途命名，仅保留单一当前有效版本，不使用版本号、日期或实验编号命名副本。确认无用的临时日志、试跑缓存、废弃分支与多余脚本直接删除，不为清理过程新增说明文件或报告。
6. **实验暂停纪律**：用户暂停实验期间，所有物理仿真采集、模型训练与正式评估保持暂停，直至用户明确授权恢复；代码维护、文档更新与只读单元测试不代表恢复实验。
