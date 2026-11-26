# 实验运行与结果复算

本目录负责四类 HighwayEnv 交通场景的整数域采样、物理执行库采集、受限付费查询账本管理、五大正式实验（E01–E05）运行调度、独立审计复算以及 600 dpi 科研插图与轨迹 GIF 生成。[统一实验计划](../docs/experiment_plan.md) · [12-ADS 统一结果](../results/lifelong_transfer/README.md)

---

## 一、模块与子包组织

| 位置 | 职责 |
|---|---|
| 统一 12-ADS 流水线 | [阶段调度](lifelong/__main__.py)、[E01–E03 任务矩阵](lifelong_transfer.py)、[E04 版本复测](regression/lifelong_replay.py)、[E05 持续演化](accumulation/lifelong.py)、[统计汇总](lifelong/report.py)、[输入封存](lifelong_seals.py) |
| 场景与响应 | [场景定义与采样](scenarios.py)、[正式坐标域](formal_scenarios.py)、[参数域](domain.py)、[风险量](domain_risk.py)、[RSS 裕度](rss.py)、[数据结构](schema.py) |
| 执行与付费查询 | [执行协议](protocol.py)、[执行库](executions.py)、[原生事件](execution_events.py)、[公共前缀](initialization.py)、[查询权限](query.py)、[断点恢复](resume.py)、[存储](storage.py)、[并行调度](parallel.py)、[资源开销](costs.py) |
| 驾驶准入与数据来源 | [能力验收](qualification.py)、[来源登记](source_registration.py)、[响应归档](historical_responses.py)、[迁移准入](transfer_admission.py)、[历史库审计](audit_transfer_history.py)、[目标库审计](audit_transfer_targets.py) |
| 原八目标队列 | [阶段入口](history_transfer_experiment.py)、[历史建模](history.py)、[基线分派](baselines.py)、[方法契约](method_contracts.py)、[固定适配参数](transfer_settings.py)、[少反馈预测](k_shot.py)、[批次](transfer_batches.py)、[任务执行](transfer_execution.py) |
| 复算与配对统计 | [记录汇集](collect_transfer_records.py)、[记录审计](record_audit.py)、[逐查询预测复算](transfer_prediction_audit.py)、[参考推断](inference_reference.py)、[失效指标](failure_metrics.py)、[配对统计](paired_statistics.py)、[队列统计汇总](history_transfer_summary.py)、[队列报表](report_history_transfer.py)、[正式结果路径](paper_results.py) |
| `public_ads/` 跨架构 ADS 准入与评测 | [prepare.py](public_ads/prepare.py)、[train.py](public_ads/train.py)、[qualify.py](public_ads/qualify.py)、[checks.py](public_ads/checks.py)、[evaluate_e01.py](public_ads/evaluate_e01.py)、[extend.py](public_ads/extend.py)、[extension_report.py](public_ads/extension_report.py)、[mechanism.py](public_ads/mechanism.py)、[accumulation.py](public_ads/accumulation.py)、[attention_repair.py](public_ads/attention_repair.py)、[attention_regression.py](public_ads/attention_regression.py)、[continuous_sac.py](public_ads/continuous_sac.py)、[collect_continuous_sac.py](public_ads/collect_continuous_sac.py)、[evaluate_continuous_sac.py](public_ads/evaluate_continuous_sac.py) |
| `budget/` E02 固定预算发现 | [protocol.py](budget/protocol.py)、[prepare.py](budget/prepare.py)、[selectors.py](budget/selectors.py)、[execution.py](budget/execution.py)、[audit.py](budget/audit.py)、[run.py](budget/run.py)、[report.py](budget/report.py) |
| `weight_ablation/` E03 权重与校准消融 | [run.py](weight_ablation/run.py)、[report.py](weight_ablation/report.py)、[effectiveness.py](weight_ablation/effectiveness.py)、[rss_ablation.py](weight_ablation/rss_ablation.py) |
| `accumulation/` E05 跨任务持续演化 | [sequence.py](accumulation/sequence.py)、[run.py](accumulation/run.py)、[report.py](accumulation/report.py)、[lifelong.py](accumulation/lifelong.py) |
| `regression/` E04 同谱系固定套件回归 | [preflight.py](regression/preflight.py)、[protocol.py](regression/protocol.py)、[execution.py](regression/execution.py)、[queries.py](regression/queries.py)、[repair.py](regression/repair.py)、[audit.py](regression/audit.py)、[fixed_inputs.py](regression/fixed_inputs.py)、[fixed_replay.py](regression/fixed_replay.py)、[fixed_report.py](regression/fixed_report.py)、[run_fixed.py](regression/run_fixed.py)、[dqn_learning.py](regression/dqn_learning.py)、[dqn_evaluation.py](regression/dqn_evaluation.py)、[dqn_reports.py](regression/dqn_reports.py)、[lifelong_replay.py](regression/lifelong_replay.py) |
| `scene_design/` 场景开发与避撞参考 | [references.py](scene_design/references.py)、[check_braking.py](scene_design/check_braking.py)、[check_formal.py](scene_design/check_formal.py)、[design_interactions.py](scene_design/design_interactions.py)、[audit_interactions.py](scene_design/audit_interactions.py)、[verify_secondary.py](scene_design/verify_secondary.py)、[audit_secondary.py](scene_design/audit_secondary.py) |
| `plotting/` 科研插图与轨迹 GIF | [style.py](plotting/style.py)、[method_schematic.py](plotting/method_schematic.py)、[lifelong.py](plotting/lifelong.py)、[k_shot.py](plotting/k_shot.py)、[public_ads.py](plotting/public_ads.py)、[budget_discovery.py](plotting/budget_discovery.py)、[discovery_space.py](plotting/discovery_space.py)、[weight_effectiveness.py](plotting/weight_effectiveness.py)、[transfer_sequence.py](plotting/transfer_sequence.py)、[fixed_regression.py](plotting/fixed_regression.py)、[animation.py](plotting/animation.py)、[replay_safe.py](plotting/replay_safe.py)、[replay_collision.py](plotting/replay_collision.py) |

---

## 二、统一 12-ADS 持续风险迁移主流程（#持续风险迁移）

统一 12-ADS（8 同范式 + 4 跨范式）的 E01–E05 完整评测与图表导出由 [统一流水线](lifelong/__main__.py) 按阶段依赖顺序调度。在项目根目录激活 `metadrive` 环境并配置公开 ADS 缓存路径（见[控制器说明](../controllers/README.md#二跨架构异质控制器4-跨范式目标)）后，可分阶段或一键复算：

```powershell
conda activate metadrive
python -B -m experiments.lifelong_seals record
python -B -m experiments.public_ads.continuous_sac release --device cuda
python -B -m experiments.public_ads.collect_continuous_sac
python -B -m experiments.public_ads.evaluate_continuous_sac
python -B -m experiments.lifelong_transfer
python -B -m experiments.accumulation.lifelong cold_start
python -B -m experiments.accumulation.lifelong cross_architecture
python -B -m experiments.accumulation.lifelong in_family
python -B -m experiments.regression.lifelong_replay --workers 4
python -B -m experiments.lifelong.report
python -B -m experiments.plotting.lifelong
```

也可以使用流水线入口从指定阶段开始运行（例如仅重跑统计复算与 7 张 600 dpi 正式图）：

```powershell
python -B -m experiments.lifelong --start-at statistics
```

持续演化阶段可用 `python -B -m experiments.accumulation.lifelong cold_start --workers 2` 将并发分支限制为两个。省略该参数时使用各阶段的默认并行度。

- **阶段 1（`Continuous SAC` 准入与基线补齐）**：训练并核验第 12 个 ADS，采集四场景 8,000 条原生轨迹至 `results/public_ads/physical/target/<scenario>/continuous_sac/`，并补齐该目标的 E01/E02/E03 基线矩阵。
- **阶段 2（12-ADS E01/E02/E03 增强矩阵）**：[lifelong_transfer.py](lifelong_transfer.py) 在 48 个场景–ADS 任务上运行 `+LocalRel`、`+Risk-UCB` 与 `Ours method (lifelong_full)` 的 144 条主动预算轨迹及 48 条共同反馈预测轨迹，逐笔核验分支概率与认知标准差。
- **阶段 3（E05-A/B/C 跨任务持续演化）**：[accumulation/lifelong.py](accumulation/lifelong.py) 依次执行 `cold_start`（两规则来源起步、192 位置）、`cross_architecture`（4 跨范式接力加末端 `PPO` 回测、160 位置）与 `in_family`（8 同范式满历史复核、256 位置，其中 `fixed` 以固定历史与增强选择器执行 32 个唯一回放并由正反序共享），所有制度统一为 `fixed`、`naive_append`、`debiased_append` 与 `full_lifelong`。
- **阶段 4（E04 同谱系策略固定套件回归）**：[regression/lifelong_replay.py](regression/lifelong_replay.py) 在 `ppo`、`dqn_continual` 与 `social_attention_ddqn` 的 36 个版本任务上回放 `lifelong_full` 并与 `DETOUR`、`Birchler`、`SDC-Scissor`、`SPECTRE`、`Ours v1` 配对计算 $Q_{90}$ 与前 100 次新增碰撞。
- **阶段 5（统一统计复算与 600 dpi 绘图）**：[统计报表](lifelong/report.py) 核验封存凭据与文档链接，导出 [summary.json](../results/lifelong_transfer/reports/summary.json) 与 [结果说明](../results/lifelong_transfer/README.md)；[plotting/lifelong.py](plotting/lifelong.py) 导出并审计 [figures/lifelong_transfer/](../figures/lifelong_transfer/) 下的 7 张正式 PNG。

---

## 三、分队列封存实验与专项复算入口

### 1. 原八目标队列（E01–E03／E05）与场景回放

[formal_scenarios.py](formal_scenarios.py) 按四维整数矩形域为每场景均匀不放回生成 500 个历史坐标（种子 `31013`）和 2,000 个目标坐标（种子 `61001`），排除全部开发坐标。[transfer_settings.py](transfer_settings.py) 登记固定六标量十步默认配置。

- **E01 少反馈预测**：`python -B -m experiments.history_transfer_experiment status`（支持 `qualify`、`check-scene`、`prepare`、`settings`、`collect-history`、`collect-targets`、`models`、`k-shot`、`report`、`figures`），结果位于 [results/k_shot_adaptation/](../results/k_shot_adaptation/README.md)。
- **E02 固定预算发现**：`python -B -m experiments.budget.run` 与 `python -B -m experiments.plotting.budget_discovery`，结果位于 [results/budget_discovery/](../results/budget_discovery/README.md)。
- **E03 动态权重消融**：`python -B -m experiments.weight_ablation.effectiveness` 与 `python -B -m experiments.plotting.weight_effectiveness`，结果位于 [results/weight_adaptation/](../results/weight_adaptation/README.md)。
- **E05 原队列知识积累**：`python -B -m experiments.accumulation.run`、`report` 与 `python -B -m experiments.plotting.transfer_sequence`，结果位于 [results/knowledge_accumulation/](../results/knowledge_accumulation/README.md)。
- **开发轨迹 GIF 回放**（只读复算，零新增仿真）：`python -B -m experiments.plotting.replay_safe` 与 `python -B -m experiments.plotting.replay_collision`，输出至 `figures/scene_replays/`，索引见 [Safe 回放](../results/scene_replays/safe/README.md)与[碰撞回放](../results/scene_replays/collision/README.md)。

### 2. 公开来源跨架构 ADS 扩展（`OnlineMC PPO`、`Social Attention DDQN`、`Decision Transformer`）

在配置 `cache/highway_agents` 的 `PYTHONPATH` 后，可分别复算三项公开策略的 E01、E02/E03 扩展、驾驶案例与低 RSS 裕度切片、E05 知识积累及 `Social Attention DDQN` 固定套件回归（结果位于 [results/public_ads/](../results/public_ads/README.md)）：

```powershell
# E01 采集与评测
python -B -m experiments.public_ads.evaluate_e01 collect --workers 8
python -B -m experiments.public_ads.evaluate_e01 control
python -B -m experiments.public_ads.evaluate_e01 evaluate
python -B -m experiments.public_ads.evaluate_e01 report
python -B -m experiments.plotting.public_ads e01

# E02 / E03 扩展与驾驶机理复算
python -B -m experiments.public_ads.extend budget --workers 5
python -B -m experiments.public_ads.extend frozen-prediction --workers 3
python -B -m experiments.public_ads.extend frozen-budget --workers 3
python -B -m experiments.public_ads.extension_report audit-budget --workers 3
python -B -m experiments.public_ads.extension_report budget
python -B -m experiments.public_ads.extension_report audit-frozen --workers 3
python -B -m experiments.public_ads.extension_report frozen
python -B -m experiments.public_ads.mechanism
python -B -m experiments.plotting.public_ads extension

# 公开三目标 E05 与 Social Attention DDQN 回归
python -B -m experiments.public_ads.accumulation run
python -B -m experiments.public_ads.accumulation report
python -B -m experiments.plotting.public_ads accumulation
python -B -m experiments.public_ads.attention_regression report
python -B -m experiments.plotting.public_ads regression
```

### 3. PPO 与修订 DQN 同谱系连续修复回归（#ppo-连续修复中的回归测试）

- **PPO 固定套件回归**（结果位于 [results/policy_regression/fixed_suite/](../results/policy_regression/fixed_suite/README.md)，配图位于 `figures/policy_regression/ppo/`）：
  ```powershell
  python -B -m experiments.regression.run_fixed physical
  python -B -m experiments.regression.run_fixed replay --workers 8
  python -B -m experiments.regression.run_fixed report
  python -B -m experiments.plotting.fixed_regression
  ```
- **修订 DQN（`Continual DQN`）持续学习与回归**（协议见[实验计划](../docs/experiment_plan.md#dqn-持续学习与替换条件)，结果位于 [results/dqn_continual_learning/](../results/dqn_continual_learning/README.md)，配图位于 `figures/policy_regression/dqn/continual_learning/`）：
  ```powershell
  python -B -m experiments.regression.dqn_learning --physical-actions <SCENE>
  python -B -m experiments.regression.dqn_evaluation register
  python -B -m experiments.regression.dqn_reports publish
  python -B -m experiments.regression.run_fixed report --policy dqn
  python -B -m experiments.plotting.fixed_regression --policy dqn
  ```

---

## 四、并行与硬件约定

[parallel.py](parallel.py) 统一使用 Windows `spawn` 多进程，物理采集最多 16 进程，常规在线 CPU 选例最多 8 进程，每个工作进程限制为单计算线程。解释器统一使用 `F:/anaconda3/envs/metadrive/python.exe`。`Target-only GP` 与本文方法在线适配在 CPU 上并行运行；`Pooled historical GP` 使用 CUDA float64 单任务串行运行。
