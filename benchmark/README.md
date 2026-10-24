# 碰撞发现基准

本目录组织 HighwayEnv 碰撞发现实验：采集驾驶系统反馈、训练测试模型、按预算比较方法、确认新增碰撞并生成报告。历史系统、目标系统、四类场景及指标的通俗解释见 [项目说明](../README.md)。

## 数据与执行规则

每项任务提供 512 个候选场景，允许查询 128 次，其中 8 次为共同初始化。算法每获得一次反馈，再选择下一个场景；未查询的目标标签不会提供给它。不同方法共用候选库、初始场景和五个配对随机种子。

候选库实验使用预先采集的真实反馈进行隔离回放，建库成本和算法计算时间分别记录。连续搜索对算法提出的新场景实际运行仿真。初始化、重复执行和额外参考系统调用都计入相应调用账本。

| 文件 | 负责什么 |
|---|---|
| [schema.py](schema.py)、[scenarios.py](scenarios.py) | 场景参数、反馈格式、困难组合、数据划分和去重规则 |
| [protocol.py](protocol.py)、[studies.py](studies.py) | 固定实验预算，登记目标、历史来源及比较任务 |
| [query.py](query.py)、[executions.py](executions.py) | 控制可读取的信息，执行场景并记录调用与耗时 |
| [collision_runs.py](collision_runs.py)、[continuous_runs.py](continuous_runs.py) | 分别运行候选库比较和连续搜索 |
| [collision_metrics.py](collision_metrics.py)、[execution_events.py](execution_events.py) | 从轨迹计算碰撞、去重、行为类别、首次发现及事件时间 |
| [certify_collisions.py](certify_collisions.py) | 用真实参考系统结果确认版本新增和历史集合特有碰撞 |
| [report_collision.py](report_collision.py)、[report_continuous.py](report_continuous.py) | 分别汇总两类实验的统计、成本和图表 |

## 运行顺序

从项目根目录激活 `metadrive` 环境。`python -B -m benchmark.run_collision` 串联资格、策略训练和下列各阶段；`--resume` 从相同配置下已完成的阶段继续。日志和进度位于 `results/gcrt/collision/pipeline/`。

分阶段执行时，先完成 [驾驶系统训练与资格检查](../controllers/README.md)，再按顺序运行：

```powershell
python -B -m benchmark.collect history
python -B -m benchmark.build_collision_models core
python -B -m benchmark.develop_collision
python -B -m benchmark.build_collision_models complete
python -B -m benchmark.freeze_collision
python -B -m benchmark.collect targets
python -B -m benchmark.evaluate_collision
python -B -m benchmark.certify_collisions
python -B -m benchmark.report_collision
```

连续搜索单独开发、固定配置并评价：

```powershell
python -B -m benchmark.develop_continuous
python -B -m benchmark.freeze_continuous
python -B -m benchmark.evaluate_continuous
python -B -m benchmark.report_continuous
```

当前登记实验已完成：48 份历史／开发档案、156 个模型、288 项开发比较、96 个目标库、10880 项候选库比较、96 份新增碰撞认证，以及 40 项连续开发和 100 项连续正式比较。保存位置见 [结果索引](../results/README.md)。

## 复核已有结果

下列命令只读已有记录，不重新运行驾驶系统。它们从原始轨迹和调用账本重算结果；`--require-complete` 要求相应矩阵全部通过，缺项或不一致时返回非零退出码。

```powershell
python -B -m benchmark.audit_collision_data --require-complete
python -B -m benchmark.audit_collision_results --require-complete
python -B -m benchmark.audit_collision_certification --require-complete
python -B -m benchmark.audit_continuous_results --require-complete
python -B -m benchmark.audit_collision_statistics --scope all --require-complete
```

对应检查内容为：历史与开发档案、正式候选库运行、新增碰撞标签、连续运行、配对统计及结果表。复核报告位于 `results/benchmark/collision/`。数据概览可通过 `python -B -m benchmark.describe_collision_data` 导出。

MetaDrive 的配置、脚本和回归测试仍在 [simulation/metadrive](../simulation/metadrive/README.md)，按该平台说明单独运行。

## 局部历史经验的开发实验

[discovery.py](discovery.py) 实施[分阶段碰撞发现方案](../docs/collision_discovery_plan.md)，使用现有历史与开发对象，在独立目录比较来源核、局部权重和前瞻。全库去重碰撞分母仅保留在评估端，选择器获得已付费反馈。

```powershell
python -B -m benchmark.discovery diagnose
python -B -m benchmark.discovery pilot
python -B -m benchmark.discovery report
```

`pilot` 包含带宽开发和 440 项主比较，完整结果自动保存，可在相同代码与配置下复用已完成运行。只有阶段 B 门槛通过后才可执行 `selection`；扩展场景和正式目标仍需后续实现与冻结。诊断和门槛位于 `results/benchmark/collision/discovery/`，实际进展见[执行记录](../docs/collision_discovery_execution.md)。

原门槛失败后的历史/开发诊断使用独立入口，保留原比较和门槛：

```powershell
python -B -m benchmark.discovery_consistency train
python -B -m benchmark.discovery_consistency_search
python -B -m benchmark.discovery_consistency_audit
python -B -m benchmark.discovery_support
python -B -m benchmark.discovery_support_search
python -B -m benchmark.discovery_consistency_audit wide
python -B -m benchmark.discovery_diagnostic_report
python -B -m benchmark.discovery_feasibility
```

匹配训练保持原网络和4/8/16训练支持集，比较四个新模型及360项配对预测验证，再做80项128查询的开发搜索。支持集诊断改用原方案最终训练列出的8/16/32/64/128范围，并保持其他训练设置不变；它增加每次更新的监督点数和计算量，成本单独报告。两项诊断都只使用已有历史/开发档案，不改变原B门槛，也不启动C/D/E。

用户确认保留128次预算后，[后续协议](../docs/collision_discovery_recovery.md)将新的开发标准固定为剩余漏检减少至少50%，同时要求数量不降低、至少两个谱系有正收益。首轮失败保留。首个新原型比较历史响应核的目标适配与固定参数对照：

```powershell
python -B -m benchmark.discovery_recovery freeze
python -B -m benchmark.discovery_recovery run
python -B -m benchmark.discovery_recovery report
```

新运行、协议和门槛位于独立的`discovery/recovery/`目录，不能覆盖原`pilot/gate.json`。

其后两项诊断继续使用同一128次预算，完整保留失败结果和相应机制对照：

```powershell
python -B -m benchmark.discovery_recovery_audit
python -B -m benchmark.discovery_safety freeze
python -B -m benchmark.discovery_safety run
python -B -m benchmark.discovery_recovery_audit safety
python -B -m benchmark.discovery_geometry_diagnosis
python -B -m benchmark.discovery_geometry_search freeze
python -B -m benchmark.discovery_geometry_search run
python -B -m benchmark.discovery_recovery_audit geometry
python -B -m benchmark.discovery_diagnostic_report recovery
```

连续安全响应轮次只增加历史间距/TTC的预测坐标。相对运动轮次只改变历史预测器的输入，并加入直接物理特征对照；该对照也参与最强基线选择。后续汇总要求三项独立审计全部通过。

强目标分量的200项诊断与仅目标融合对照：

```powershell
python -B -m benchmark.discovery_complementarity
python -B -m benchmark.discovery_adaptive_target freeze
python -B -m benchmark.discovery_adaptive_target run
python -B -m benchmark.discovery_adaptive_target_audit
python -B -m benchmark.discovery_adaptive_target report
python -B -m benchmark.discovery_diagnostic_report adaptive
python -B -m benchmark.discovery_margin_diagnosis
```

这些入口保持碰撞为唯一在线监督。利用同次间距反馈的[联合响应方案](../docs/collision_discovery_joint_response.md)单独登记，四个历史训练模型、16个来源后验和280项冻结开发运行均已完成并独立复核，主方法未通过开发门槛：

```powershell
python -B -m benchmark.discovery_joint_response train
python -B -m benchmark.discovery_joint_response prior
python -B -m benchmark.discovery_joint_response freeze
python -B -m benchmark.discovery_joint_response run
python -B -m benchmark.discovery_joint_response_audit available
python -B -m benchmark.discovery_joint_response audit
python -B -m benchmark.discovery_joint_response_negative_controls
python -B -m benchmark.discovery_joint_response report
python -B -m benchmark.discovery_diagnostic_report joint
```

`available`只检查已完整保存的运行；完整`audit`要求280项全部完成。模型与运行在`results/gcrt/collision/discovery/joint_response/`，协议、独立复核、阶段结论及307项完整回归的JUnit在`results/benchmark/collision/discovery/joint_response/`。同次间距作为新增在线信息，需要与同反馈的目标联合模型比较；不能只对原二值模型解释收益。

`python -B -m benchmark.discovery_representation paid`复现相同已付费前缀的物理特征预测诊断：40项前缀、3种预测器，不增加选例或仿真；输出在`joint_response/physical_feature_diagnosis/`。它用于分析剩余碰撞的概率和排名，不能替代128次查询的实际发现曲线。

[独立物理分量方案](../docs/collision_discovery_physical_mixture.md)已实现选择、320项矩阵、门槛与独立审计入口。先完成并审计整个联合响应矩阵，再依次使用`python -B -m benchmark.discovery_physical_mixture freeze`、`smoke`、`run`、`audit`、`report`。`smoke`的8项属于320项主矩阵；先用`python -B -m benchmark.discovery_physical_mixture_negative_controls`独立检查全部8项及损坏记录，再运行其余项。此前强对照、全部7个联合响应版本、FSBO与物理GP混合及同CUDA后端FSBO均参与参考选择。当前已冻结并运行；25项针对性检查、332项全工程回归和8项实际首轮复核通过，JUnit及审计位于`results/benchmark/collision/discovery/physical_mixture/`。

运行期间使用`python -B -m benchmark.discovery_physical_mixture_audit available`增量复核已完成项；完整审计后使用`python -B -m benchmark.discovery_physical_mixture_report`生成曲线、预算点及成本。该报告入口要求整组审计通过。已验证聚合曲线与上一轮9个完整方法一致，以及零碰撞召回和全部8种计时格式。`python -B -m benchmark.discovery_representation paid_report`从已保存数值重建物理特征诊断的中文说明，不重新拟合或查询。

物理分量主方法40项已完成并独立复核，召回98.385623%，相对最强参考的剩余漏检减少35.685464%，未达到50%门槛；整组320项随后全部完成并通过审计。`python -B -m benchmark.discovery_physical_mixture_diagnosis`复现PPO切出的相同付费前缀预测诊断，输出位于`physical_mixture/paid_prefix_diagnosis/`。它核对30个真实下一选点及75项独立NumPy全库概率，并分析两组预测的静态凸混合排名下界；没有新增查询，也不将预测排名当作实际发现。

`python -B -m benchmark.discovery_physical_mixture_diagnosis influence`复现单条已付费碰撞对后验条件化及重新适配的影响，输出在上述目录的`collision_influence/`。`python -B -m benchmark.discovery_phase_diagnosis`核对24份历史/开发档案的12,288条事件记录，输出在`physical_mixture/phase_diagnosis/`。这两个入口只做评估端诊断，旧标签、候选与冻结运行保持不变。

物理分量320项现已全部完成并独立审计，完整曲线、预算点与成本见`physical_mixture/report.md`；主方法准入未通过。`python -B -m benchmark.discovery_conditional_adaptation`复现40个已付费前缀上的160项初始化/适配目标预测对照，输出在`conditional_adaptation/`。全部1,440组分量全库概率已独立核对，7项机制测试通过；这组预测没有新增选例或仿真，也不作为在线碰撞发现结果。

`python -B -m benchmark.discovery_phase_response_diagnosis`执行[阶段响应方案](../docs/collision_discovery_phase_response.md)预先登记的240项预测诊断，结果在`phase_response/`。比较历史阶段迁移、同信息目标GP和未观察阶段填零消融，分别使用联合响应与FSBO初始化。它只读取既有查询前缀的阶段标签作目标适配，未查询阶段与碰撞仅用于评估；后续在线接口和矩阵须另行登记。

240项阶段预测已完成并通过4,560组完整候选分量概率核对，当前没有支持扩大在线验证的排名收益。`python -B -m benchmark.discovery_phase_prior_diagnosis`复现12项历史频率与初始边际概率的核查，输出在`phase_calibration/prior_check.md`。这一入口只检查旧核下均值初始化的概率含义，不执行新的选例或完整比较矩阵。

`python -B -m benchmark.discovery_phase_calibration`执行已登记的240项同付费前缀重复，只调整初始潜在均值使初始独立分量的边际概率匹配相同历史频率。旧阶段预测保持原样，输出在`phase_calibration/`；该入口先核对旧协议、源码与初始化证据，不启动在线搜索。

该重复已完成，排名仍未支持在线扩张。[历史残差尺度方案](../docs/collision_discovery_residual_scale.md)回到碰撞为唯一在线监督：依次执行`python -B -m benchmark.discovery_residual_scale prepare`、`cache`、`audit`、`fit`，只用历史伪目标前缀估计一个核幅度。16个历史来源后验已经独立核对，前缀缓存正在构建；结果在`residual_scale/`，模型特征在`results/gcrt/collision/discovery/residual_scale/`。这些是历史参数学习，不计为新目标查询或新的碰撞发现运行。

历史尺度学习现已全部完成。`python -B -m benchmark.discovery_residual_selection_audit`复核16个缓存任务、800项参数比较及所选β；独立拟合64次、概率前缀320个。已选尺度为切入0.3、切出1.0，240项在线控制矩阵先预登记在`residual_scale/online_design.json`，现已完成并保留失败结论。在线冻结后不要重写历史选择报告或其审计证据；后续只读取和核对已绑定的文件。

在线实现现已提供`python -B -m benchmark.discovery_residual_online freeze`、`smoke`、`run`、`audit`、`report`。全工程回归通过后先运行六种方法各一项，再执行`python -B -m benchmark.discovery_residual_online_audit smoke`和`python -B -m benchmark.discovery_residual_online_negative_controls`。正控制及损坏控制齐备后才能启动其余矩阵。运行在`results/gcrt/collision/discovery/residual_online/runs/`，协议与报告在`residual_online/`。`python -B -m benchmark.discovery_residual_online_audit available`用于增量验收；完整审计要求240项，并独立重建720次完整候选选点。

392项全工程/MetaDrive回归、六项实际首轮及其18次独立完整候选选点、10项损坏控制均已通过，240项矩阵现已冻结并启动。实时进度位于`residual_online/progress.json`；同一会话已串接全量审计及报告，任一失败即停止。不要修改已冻结的`discovery_residual_online.py`、`residual_search.py`及历史选择证据；只读增量审计可与运行并行。完整验证之前不把局部完成数或首项结果视为开发成功。

该240项已完整审计并保留失败结论。后续RGPE80项、ScaML160项和[收敛来源推断比较](../docs/collision_discovery_converged_inference.md)按依赖顺序独立保存。`python -B -m benchmark.discovery_converged_response`的`prepare/prefix/history`只做历史数值修正与参数复核，已完成16任务、2,400评分；不计为开发发现。新`discovery_converged_search`的`prepare/freeze/smoke/run/report`提供160项配对搜索，`discovery_converged_search_audit`的`smoke/negative/complete`负责原始反馈、因果可靠性及独立收敛选点核对。冻结前必须完成前序强基线与全工程回归，不重写已绑定的旧协议和报告。

RGPE80项和ScaML160项现均已完整运行、独立审计并报告。收敛来源160项随后也已全部完成，20,480条原始反馈、480次独立全候选重建及12项损坏控制通过；全部38种方法的汇总、曲线、预算点和12组关键配对区间核查通过。主方法召回98.195405%，低于最强参考98.585721%，未通过50%漏检减少门槛，不进入C/D/E。当前状态见`results/benchmark/collision/discovery/execution_status.json`，完整结果见`converged_search/report.md`。

`python -B -m benchmark.discovery_source_prior_stationarity`只读检查16个历史来源后验的完整512点训练模式，使用原CUDA核及迭代值独立重算白化梯度，并复现保存的联合均值/协方差。它不重训、不修改来源文件，也不替代目标条件推断或完整搜索验收；结果保存在`converged_response/source_prior_stationarity.json`。

`python -B -m benchmark.discovery_converged_diagnosis sets/smoke/prefix`分别执行已付费主方法的配对集合、首项及全量状态重建，阶段参数逐个执行；`python -B -m benchmark.discovery_converged_diagnosis_report`独立复算保存数组并汇总六种固定状态排序。结果在`converged_search/diagnosis/`。它只使用原付费支持集复现预测，不调用查询器；剩余完整库标签仅在评估分组时使用，排序对照不视作在线发现结果。

`python -B -m benchmark.discovery_feature_reliability prepare/run`用于[目标适配特征距离的历史诊断](../docs/collision_discovery_feature_reliability.md)。它复用历史目标适配状态，在收敛条件推断下比较参数、固定特征、适配特征及全局可靠性；不重算过去标签的预测得分，不增加开发选例，也不修改已登记160项搜索。

`python -B -m benchmark.discovery_response_geometry prepare`、`smoke`、`run`依次登记并复现四种核几何的固定付费状态；`python -B -m benchmark.discovery_response_geometry_report`独立汇总数组。输出在`converged_search/diagnosis/geometry/`。40项/120状态已完成；相同均值、原可靠性权重及平均方差用于隔离核几何，未增加查询或仿真，不作为在线收益证据。

`python -B -m benchmark.discovery_geometry_retention prepare`、`run`比较历史任务上的适配核、冻结核、固定各半与查询前学习视图；`python -B -m benchmark.discovery_geometry_retention_report`从保存数组独立重算。输出在`geometry_retention/`。16任务/640项评分、128组独立分量核对及160项原控制复现已完成，预登记历史筛选未通过，未增加在线矩阵、网络训练、开发选例或物理调用。

`python -B -m benchmark.discovery_rank_diagnosis prepare`、`run`对[固定历史状态](../docs/collision_discovery_rank_diagnosis.md)比较概率及百分位融合。16任务/80状态/480条评分已独立核对，预登记筛选未通过，不启动相应在线变体。输出在`rank_diagnosis/`。

`python -B -m benchmark.discovery_tree_screening prepare`、`smoke`、`run`后，执行`python -B -m benchmark.discovery_tree_screening_audit`及`python -B -m benchmark.discovery_tree_screening_report`，完成[逐来源响应的历史树模型筛选](../docs/collision_discovery_tree_screening.md)。48项、6,144条反馈、144次独立选点及10项损坏控制已验收，逐来源版召回84.108227%、汇总响应版82.013527%、物理版74.366179%，历史筛选通过。首次审计及报告的序列化失败、独立恢复和成本完整保存，两个报告回归检查通过，全部48项查询复用。输出在`tree_screening/`；下一步先做来源坐标交叉拟合，尚未登记新的开发矩阵或进入C/D/E。

来源坐标交叉拟合入口为`python -B -m benchmark.discovery_tree_crossfit prepare`，再执行`python -B -m benchmark.discovery_tree_crossfit_audit sources`；首轮和完整回放为`python -B -m benchmark.discovery_tree_crossfit smoke`、`run`，最终核对为`python -B -m benchmark.discovery_tree_crossfit_audit complete`，报告为`python -B -m benchmark.discovery_tree_crossfit_report`。3项新增及2项报告检查、80次来源独立重建、8,192个折外预测、完整48项/6,144条反馈/144次独立选点及10项损坏控制已通过；筛选未通过，逐来源版77.340510%召回但数量及AUC低于汇总控制，不增加开发矩阵。输出在`tree_crossfit/`，没有新增仿真。

`python -B -m benchmark.discovery_source_safety_prediction`只诊断既有历史间距/TTC的折外来源预测，不训练目标模型或查询选择器。80次回归、8,192个折外坐标、独立叶均值、未查询安全量污染及48条排序指标均已核对，预定主要间距通道未通过双场景条件，未追加搜索矩阵；TTC辅助结果完整保留。输出在`source_safety_prediction/`。

`python -B -m benchmark.discovery_budget_lookahead prepare/probe/run`（每次选择一个阶段参数），以及`python -B -m benchmark.discovery_budget_lookahead_report`诊断固定历史状态的全候选前瞻。16状态的129,024个虚拟分量、162组独立分支和128条实际批量记录已完整核对。局部64步原ENS出现微小正线索，5步未改善；不作为正式阶段C或128次在线结果。输出在`budget_lookahead/`，分支计算与中止报告的成本分别保存。

`python -B -m benchmark.discovery_batch_reordering prepare/run`复用上述概率场比较附加计分。主要局部64步因切出退步未通过，192条记录和256个分数数组另经重算；不把全局辅助结果改为主要结论。输出在`budget_lookahead/reordering/`，没有新开发查询或仿真。阶段顺序调整仅为[待确认建议](../docs/collision_discovery_budget_lookahead.md)，不提供已生效的开发准入。

`python -B -m benchmark.discovery_batched_virtual`验收同一虚拟条件推断的批量实现。六项专门检查、16个完整历史状态的129,024个模式和64次原选点全部通过；另以两场景支持8/64/127下的48,132个CPU模式核对逐分量概率，最大差3.34e-15。64点配对参考最低加速28.46倍，计算准备条件通过；结果在`budget_lookahead/batched_compute/`。原模型、计分和准入条件保持，新的开发矩阵仍待阶段顺序决定。
