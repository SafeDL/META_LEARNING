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
