# DIVA-Mine Cut-in 技术设计

DIVA-Mine 学习历史被测系统在同一物理场景空间中的漏洞响应结构，并以少量目标系统诊断测试个性化选景。模型不接收 SUT 身份作为输入；SUT 身份只用于执行、数据分组和可追溯记录。

当前唯一有效的 Cut-in 场景合同见 [DIVA_Mine_CutIn_Scenario_Contract.md](DIVA_Mine_CutIn_Scenario_Contract.md)。场景由左右切入候选和五个连续参数唯一确定：初始车间距离、主车初始速度、相对速度、切入开始位置、切入路径长度。红车只按给定速度和给定空间轨迹行驶，不与主车交互。

正式得分为：有效碰撞 1.0、有效 critical near-miss 0.5、其他 0。只有有效碰撞或完整完成且无事件的场景可用于更新漏洞后验；无效与截断场景仍计入预算，但不作为 latent 标签。

source 阶段在四个训练 SUT 的公共可评估 anchors 上拟合共享均值和低秩漏洞基。每侧单独拟合 GP；LOSO 仅以 source 标签选择 rank 1 或 2。target 阶段先进行 K-shot 诊断，再在相同预算下比较随机、共享均值、target-only GP 与 DIVA 的累计危险发现收益。

当前物理空间已经通过快速可达性探查确认存在有效碰撞区域；该探查只用于验证场景空间包含风险，不构成 DIVA 方法性能结论。新的 source bank、先验拟合、验证和测试都必须使用当前合同重新执行。

每次正式 DIVA rollout 的 runner 与环境 horizon 都固定为 720 步。该长度覆盖当前 Cut-in 中无事故主车到达规定路线终点所需的约 581 步；480 步的截断记录只能作为 `censored`，不能作为完整零分标签。

## 代码与产物

- `mvr/diva/`：方法实现。`types.py` 定义设计与观测记录，`episode_executor.py` 运行物理场景，`source_bank.py` 构造和校验 source bank，`factorization.py`、`basis_gp.py` 与 `prior.py` 拟合低秩先验，`posterior.py`、`acquisition.py` 与 `miner.py` 实现目标系统诊断和挖掘，`baselines.py` 实现 target-only GP。
- `mvr/scripts/`：命令入口，所有 DIVA 入口均以 `*_diva_*` 或 `*_diva_cutin_*` 命名。它们只编排配置、文件和物理调用，不承载方法逻辑。
- `mvr/configs/diva_cutin.yaml`：唯一的 DIVA Cut-in 冻结配置。
- `results/diva/cutin_g01/`：当前物理合同下的可复现实验包。此目录只保存 casebook、物理观测、先验、评价报告和重放文件；不使用版本号目录。

当前已保存的 `hazard_probe.*`、`verified_collision.*` 和 `parameter_coverage.*` 仅证明场景空间可执行且含有危险区域。`source_casebook.json` 是待执行的 source 输入，不是 source 实验结果。

## 执行顺序

```powershell
conda run -n metadrive python -m mvr.scripts.build_diva_cutin_casebook --config mvr/configs/diva_cutin.yaml --output results/diva/cutin_g01/source_casebook.json
conda run -n metadrive python -m mvr.scripts.collect_diva_source_bank --config mvr/configs/diva_cutin.yaml --casebook results/diva/cutin_g01/source_casebook.json --output results/diva/cutin_g01/source_observations.jsonl
conda run -n metadrive python -m mvr.scripts.analyze_diva_prior --config mvr/configs/diva_cutin.yaml --source results/diva/cutin_g01/source_observations.jsonl --output results/diva/cutin_g01/source_analysis.json
conda run -n metadrive python -m mvr.scripts.fit_diva_prior --config mvr/configs/diva_cutin.yaml --source results/diva/cutin_g01/source_observations.jsonl --output results/diva/cutin_g01/prior.pt
```

只有 source gate 通过后，才运行 `evaluate_diva_cutin` 进行 validation；test 必须使用冻结的相同配置与输入。
