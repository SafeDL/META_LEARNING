# FBRT 失效记忆回归测试

## S01 全量真值失效发现开发试验

S01 使用一份固定的 2,048 场景清单，包含 4 个参数轴。六个历史 source 是六个 IDM 构建版本；每个版本都在全部场景上执行一次，形成 12,288 条历史响应。目标 SUT 是 `merge_blind06`，也在全部 2,048 个场景上执行，得到仅供评价使用的真实失效集合。FM²-FBT、消融和各基线只通过 `TargetOracle.query` 获得目标反馈，每方法限 50 次查询。全量目标真值不得用于训练、模型选择或查询前排序。本轮训练和选择器各用一个种子。

从仓库根目录在 `metadrive` 环境中，以新的空结果目录运行 `python -m methods.failure_memory_regression.fm2_s01_pilot --root <新结果目录> --stage all`。阶段依次为 `freeze → history → train → bank → evaluate_a → evaluate_b → report`；也可用 `--stage` 单独执行。默认配置为 `configs/fm2_s01_full_history.json`，已完成的冻结结果位于 [`fm2_s01_full_history/`](../../results/method_chains/failure_memory_regression/fm2_s01_full_history)。已冻结的清单不能覆盖；既有结果请直接读取。历史与目标响应库中的既测条目由冻结配置指定并经场景及构建 ID 校验后复用，成本账本区分复用和新执行。

本轮目标真值为 408/2,048 个确认失效。50 次查询下，HistoryRank-UCB-v2 找到 50 个，FM²-FBT 找到 36 个；因此 FM²-FBT 没有超过最强基线。完整指标、逐次查询和审计见结果目录的 [`report.md`](../../results/method_chains/failure_memory_regression/fm2_s01_full_history/report.md)。这是一轮开发试验，不能视作独立确认。

当前方法统一称为“FBRT 失效记忆后验利用”，冻结协议中的方法 ID 是 `FBRT-Memory-Exploit-v3`：从历史碰撞及附近通过记录建立失效模式，在每次目标查询后更新风险后验，按预测的失败概率选择下一场景。它与五种保留的对照方法共用候选库、目标结果和查询预算。活动源码与配置文件使用职责名称。

`selector.py` 实现选择器和只在查询后揭示目标结果的 `TargetOracle`；`pattern_memory.py`、`bayes_model.py` 分别构造模式特征和概率模型。`catalogue.py` 编译场景，`experiment.py` 使用项目的 `highway_sim_env/` 与 `sut_algorithms/highway_env/` 测量物理结果，`replay.py` 读取已测银行并完成六方法离线比较，`benchmark.py` 进行留一物理种子簇的覆盖次数分析。`archive.py`、`schema.py` 和 `replay_utils.py` 提供共同的数据契约；`report.py` 生成物理实验报告，`audit.py` 检查初始状态场景。

历史 IA/IB 多车交互支线已退出活动测试：其场景配置、生成器、测量入口和专用测试均已移除。已测数据及阴性结果保留在 `results/method_chains/failure_memory_regression/interaction*` 供来源核查，不纳入当前双向版本变化结论。下一轮各功能场景的独立参数轴和开发候选清单见 [`FBRT_Scenario_Parameter_Space_V3.md`](../../docs/FBRT_Scenario_Parameter_Space_V3.md) 与 `prepare_scenario_sampling.py`。

历史交互实验的构建 ID 仍出现在已保存的结果中：`mobil_rear_state_age` 表示后车状态滞后 300 ms，`mobil_rear_state_age080` 表示 800 ms，`ppo_obs_age020_v2` 表示 PPO 观测滞后 200 ms。`mobil_rear_state_age080` 已从当前构建注册表移除，仅用于识别冻结的历史结果；这些 ID 不进入新的 V3 场景清单。

正式结果索引见 [`results/method_chains/failure_memory_regression/README.md`](../../results/method_chains/failure_memory_regression/README.md)：六方法冻结银行离线比较位于 `repair_exploit_v3_fullbank/`，六方法统计比较位于 `current_method_statistical_comparison.md`，覆盖槽留一物理种子簇评测位于 `memory_exploit_v3/`。已淘汰原型及交互留出验证的结论在结果索引中保留，不进入活动代码。

在项目根目录运行离线回放；它不会新增物理回合：

```powershell
conda run -n metadrive python -m methods.failure_memory_regression.replay --offline-only --paired-repeats 10 --output results/method_chains/failure_memory_regression/memory_exploit
```

V6.1 的双向版本测试另由 [`bidirectional.py`](bidirectional.py) 实现，配对真值、双池预算、共享目标响应模型、有向历史边界与消融对照均不混入上述冻结单向 V3 回放。`bidirectional_figures.py` 生成仅用于评估的有限网格图。首个非学习型 IDM release chain 和运行说明见 [`nl_release/README.md`](../../results/method_chains/failure_memory_regression/nl_release/README.md)；其开发/确认结果必须分别解释，不能把单向旧结果作为双向方法优势证据。

核心确认未支持 `directed_residual` 相对静态风险或坐标消融的一致优势。`bidirectional.py` 现包含阶段二 `coordinate_context_ucb_pure`、`directed_context_ucb_pure` 与弱先验 `directed_context_ucb_loose`：它们使用共享目标失败后验和父风险 offset，以确定性后验 UCB 排序回退，以后验均值互补分数排序改善，不使用覆盖探测或邻域前沿覆盖。候选及新物理确认协议见 [`contextual_confirmation`](../../results/method_chains/failure_memory_regression/contextual_confirmation/README.md)。

阶段二完整 NL/PPO bank 未显示一致优越性或有向边增益。阶段三加入目标反馈驱动的上下文局部父风险斜率校准：它在 PPO V1→V2 回归方向优于静态风险和普通坐标残差，但坐标校准消融找到更多；它仍漏掉全部 NL 回归，且在 PPO V0→V1 改善方向低于 target-only。阶段三结果见 [`offset_calibration_confirmation`](../../results/method_chains/failure_memory_regression/offset_calibration_confirmation/findings.md)。

阶段四显示上下文启动主要帮助回归搜索，却压缩改善方向的后验排序预算；没有一致双向优势，见 [`context_bootstrap_confirmation`](../../results/method_chains/failure_memory_regression/context_bootstrap_confirmation/findings.md)。阶段五已转向只在回归方向做上下文启动，改善方向立即使用校准后验，并用新物理 bank 确认。

第四组 NL-IDM 独立冻结银行显示 `directed_margin_frontier` 在有限网格上同时提高回退和改善发现数，详情见 [`确认记录`](../../results/method_chains/failure_memory_regression/nl_release/chain2_confirmation_margin/confirmation_findings.md)。改进中的回退收益主要来自父版本连续安全余量覆盖和反馈前沿，定向特征的额外收益主要体现在改善方向；变化仅来自一个场景族，尚无跨族显著性或 PPO 权重继承证据。


Stage five has completed. Regression-only context bootstrap improved the NL regression pool over its unbootstrapped counterpart, but it did not beat role-gated search there or directed offset calibration on PPO regressions and improvements. The full frozen comparison is documented in results/method_chains/failure_memory_regression/regression_bootstrap_confirmation/findings.md. Post-confirmation replays rejected direction-isolated calibration, a static/directed online mixture, and a symmetric change-probability model. The primary's suite-wide count is descriptively ahead of the preregistered controls, so the next step is an independent context replication with a larger fixed holdout before making a superiority claim.


The stage-six independent context replication is complete at [`superiority_replication_confirmation`](../../results/method_chains/failure_memory_regression/superiority_replication_confirmation/README.md). Both NL and PPO banks contain 3,267 paired physical episodes and pass the no-new-execution cache audit. The stage-five primary selector did not establish consistent superiority: it found 45 changes at `D@20` across six valid task directions, compared with 54 for coordinate role-gated search; the family-clustered comparisons remain descriptive with only three scenario families. Post-confirmation replays keep role-gated variants exploratory and provide no clear incremental benefit from directed edge features. The next selector revision must be frozen and tested on new contexts; neither stage-five nor stage-six banks can confirm it. Direction-conditioned, prequential-mixture, and flip-label models remain exploratory implementations only.

Stage seven nominates the existing `coordinate_role_gated` selector for a fresh-family confirmation. Its protocol is frozen across eight executable families and 24 unseen contexts for both release chains at [`role_gated_generalization_confirmation`](../../results/method_chains/failure_memory_regression/role_gated_generalization_confirmation/README.md). The directed-edge variant remains an explicit ablation; stage-five/six results are development evidence, not confirmation. Physical measurement is paused before evaluation at NL 4,153/8,712 and PPO 3,951/8,712 valid cached episodes; rerunning the measure command resumes from cache.

当前单上下文开发测试使用同一组 8 个可执行场景族、每族 1 个基准上下文及完整 11×11 网格。`prepare_single_context_grid_development.py` 生成 NL/PPO 各 968 个场景的独立清单；三上下文阶段七银行仍按原冻结协议保存。操作说明见 [`single_context_grid_development`](../../results/method_chains/failure_memory_regression/single_context_grid_development/README.md)。
