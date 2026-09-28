# FBRT 失效记忆回归测试

当前方法统一称为“FBRT 失效记忆后验利用”，冻结协议中的方法 ID 是 `FBRT-Memory-Exploit-v3`：从历史碰撞及附近通过记录建立失效模式，在每次目标查询后更新风险后验，按预测的失败概率选择下一场景。它与五种保留的对照方法共用候选库、目标结果和查询预算。活动源码与配置文件使用职责名称。

`selector.py` 实现选择器和只在查询后揭示目标结果的 `TargetOracle`；`pattern_memory.py`、`bayes_model.py` 分别构造模式特征和概率模型。`catalogue.py` 编译场景，`experiment.py` 使用项目的 `highway_sim_env/` 与 `sut_algorithms/highway_env/` 测量物理结果，`replay.py` 读取已测银行并完成六方法离线比较，`benchmark.py` 进行留一物理种子簇的覆盖次数分析。`archive.py`、`schema.py` 和 `replay_utils.py` 提供共同的数据契约；`report.py` 生成物理实验报告，`audit.py` 检查初始状态场景。

本方法的四份配置集中在 [`configs/`](configs/)；当前交互场景使用 `interaction_catalogue.yaml`。`interaction.py` 是物理测量入口。`interaction_holdout_catalogue.yaml` 保留“后车状态滞后 800 ms 交互留出验证”的冻结场景定义；`prepare_holdout.py` 和 `validate_holdout.py` 对应其独立验证，结果中目标回归池为零。

交互实验的控制器条件统一用毫秒表示：`mobil_rear_state_age` 是“后车状态滞后 300 ms”，`mobil_rear_state_age080` 是“后车状态滞后 800 ms”，`ppo_obs_age020_v2` 是“PPO 观测滞后 200 ms”。这些字符串是已写入测量文件和协议的构建 ID；后续新增条件按 [`docs/style.md`](../../docs/style.md) 使用包含参数和单位的语义名称。

正式结果索引见 [`results/method_chains/failure_memory_regression/README.md`](../../results/method_chains/failure_memory_regression/README.md)：六方法冻结银行离线比较位于 `repair_exploit_v3_fullbank/`，六方法统计比较位于 `current_method_statistical_comparison.md`，覆盖槽留一物理种子簇评测位于 `memory_exploit_v3/`，后车状态滞后 800 ms 交互留出验证位于 `interaction_holdout_age080/`。已淘汰原型的结论在结果索引中概述，不进入活动代码。

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
