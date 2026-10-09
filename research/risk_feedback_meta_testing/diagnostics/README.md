# 锁定模型的验证诊断

本目录只读取既有模型和训练／验证数据，不改变主方法、训练参数、开发回放或物理数据。

`validation_contexts.py` 在 CPU 单线程运行。使用 12 个验证配置、5 个模型种子、6 种支持规模及 2 种查询方式，共 720 个上下文、两种锁定碰撞读出。每个配置／种子使用同一未查询场景集合；两种支持集按规模嵌套，查询集排除两种方式的最大支持集。支持规模为零时复用相同推断结果。池选择在各控制器内均衡。

入口：`python -B -m research.risk_feedback_meta_testing.diagnostics.validation_contexts`。

输出为上级 `results/validation_context_diagnostic.json` 与 `results/validation_context_diagnostic.md`。排名指标只针对固定的 256 个验证查询场景，不是完整池策略回放，也不作为独立显著性证据。

`posterior_contexts.py` 复用其中 10/50/150 支持规模的 360 个上下文，检查后验熵、控制器后验质量及未查询风险预测。只在 CPU 运行；目标参数与查询风险只作为离线诊断真值，推断仍只读支持风险。入口为 `python -B -m research.risk_feedback_meta_testing.diagnostics.posterior_contexts`，输出为上级 `results/posterior_context_diagnostic.json` 与同名 Markdown。

600 次开发回放完成并执行上级 `summarize` 后，运行 `python -B -m research.risk_feedback_meta_testing.diagnostics.development_comparison`，生成五个新对照与七个冻结参照的完整同池表。冻结参照的历史信息差别与 RAS 的碰撞反馈差别单列，配对区间只作开发描述。
