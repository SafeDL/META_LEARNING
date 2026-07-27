# PEARL 逻辑场景元强化学习

本目录实现了面向逻辑汇入场景的 PEARL 元强化学习流程。模型在多个冻结的 `meta_train` 任务上训练；到达未见 `meta_test_logical` 任务后，仅用 K 个 support episode 推断任务后验，不进行梯度更新或参数微调，再在独立的 query case 上评估。

实现以 `configs/merge_family_pearl.yaml` 为唯一配置来源。它定义训练与评估参数、37 维无标签观测、几何与路由约束及安全指标。

## 实验对象与指标

- 每个 `logical_merge_task` 固化几何、对抗车和 SUT 路线、出生区间、优先权、冲突定义及内容哈希。
- `YMergeAdapter` 实现独立的 Y 型汇入 PGMap 几何；不复用 on-ramp 或随机地图。
- `logical_merge_obs` 为 37 维连续观测，包含基于完整 route polyline 的弧长距离、投影速度/加速度、路线进度和冲突角；不输入任务 ID、模板索引或其他标签。
- 一个 episode 仅在目标 adversary–SUT 接触或低 TTC 时记为 critical；若同时发生非目标碰撞、对抗车/SUT 出界或错误路线，则记为 invalid。主指标 `valid_critical_strict_rate` 是“critical 且有效”的 query episode 比率，不能直接等同于碰撞率。

## 冻结输入与可复现性

`build_taskbook.py` 在真实 MetaDrive 中解析全部地图后写出 taskbook 和 casebook。taskbook 的 provenance 文件保存 taskbook/casebook/几何哈希；后续训练和评估均校验这些冻结输入。评估还校验 checkpoint 的 taskbook 哈希，并在开始时恢复 checkpoint 的随机数状态。

每个任务的 casebook 包含彼此隔离的训练、验证、测试 support 和测试 query case。support episode 只用于更新 PEARL 的潜变量后验；query case 从未参与该任务的后验构建。

## 推荐流程

以下命令把新实验写入临时输出目录，避免覆盖已保留的最终结果。`<...>` 均应替换为实际路径。

```powershell
# 1. 解析真实地图，冻结 taskbook 与 casebook
conda run -n metadrive python -m pearl_learning.scripts.build_taskbook `
  --config pearl_learning/configs/merge_family_pearl.yaml `
  --output <taskbook-dir>

# 2. 正式训练前：拓扑、完整性与任务异质性审计
conda run -n metadrive python -m pearl_learning.scripts.audit_topologies `
  --config pearl_learning/configs/merge_family_pearl.yaml `
  --taskbook <taskbook-dir> --casebook-root <casebook-root> `
  --output <topology-audit-dir>

conda run -n metadrive python -m pearl_learning.scripts.audit_integrity `
  --config pearl_learning/configs/merge_family_pearl.yaml `
  --taskbook <taskbook-dir> --casebook-root <casebook-root> `
  --output <integrity-audit.json>
```

正式 PEARL 训练还需要在相同冻结输入上完成所需 SAC 基线，随后运行 `audit_task_heterogeneity.py` 和 `build_formal_gate.py` 生成 gate manifest。这样做是为了确认任务确有差异，并把训练前提与对应 taskbook 绑定。`--smoke` 训练不需要 gate，仅用于快速链路检查。

```powershell
# 3. 正式 PEARL 训练；若仅检查链路，可添加 --smoke
conda run -n metadrive python -m pearl_learning.scripts.train_pearl `
  --config pearl_learning/configs/merge_family_pearl.yaml `
  --taskbook <taskbook-dir> --casebook-root <casebook-root> `
  --gate-manifest <formal-gate.json> --seed 0 --run-name pearl `
  --max-env-steps <steps> --output-root <output-root>

# 4. 无梯度 few-shot 评估
conda run -n metadrive python -m pearl_learning.scripts.evaluate_fewshot `
  --config pearl_learning/configs/merge_family_pearl.yaml `
  --checkpoint <output-root/models/pearl/best_model.pt> `
  --taskbook <taskbook-dir> --casebook-root <casebook-root> `
  --split meta_test_logical --run-name pearl_fewshot `
  --output-root <output-root>
```

训练默认只保留 `best_model.pt`、其 manifest、解析后的配置和轻量级 `training_summary.json`；只有显式传入 `--checkpoint-interval-steps` 才额外写出可恢复的 `last_model.pt`。few-shot 命令输出为 `<output-root>/evaluations/<split>/<run-name>/metrics.json`，仅保留逐任务/逐 K 的汇总指标和 support 环境步数，不保存逐轨迹记录。

## 对比基线

`run_baselines.py` 在同一 taskbook、casebook 和指标下运行：per-task SAC、跨任务策略矩阵、拓扑条件 pooled SAC、scratch SAC、pooled fine-tune SAC、oracle task-conditioned SAC，以及 PEARL no-context 消融。长预算基线回答最终性能上限；它们不应与 few-shot PEARL 直接作样本效率比较。

对于公平的少样本比较，请使用 `run_equal_budget_analysis.py`：它令 scratch SAC 和 pooled fine-tune SAC 在每个新任务上获得与 PEARL 前 K 个 support episode 相同的累计环境步数，并以相同的冻结 query case 评估。默认仅写出 `equal_budget_summary.json`；如需保留逐 query 记录和断点续跑状态，才显式添加 `--detailed-output`。

## 目录

- `configs/merge_family_pearl.yaml`：唯一实验配置。
- `src/`：任务定义、MetaDrive 环境、PEARL、评估、指标、检查点和基线实现。
- `scripts/`：冻结输入、审计、正式训练、few-shot 评估及基线比较入口。
- `tests/test_contract.py`：冻结输入、观测、无梯度适应和精简结果结构的契约测试。

最终可复核模型和报告性结果位于 [`../results/pearl_learning/`](../results/pearl_learning/README.md)。
