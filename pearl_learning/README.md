# PEARL 逻辑场景元强化学习

Stage 2 现在只执行经过冻结几何、显式角色路线和全任务 topology audit 验证的逻辑汇入任务。它不修改 `sac_scenario_mining/`，且旧的 Stage 2 artifact 与当前 schema 不兼容。

`configs/merge_family_pearl.yaml` 是唯一配置文件，包含训练、评估和全部物理几何定义；运行环境需要 PyYAML 读取该配置。

- `logical_merge_task` 明确保存物理 `geometry_id`、完整 adversary/SUT 路线、出生区间、优先级、冲突定义及 hash。
- `logical_merge_obs` 保持 37 维，但以完整 route polyline 计算弧长距离、投影速度/加速度、路线进度和冲突角；不输入 metadata 标签。
- `y_merge` 由独立 `YMergeAdapter` 的 2→1 PGMap Merge 几何实现；不再复用 on-ramp 或随机地图。
- 目标碰撞使用物理 contact pair 或 OBB，压线与 wrong-route 分开记录；horizon truncation 不会关闭 SAC bootstrap。
- 训练、few-shot 评估和基线命令均要求加载已冻结的 taskbook/casebook，正式 PEARL 训练还要求完整 gate manifest。

## 工作流

所有示例均可将产物放入临时目录，避免覆盖版本化实验结果。

```powershell
# 1. 在真实 MetaDrive 中解析全部地图、冻结 taskbook/casebook
conda run -n metadrive python -m pearl_learning.scripts.build_taskbook `
  --config pearl_learning/configs/merge_family_pearl.yaml --output <taskbook-dir>

# 2. 全任务物理与静态完整性审计
conda run -n metadrive python -m pearl_learning.scripts.audit_topologies `
  --config pearl_learning/configs/merge_family_pearl.yaml --taskbook <taskbook-dir> `
  --casebook-root <taskbook-dir/..> --output <audit-dir>
conda run -n metadrive python -m pearl_learning.scripts.audit_integrity `
  --config pearl_learning/configs/merge_family_pearl.yaml --taskbook <taskbook-dir> `
  --casebook-root <taskbook-dir/..> --output <integrity.json>

# 3. smoke 训练；正式训练移除 --smoke 并提供 complete gate
conda run -n metadrive python -m pearl_learning.scripts.train_pearl `
  --config pearl_learning/configs/merge_family_pearl.yaml --taskbook <taskbook-dir> `
  --casebook-root <taskbook-dir/..> --seed 0 --max-env-steps 1000 --run-name smoke --smoke `
  --output-root <temporary-output-root>

# 4. checkpoint-hash 隔离的 few-shot 评估
conda run -n metadrive python -m pearl_learning.scripts.evaluate_fewshot `
  --config pearl_learning/configs/merge_family_pearl.yaml --checkpoint <checkpoint.pt> `
  --taskbook <taskbook-dir> --casebook-root <taskbook-dir/..> --split meta_test_logical `
  --run-name run_001 --output-root <output-root>
```

`run_baselines.py` 提供 per-task、pooled、scratch、fine-tune、oracle、cross-task matrix 和 PEARL no-context 的统一 baseline manifest；`build_formal_gate.py` 只有在全任务审计和全部正式 baseline 完成后才生成可用于正式 PEARL 训练的 gate。
