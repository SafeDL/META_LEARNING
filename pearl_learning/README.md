# PEARL 逻辑场景元强化学习

`pearl_learning` 在 MetaDrive 0.4.3 中实现 PEARL-SAC：RL 控制一辆对抗车，固定 IDM SUT；从 support episode 推断任务潜变量 `z`，再以无梯度的 query episode 评估 few-shot 适配。它不修改 `sac_scenario_mining/`。

## 保留的核心设计

- 四种 merge-like 逻辑场景：`on_ramp_merge`、`lane_drop_merge`、`bottleneck_merge`、`y_merge`；`y_merge` 仅用于 held-out logical test。
- task 仅保存 ID、split、逻辑类型、地图/冲突配置和随机种子；case 仅保存 ID、随机种子、对抗车速度和到达偏移。
- SUT 的速度和换道策略固定在配置中，不被 task 或 case 随机化。
- `logical_merge_obs` 是无任务标签的 37 维双车观测：两车状态、两车交互和道路拓扑。
- 严格有效关键场景定义为：目标碰撞或低 TTC，且没有非目标碰撞、出界或 wrong-route。
- 每个 task 具有独立 replay buffer；query 不写入 context，也不更新网络权重。
- 训练周期性验证并保存 `best_model.pt`；评估保存 top-K 场景的参数化任务、case、动作和指标，可脱离策略回放。

冲突框架由 adversary 与 SUT 实际导航车道的最近几何交会区域计算。当前不进行 conflict-centric 观测下的重新训练。

## 环境

在仓库根目录使用：

```powershell
conda run -n metadrive python -c "import metadrive; print(metadrive.__version__)"
```

配置文件名为 `.yaml`，内容为 JSON，由标准库读取。

## 常用命令

```powershell
# 冻结 taskbook/casebook，并在真实 MetaDrive 中检查四类代表场景
conda run -n metadrive python -m pearl_learning.scripts.build_taskbook --config pearl_learning/configs/merge_family_pearl.yaml
conda run -n metadrive python -m pearl_learning.scripts.audit_topologies --config pearl_learning/configs/merge_family_pearl.yaml

# 确认 Stage 1 SAC 兼容性
conda run -n metadrive python -m pearl_learning.scripts.verify_stage1_compatibility

# 连通性训练；正式实验删除 --smoke
conda run -n metadrive python -m pearl_learning.scripts.train_pearl --config pearl_learning/configs/merge_family_pearl.yaml --seed 0 --max-env-steps 1000 --run-name pearl_smoke --smoke

# few-shot 评估与 top-K 回放
conda run -n metadrive python -m pearl_learning.scripts.evaluate_fewshot --config pearl_learning/configs/merge_family_pearl.yaml --checkpoint <run>/best_model.pt --split meta_test_logical --query-cases 20
conda run -n metadrive python -m pearl_learning.scripts.replay --manifest <evaluation>/<task_id>/shot_5/critical_scenarios/rank_001
```

旧的 56 维 smoke checkpoint 已删除，不能与当前实现混用。
