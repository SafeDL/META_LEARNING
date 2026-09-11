# DIVA-Mine Cut-in 技术设计

DIVA-Mine 在固定的 Cut-in 物理合同中学习多个 source SUT 的连续漏洞响应，并用少量目标系统诊断测试个性化排序。SUT 身份只用于执行、分组和可追溯记录，不是模型输入。

场景合同见 [DIVA_Mine_CutIn_Scenario_Contract.md](DIVA_Mine_CutIn_Scenario_Contract.md)。每个场景由左右切入候选和五个连续物理参数唯一确定：初始车间距离、主车初始速度、相对速度、切入开始位置和切入路径长度。红车遵循给定速度和空间轨迹，不读取主车状态。

## 评价与学习信号

正式评价固定为：有效碰撞 `1.0`、有效 critical near-miss `0.5`、其他 `0.0`。这也是后续 B=20 挖掘的唯一成功指标。

每个完整可评估 episode 还保存 `vulnerability_response`，只用于 source prior、后验更新和 acquisition。它由 challenge phase 的最小 TTC 与最小距离计算：无事件样本最多 `0.74`，有效 near-miss 至少 `0.75`，有效碰撞为 `1.0`。invalid 与 censored 样本没有该学习标签，也不会更新 latent posterior。

## 冻结的 Stage A v2

`mvr/configs/diva_cutin.yaml` 定义独立的 source study domain `diva_source_interaction_v2`：

| 参数 | 范围 |
|---|---:|
| 初始车间距离 | 7–13 m |
| 主车初始速度 | 8–12 m/s |
| 相对速度 | −2.5–0.5 m/s |
| 切入开始位置 | 35–250 m |
| 切入路径长度 | 45–105 m |

该域在每个候选侧使用 32 个 Sobol anchors。四个 source SUT 对同一 64 个 concrete anchors 执行，保证 response matrix 对齐。所有 v2 文件使用 `diva_mine_cutin_constant_speed_physical_v2` schema；旧 schema 被读取器拒绝。

执行顺序：

```powershell
conda run -n metadrive python -m mvr.scripts.build_diva_cutin_casebook --config mvr/configs/diva_cutin.yaml --output results/diva/cutin_g01/source_casebook_v2.json
conda run -n metadrive python -m mvr.scripts.collect_diva_source_bank --config mvr/configs/diva_cutin.yaml --casebook results/diva/cutin_g01/source_casebook_v2.json --output results/diva/cutin_g01/source_observations_v2.jsonl
conda run -n metadrive python -m mvr.scripts.analyze_diva_prior --config mvr/configs/diva_cutin.yaml --source results/diva/cutin_g01/source_observations_v2.jsonl --output results/diva/cutin_g01/source_analysis_v2.json
conda run -n metadrive python -m mvr.scripts.fit_diva_prior --config mvr/configs/diva_cutin.yaml --source results/diva/cutin_g01/source_observations_v2.jsonl --output results/diva/cutin_g01/prior_v2.pt
conda run -n metadrive python -m mvr.scripts.evaluate_diva_source_loso --config mvr/configs/diva_cutin.yaml --source results/diva/cutin_g01/source_observations_v2.jsonl --output results/diva/cutin_g01/source_loso_g1_v2.json
```

Stage A 的结果图与一条可复现的 source 碰撞重放可由下列命令生成。重放从
source bank 选择一个真实的碰撞记录，以同一设计、SUT 和随机种子再次执行；它是
展示产物，不计入 source 或 target 的实验预算。

```powershell
conda run -n metadrive python -m mvr.scripts.render_diva_stage_a --observations results/diva/cutin_g01/source_observations_v2.jsonl --loso results/diva/cutin_g01/source_loso_g1_v2.json --output-dir results/diva/cutin_g01/visualizations
```

G0 检查 source response 的可用率、变化范围、跨 SUT 分歧和低秩解释方差；formal event rate 仅作描述。G1 在 source-only LOSO 中比较 K=0/1/2/4 的 shared prior、随机支持、诊断支持和 highest-shared-risk 支持。只有 G0 和 G1 同时通过，`evaluate_diva_cutin` 才允许创建 validation/test SUT 环境。

当前 v2 证据：256 次 source 调用全部完整可评估；G0 通过，rank-2 解释方差为 `0.977`；G1 通过，K=4 diagnostic 的 NDCG@8 为 `0.968`，相对随机支持的 bootstrap 95% CI 为 `[0.0399, 0.0825]`。这些结果支持进入 validation，但尚不构成 unseen target 的正式挖掘收益结论。
