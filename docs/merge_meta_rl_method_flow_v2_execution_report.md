# Merge Meta-RL v2 执行报告

执行日期：2026-08-17  
环境：`conda run -n metadrive`，MetaDrive 0.4.3

## 结论

- 工程链路通过：严格 near-miss、Casebook v2、24D 动态观测、校准、训练、few-shot 和因果干预均已跑通。
- 36-episode validation calibration 通过，无需扩展到 60 条；test/OOD 未参与阈值选择。
- matched 20k Vanilla/Structure 训练均完成，但机制 gate **未通过**：正确 context 没有带来可辨识的动作变化或 query 增益。
- 按预注册规则停止，不进入 150k+ 正式训练，也不调整 PEARL 容量、KL、学习率或训练预算。

## 冻结资产

- Taskbook hash：`9fe5d461a53542b148c291e6758aa54316a3b61cdec06d7a70d5832150057473`
- Calibration hash：`8efed1246441c49b2c347fe23c6e7c5ebc1085b2c9410b10251e8b10426c4482`
- Metric schema：`spatiotemporal_near_miss_v2`
- Observation schema：`logical_merge_dynamic_obs_v1`（24D）
- Casebook schema：`logical_merge_casebook_v2`
- Integrity audit：9/9 pilot Task、全部静态契约通过；历史 Casebook 未混入 v2 审计。

Validation calibration 的有效阈值为：

| profile | arrival gap (s) | joint conflict distance (m) | pair distance (m) |
|---|---:|---:|---:|
| bottleneck | 0.140564 | 4.062757 | 3.114947 |
| lane drop | 0.114186 | 0.581659 | 5.134892 |

两个 profile 中 zero/random 的 near-miss 与 collision rate 均为 0；heuristic near-miss rate 均为 1/3，invalid rate 为 0。OOD 使用两个 validation profile 的逐项最严格组合。

Casebook 固定为 4 个 train Task、2 个 40m validation、2 个 48m parametric test 和 `y_merge_32`。每个 train Task 为 12 个 train cases；validation 为 4 support + 6 query；test/OOD 为 4 support + 8 query。所有 query 均完成 zero/random/heuristic hardness 筛选。

## 20k 机制 gate

Vanilla 和 Structure 均使用 seed 0，实际完成 20,019 环境步、560 次梯度更新。两组仅在 Scenario Encoder/conditional prior 是否启用上不同。

Deterministic actor mean 的标准 few-shot 结果：

| Task | Method | K=0 VCSR | K=4 VCSR | collision | invalid |
|---|---|---:|---:|---:|---:|
| bottleneck_48 | Vanilla | 0.000 | 0.000 | 0.375 | 0.000 |
| bottleneck_48 | Structure | 0.000 | 0.000 | 0.375 | 0.000 |
| lane_drop_48 | Vanilla | 0.125 | 0.125 | 0.375 | 0.000 |
| lane_drop_48 | Structure | 0.125 | 0.125 | 0.375 | 0.000 |
| y_merge_32 | Vanilla | 0.000 | 0.000 | 0.375 | 0.000 |
| y_merge_32 | Structure | 0.000 | 0.000 | 0.375 | 0.000 |

Structure checkpoint 的 fixed-state action audit 显示：

- 两个 48m Task：correct/prior action L2 均值约 `1e-5`–`7e-5`；correct/wrong 同量级或更小。
- `y_merge_32`：correct/prior action L2 均值约 `1e-5`–`7e-5`；correct/wrong 约 `0`–`1e-5`。
- 所有 K=1/2/4 的 correct-context 相对 K=0 和 wrong-context 的 VCSR paired gain 均为 0。
- 所有审计前后参数哈希一致，无梯度、无 replay/context 写入。

因此，当前证据支持“Actor 基本忽略 evidence-induced latent change”，不支持“Meta-RL 已发生有效 few-shot adaptation”。下一轮应优先增强 Task/case 的可辨识性和 context-action 因果信号，不应通过增加 PEARL 复杂度掩盖该问题。

## 产物

- 校准与冻结资产：`results/pearl_learning/merge_method_flow_pilot/v2_assets/`
- 1k smoke：`results/pearl_learning/merge_method_flow_pilot/v2_smoke/`
- matched 20k checkpoints：`results/pearl_learning/merge_method_flow_pilot/v2_20k/mechanism_gate/`
- 标准 few-shot：`results/pearl_learning/merge_method_flow_pilot/v2_20k/evaluations/`
- 48m 因果审计：`method_flow_v2_structure_20k/latent_context_audit_test_48m.json`
- Y-merge 因果审计：`method_flow_v2_structure_20k/latent_context_audit_y_merge_32.json`
