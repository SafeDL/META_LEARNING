# Gate 3 Vanilla PEARL 因果链报告（Round 1 & Round 2）

日期：2026-08-18。该记录是机制门禁证据，不是论文性能结果。

## 配置与协议

- 任务：物理完全匹配的 `adversary_first` / `sut_first` 对；仅冻结的 conflict-entry order 不同。
- 算法：Vanilla PEARL（Dense Actor，unit-normal prior，Scenario Encoder / MoE OFF），latent dim 5。
- 训练：每轮 2 tasks × 20k 环境步 × seed 1，meta-batch 2。只验证机制，不追求性能。
- 案例簿：`order_boundary_fewshot_screened_v1` —— train 6（Gate 1 筛后可行子集）+ support 4 + query 4（query 经 oracle 筛选）。
- strict 指标：`logical_order_spatiotemporal_near_miss_v3`，calibration 与 Gate 1/1B/2 同一 manifest。
- 训练入口对 v2/v3 统一强制 `--critical-thresholds`（`resolve_calibration`）。

证据目录：

```text
Round 1 配置   pearl_learning/configs/merge_method_flow_gate3_vanilla_pearl.yaml
Round 2 配置   pearl_learning/configs/merge_method_flow_gate3_film_critic.yaml
案例簿         results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/mechanism_assets_screened/casebooks/
Round 1 训练   results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/mechanism_gate/gate3_vanilla_pearl_20k_seed1/
Round 2 训练   results/pearl_learning/merge_method_flow_gate3_film_critic/mechanism_gate/gate3_film_critic_20k_seed1/
oracle 审计    results/pearl_learning/merge_method_flow_gate3_vanilla_pearl/gate_3_query_oracle_screened/
Round 2 audit  results/pearl_learning/merge_method_flow_gate3_film_critic/gate_3_causal_audit/
```

## Round 1（Dense Critic）：fail，定位在 Stage B 且指向 Critic

- Stage A pass：‖μ_c − μ_w‖₂ ≈ 7.7；Stage B fail：action L2 ≈ 0.030；Stage C blocked_by_stage_b。
- Stage-B 诊断：cos(z_c,z_w)=0.9972；pre-tanh D_raw 0.09 → tanh 0.03（非纯饱和）；插值曲线水平；**Q-grid argmax 距离 0.0 且 Q 沿动作网格 100% 单调递增** → 情况 2：Critic 对 correct/wrong latent 不敏感。
- 附加发现：Round 1 的 4 个 query conditions 未筛选 —— oracle 审计 feasibility fail（sut_first 对角优势 −0.25、可实现 0/4）。

## Casebook 修复（0 训练步）

8 个候选条件经两个单任务 SAC oracle 筛选，冻结 4 个聚合门禁通过者 → `order_boundary_fewshot_screened_v1`。oracle feasibility **pass**（对角优势 +0.5/+0.75，可实现 2/4、3/4）。

## Round 2（Latent-FiLM Critic，分支 B）：fail，但 policy 质量大幅提升

判定（决策 shot K=4，顺序门禁）：

```text
Stage A  PASS；Stage B  FAIL；Stage C  blocked_by_stage_b；R_policy ≈ 0.011 / 0.003
```

| 指标 | Round 1 | Round 2 |
| --- | --- | --- |
| adversary_first correct VCSR | 0.00 | **1.00 (4/4)** |
| sut_first prior VCSR | 0.00 | **0.75 (3/4)** |
| q_loss 收敛 | ~63k | **~2.6k** |
| critic_latent_gradient_norm | — | 2.6 → 285 |
| Q-grid argmax 距离 | 0.0 | **仍 0.0**（81 点网格全撞 +1.0 边界） |
| correct/wrong action L2 | 0.030 | 0.011 / 0.003 |
| cos(z_c, z_w) | 0.9972 | 0.9997-0.99995 |

## 结论

```text
Gate 3 两轮 status = fail；passed_stages = [stage_a]（两轮一致）
```

1. **Critic 分支 B 判据未达成**：FiLM Critic 20k 后 Q audit 仍显示 a_c^Q* = a_w^Q*，task-dependent Q 未在 correct/wrong 方向形成；按判据 Actor 不可能自然分化。
2. **真实进展**：FiLM Critic 显著改善控制质量（adversary_first correct 4/4 VCSR），并确认 policy 已学会"有 context vs 无 context"的行为切换 —— 只是没有学会"哪个 task 的 context"。
3. **深层瓶颈浮现**：两个 posterior 几乎共线（R_sep 0.07-0.14），task 信息只是幅值差；Stage A 的 L2 阈值（0.5）不足以保证分离方向是 task-discriminative 的。

## 下一步（待定，未实施；只选一个方向）

```text
1) 加强 posterior 的 task-discriminative 分离质量
2) FiLM Critic 延续或 FiLM Actor + FiLM Critic 组合
3) 重新审视 support evidence 构造
```

在 matched query cases 上跑出完整链条前，不进入 Gate 4、不恢复真实物理 Merge Task、不做多模型并行搜索。
