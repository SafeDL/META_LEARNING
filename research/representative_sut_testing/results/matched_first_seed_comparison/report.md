# 同选择种子的完整比较核查

全部比较使用六个固定 SUT、两个共享实测池、种子 11、200 次唯一查询。14 个冻结基线与已评估机制的 228 条曲线从实际索引和目标标签重算，预算、披露顺序和指标一致；没有重跑策略或新增物理执行。[结构化记录](summary.json)。

这些是开发结果，不是独立显著性确认。标准／原方法使用原六来源冻结划分，部分行为模型使用额外配置；反馈权限也按冻结协议保留。同信息的机制归因必须使用其相应对照。

Pareto 前沿由 Student-t 受保护、Student-t 无保护和 ADF 发现损失支持三个条目组成。EP 混合评分被 Student-t 受保护方法同时超过。因此，不能只通过超过冻结原方法或某一较弱组件宣布优势；下一阶段需同时核查较强条目的早期面积与终点召回。

`pooled_full_history` 的 Area 53.294%、Recall 77.752% 来自原六来源全部 2048 个历史坐标及其自有初始化。它是额外数据的工程参照，不能当作与当前 1638 个训练坐标完全匹配的组件消融，也不能删除这个已有较强结果。

| Method | Area % | Recall % | Mean F200 |
|---|---:|---:|---:|
| uniform_random | 5.433 | 11.256 | 26.58 |
| farthest_first | 6.383 | 12.076 | 25.75 |
| knn_history | 36.580 | 50.192 | 75.58 |
| gp_ucb | 15.504 | 26.662 | 50.58 |
| gp_ei | 41.396 | 68.859 | 111.33 |
| bop_elites | 29.892 | 44.696 | 80.75 |
| bas | 4.232 | 9.318 | 20.50 |
| rf_bo | 42.706 | 72.910 | 106.83 |
| ras_frt_uq | 52.332 | 74.429 | 113.83 |
| frozen_original | 52.837 | 76.885 | 118.92 |
| previous_best | 52.966 | 76.592 | 116.75 |
| risk_conditioned | 50.335 | 67.744 | 98.08 |
| behavior_posterior | 49.931 | 66.839 | 96.00 |
| matched_collision_gp | 52.262 | 74.936 | 114.08 |
| student_protected | 53.048 | 77.157 | 119.25 |
| student_unprotected | 53.025 | 77.473 | 118.50 |
| ep_mixed_score | 53.040 | 77.067 | 119.75 |
| adf_loss_supported | 53.052 | 76.915 | 119.08 |
| pooled_full_history | 53.294 | 77.752 | 121.50 |

## Budget ceiling and existing headroom

These ceilings require oracle ordering. The best method per SUT is selected in hindsight and is not an executable selector.

| SUT | Area ceiling % | Best existing Area % | Recall ceiling % | Best existing Recall % |
|---|---:|---:|---:|---:|
| fvdm_target | 77.500 | 76.485 | 100.000 | 100.000 |
| idm_ref | 12.690 | 12.599 | 25.253 | 25.126 |
| mcts_cv_ref_audit_v4 | 69.250 | 55.766 | 100.000 | 87.097 |
| mobil_ref_v2 | 82.500 | 82.423 | 100.000 | 100.000 |
| ppo_ref_v2 | 79.875 | 79.144 | 100.000 | 100.000 |
| vi_ttc_ref_audit_v4 | 31.959 | 28.043 | 63.601 | 56.598 |
