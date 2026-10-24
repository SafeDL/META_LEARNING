# 元训练支持集规模诊断

保持网络、1500次优化更新、学习率、种子和历史档案不变。
训练支持集由4/8/16扩展到8/16/32/64/128；每次更新的监督点数与计算量因此增加，成本单独记录。
配对验证均使用128个支持点和32个查询点；4个开发家族等权，5个固定种子，完整160项。
只回答预测与支持集规模问题，不改变原门槛，也不证明碰撞发现收益。

| 场景 | 模式 | 训练支持集 | 碰撞 NLL | Brier |
|---|---|---|---:|---:|
| cut_in | source_global | short | 0.173479 | 0.047884 |
| cut_in | source_global | wide | 0.150543 | 0.042408 |
| cut_in | source_local | short | 0.175555 | 0.048067 |
| cut_in | source_local | wide | 0.159398 | 0.044786 |
| cut_out | source_global | short | 0.127520 | 0.033719 |
| cut_out | source_global | wide | 0.119759 | 0.031534 |
| cut_out | source_local | short | 0.126467 | 0.033135 |
| cut_out | source_local | wide | 0.123749 | 0.032868 |
