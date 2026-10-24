# 核与来源权重的训练/推断一致性诊断

网络规模、历史/开发角色、1500次更新、学习率、训练支持集和辅助损失均与原训练保持一致。
原共享核检查点作为对照；新增来源独立核的全局和局部权重匹配训练。
比较360项配对监督验证：支持集16、查询集32，5个固定种子；其中1104曾用于开发检查点选择。
该结果仅回答预测与训练一致性问题，不替代主动搜索实验，不改变原阶段B门槛，也不启动C/D/E。

| 场景 | 训练模式 | 推断模式 | 碰撞 NLL | Brier |
|---|---|---|---:|---:|
| cut_in | shared_global | shared_global | 0.240215 | 0.062817 |
| cut_in | shared_global | source_global | 0.242028 | 0.063171 |
| cut_in | shared_global | source_local | 0.273974 | 0.071125 |
| cut_in | source_global | shared_global | 0.237202 | 0.062235 |
| cut_in | source_global | source_global | 0.233691 | 0.061575 |
| cut_in | source_global | source_local | 0.268565 | 0.069741 |
| cut_in | source_local | shared_global | 0.233873 | 0.061768 |
| cut_in | source_local | source_global | 0.235637 | 0.062082 |
| cut_in | source_local | source_local | 0.261784 | 0.068312 |
| cut_out | shared_global | shared_global | 0.154248 | 0.037120 |
| cut_out | shared_global | source_global | 0.156294 | 0.037692 |
| cut_out | shared_global | source_local | 0.163280 | 0.038173 |
| cut_out | source_global | shared_global | 0.156334 | 0.038056 |
| cut_out | source_global | source_global | 0.154972 | 0.037473 |
| cut_out | source_global | source_local | 0.162314 | 0.037988 |
| cut_out | source_local | shared_global | 0.152772 | 0.036709 |
| cut_out | source_local | source_global | 0.153189 | 0.036751 |
| cut_out | source_local | source_local | 0.160446 | 0.037284 |

NLL和Brier越小越好；全部推断组合、零碰撞开发对象与种子都保留。
