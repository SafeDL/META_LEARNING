# 主程序流程验证

本目录只保存 PEARL 主流程的单种子 smoke 验证，不是正式性能实验，也不用于算法、消融或基线比较。

- 训练：seed 0，两个 `meta_train` 任务，一个 `meta_validation` 任务，实际完成 1,027 个环境步和 3 次梯度更新。
- 检查点：`smoke/main_flow_smoke/best_model.pt` 与可恢复的 `last_model.pt` 均已写入，二者对应步数均为 1,027。
- 少样本评估：加载 `best_model.pt` 后，在一个未见 `meta_test_logical` 任务上完成 K=0/1/2/5/10 的 support 后验适应和 2 个独立 query case 的评估；结果位于 `smoke/meta_test_logical/main_flow_smoke/metrics.json`。
- 不变性：评估结果记录 `no_gradient_adaptation=true`，且参数哈希在评估前后相同，表明适应阶段没有更新网络参数。

该输出只证明训练、保存、加载、少样本适应和评估链路可运行。由于预算很小且只使用一个种子，任何指标数值都不能解释为方法性能或样本效率结论。
