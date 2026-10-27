# 随机场景选择

[collision.py](collision.py) 实现 Random 基线：使用固定算法种子生成候选随机顺序，先执行共同的8次初始化，再查询剩余候选，直到达到128次预算。每个候选最多查询一次。

该方法只使用公开候选和随机种子选择场景。全来源与严格留家族条件使用相同的实际运行，评估端通过引用复用结果并核对预算。

实验设置与执行入口见 [experiments](../../../experiments/README.md)，结果见 [独立验证报告](../../../experiment_outputs/benchmark/collision/discovery/independent_validation/report.md)。
