# Random 基线

[collision.py](collision.py)按算法种子生成候选随机顺序，先执行给定的共同初始化，再依次查询未执行候选，直到达到预算。每个候选最多查询一次。

调用入口为 `run(candidates, oracle, initial_indices=..., budget=200, seed=11)`。选择仅依赖公开候选和随机种子；查询返回的碰撞与其他反馈用于记录。

统一接入见 [experiments.pilot](../../../experiments/pilot.py)，结果见[论文报告](../../../artifacts/runs/joint_response/paper/reports/results.md)。
