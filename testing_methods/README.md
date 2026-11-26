# 测试场景选择方法

本目录实现有限候选场景库下的预算受限失效发现与版本回归排序算法。各方法仅接收公开候选场景参数、获准开放的历史档案以及当前任务已付费揭示的反馈，输出选例顺序、逐次查询前风险预测及内部状态记录。

| 目录 / 文件 | 职责 |
|---|---|
| [history_transfer/](history_transfer/README.md) | 本文方法（`Ours method`）：共享–专属双输出深度核联合响应模型、多源独立后验、六标量十步在线目标适配、空间局部可信度加权、失配门控不确定性采集、倾向去偏归档与 L2-SP 锚定／LOSO 门控持续训练 |
| [baselines/](baselines/README.md) | 领域对比基线：`Target-only GP`、`Pooled historical GP`、`Random`、`ScenarioFuzz`、配对系统主动评估，以及同谱系回归测试基线 `DETOUR`、`Birchler`、`SDC-Scissor` 与交通属性 `SPECTRE` |
| [common/response.py](common/response.py) | 定义 `ResponseArchive`，计算档案指纹、等权初始来源权重与历史核平滑风险代理预测 |
| [common/gp_classifier.py](common/gp_classifier.py)、[common/classification.py](common/classification.py)、[common/gp_laplace.py](common/gp_laplace.py) | 本文方法与 GP 基线共用的 Matérn 核、32 节点 Gauss–Hermite 碰撞概率求积与阻尼 Newton–Laplace 众数求解 |
| [selection.py](selection.py) | 校验共同初始化候选索引与查询预算合法性 |

各方法通过 [QueryLedger](../experiments/query.py) 按次付费获取目标反馈。被测驾驶控制器位于 [controllers/](../controllers/README.md)，统一实验调度与独立复算入口见 [experiments/](../experiments/README.md)。
