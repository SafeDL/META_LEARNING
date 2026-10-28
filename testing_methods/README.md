# 测试场景选择方法

本目录实现“下一次测试哪个交通场景”的算法。方法接收公开场景参数，以及允许使用的历史记录和已查询反馈，在给定预算内寻找碰撞场景。被测车辆的驾驶控制器位于 [controllers](../controllers/README.md)。

| 位置 | 用途 |
|---|---|
| [history_transfer](history_transfer/README.md) | 本文方法：利用历史系统响应和目标反馈选择场景 |
| [baselines](baselines/README.md) | Random、RF-EI、SAMOTA、ScenarioFuzz、配对系统主动评估和 Indago |
| [common](common/README.md) | 测试档案、概率模型、分类推断和模型加载工具 |
| [selection.py](selection.py) | 检查共同初始化和查询预算，统一并列候选的选择规则 |

原间距实验的方法描述按 [问题定义](paper_section.md)、[主方法](history_transfer/paper_section.md)、[概率模型与推断](common/paper_section.md) 的顺序阅读。当前RSS辅助响应的定义见 [安全度量说明](../docs/rss.md)，训练入口见 [history_transfer](history_transfer/README.md)。实验设置、数据权限和评价指标见 [实验运行说明](../experiments/README.md)。
