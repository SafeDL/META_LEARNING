# 共享模型与数据工具

本目录提供主方法和部分基线共同使用的测试档案、概率模型和训练工具。

| 文件 | 用途 |
|---|---|
| [response.py](response.py) | 保存场景参数、碰撞标签、安全量和来源信息；提供历史响应预测与来源权重 |
| [classification.py](classification.py) | 将潜在高斯变量转换为碰撞概率，计算相关导数 |
| [gp_classifier.py](gp_classifier.py) | 高斯过程碰撞分类、深核和目标反馈适配 |
| [gp_laplace.py](gp_laplace.py) | 用 Laplace 近似计算分类后验 |

`source_weights` 对允许使用的历史来源等权初始化。控制方法类别用于划分实验，系统实现路线用于统计汇总；在线来源权重由目标反馈更新。

碰撞标签来自仿真的实际碰撞事件，最小间距和 TTC 按各方法的设计作为辅助响应。公共初始化与并列选择规则位于 [selection.py](../selection.py)。

数学定义、推断公式与参数设置见 [paper_section.md](paper_section.md)；主方法流程见 [history_transfer](../history_transfer/README.md)，实验设置见 [experiments](../../experiments/README.md)。
