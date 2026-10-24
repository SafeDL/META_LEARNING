# 共享方法组件

本目录提供主方法和基线共同使用的数据表示与概率计算，避免各方法分别维护一套相同工具。

| 文件 | 职责 |
|---|---|
| [response.py](response.py) | 读取带来源与用途标记的测试档案，预测历史系统在新场景上的响应，学习场景表示 |
| [classification.py](classification.py) | 用碰撞或未碰撞标签更新概率分布；数值实现采用积分和矩匹配近似 |
| [gp_classifier.py](gp_classifier.py) | 高斯过程碰撞分类、深核模型及目标反馈微调 |
| [gp_joint.py](gp_joint.py) | 联合潜在高斯后验；支持RGPE固定核下逐折重新求解分类后验 |
| [gp_laplace.py](gp_laplace.py) | 一般高斯先验的收敛分类推断，供ScaML基线与新增来源条件修正共用；旧冻结实验的求解器保留原样 |
| [model_contract.py](model_contract.py) | 在复用模型前检查训练代码、软件版本和配置是否一致 |

这里的碰撞概率由实际碰撞标签学习。最小间距和 TTC 是辅助输入或监督，不代替碰撞标签。调用关系见 [主方法](../gcrt/README.md) 和 [对比方法](../baselines/README.md)，实验设置见 [项目说明](../../README.md)。
