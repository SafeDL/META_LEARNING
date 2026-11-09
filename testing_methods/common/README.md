# 共享响应与推断工具

本目录提供主方法和基线共用的数据结构与概率计算。

| 文件 | 职责 |
|---|---|
| [response.py](response.py) | 定义 `ResponseArchive`，计算档案指纹、初始来源权重和历史响应预测 |
| [gp_classifier.py](gp_classifier.py) | 定义 Matérn 核及可选特征网络 |
| [classification.py](classification.py) | 通过 Gauss–Hermite 求积计算碰撞概率及导数 |
| [gp_laplace.py](gp_laplace.py) | 计算 Bernoulli 观测的 Laplace 后验 |

响应档案保留场景、实际碰撞、安全量、RSS 及执行来源。`shared_domain` 核对各来源的参数定义，`source_weights` 提供等权初始化；目标反馈驱动的权重更新位于 [history_transfer/inference.py](../history_transfer/inference.py)。

联合响应的训练和适配见[主方法](../history_transfer/README.md)，独立 NumPy 推断参考见 [inference_reference.py](../../experiments/inference_reference.py)。
