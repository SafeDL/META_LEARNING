# 历史风险核与模型

本目录保存 HGT 及其他方法复用的基础风险核。原始历史记录统一位于 [data/history](../../../data/history/)。

| 位置 | 内容 |
|---|---|
| [models](models/) | 五个随机种子的风险核、RAS 模型及训练记录 |
| [protocol.json](protocol.json) | 历史采集和模型训练设置 |

历史记录来自六种 IDM／FVDM 配置，包含切入、前车紧急制动各 1024 个场景，共 12288 条响应。联合训练按 1638／410 划分训练与验证；按功能场景训练时为每类 819／205。

GCRT 按类模型中的基础核位于 `results/gcrt/cutin/kernel/`、`results/gcrt/brake/kernel/`。实现见 [HGT](../../../methods/baselines/hgt/README.md)，比较结果见 [CG-BRT](../cg_brt/comparison.md) 和 [LRT 按类实验](../lrt/by_family/README.md)。
