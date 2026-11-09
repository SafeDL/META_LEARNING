# 实验运行记录

本目录按实验组织模型、协议、付费记录、报告和图片。

| 位置 | 内容 |
|---|---|
| [joint_response/paper/](joint_response/paper/) | 当前论文的主比较、模型选择及复用登记 |
| [joint_response/attribution/](joint_response/attribution/) | 共同反馈、固定历史记录对照和收益归因 |
| [joint_response/method_exploration/](joint_response/method_exploration/) | 方法比较、当前共享／独立核模型及构建证据 |
| [joint_response/](joint_response/) 下的 `fixed/`、`mechanisms/`、`sequential/`、`sensitivity/`、`scaling/` | 原配置的固定历史、机制、积累及规模记录 |
| [qualification/reports/](qualification/reports/)、[preparation/reports/](preparation/reports/) | 控制器资格和驾驶接口登记 |

`methods/` 或 `tasks/<ADS>/methods/` 保存逐轮结果和付费归档，`reports/` 保存协议、汇总与图片。顺序实验的各分支有独立的 `completed/` 和更新模型；`pilot/` 保存后续比较仍复用的首轮记录与基线分类器。

原联合核位于 `joint_response/models/`，当前共享／独立核位于 `joint_response/method_exploration/e2/models/`。模型旁的来源后验记录用于复现，`source_cache/` 是可重建的运行缓存。

付费归档通过 `execution_ref` 共享原始轨迹。报告入口见 [experiments/](../../experiments/README.md)，各组报告与图片见[产物索引](../README.md)。
