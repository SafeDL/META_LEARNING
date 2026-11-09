# 维护与核验记录

本目录保存文件迁移、来源身份和工程回归的凭据。

| 文件 | 用途 |
|---|---|
| [runs_cleanup.json](runs_cleanup.json) | 整理范围、文件迁移及结果一致性核验 |
| [result_compression.json](result_compression.json) | 结果压缩和轨迹引用迁移记录 |
| [runs_cleanup_regression.xml](runs_cleanup_regression.xml) | 最近一次完整工程回归结果 |
| [baseline_review.json](baseline_review.json) | 基线文献核对及比较范围 |
| [source_changes.json](source_changes.json) | 已审核的源码修改身份 |
| [source_reorganization.json](source_reorganization.json)、[reorganization.json](reorganization.json) | 源码、数据和策略权重的原始位置对应 |

`source_changes.json`、`source_reorganization.json` 和 `reorganization.json` 由 [source_registration.py](../../experiments/source_registration.py)读取，用于核对现有模型与执行档案。实验结果统一见[产物索引](../README.md)。
