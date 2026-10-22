# 对比方法的实验产物

这里保存需要预训练的基线模型。随机、空间填充、历史排序、最近来源残差和无历史 GP 的查询结果，与主方法一起保存在统一比较目录。

| 目录 | 内容 |
|---|---|
| [fsbo/collision/models](fsbo/collision/models/) | 20 个 FSBO 碰撞适配模型及训练记录，覆盖全来源和严格留类别条件 |
| [pooled_ft/collision/models](pooled_ft/collision/models/) | 20 个混合预训练与微调模型及训练记录 |
| [候选库比较](../gcrt/collision/report/report.md) | 各候选库方法的发现数量、曲线、行为覆盖、统计和成本 |
| [连续搜索比较](../gcrt/collision/continuous/report/report.json) | 本方法、随机、Indago、SAMOTA、DeepHyperion 的独立比较 |

完整构建入口为 `python -B -m benchmark.build_collision_models complete`。模型仅从允许使用的历史和开发档案训练，目标测试结果不参与预训练或选点设置。

各方法的机制与适配说明见 [对比方法](../../methods/baselines/README.md)。ENS 和 RGPE 尚未实现，不包含在这些结果中；旧内部方法结果已在工程外归档。
