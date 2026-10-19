# 共用实验核验与标准搜索结果

本目录保存两个原始场景测试集的采集协议、数据核验和标准搜索轨迹。

| 位置 | 内容 |
|---|---|
| `pool_*/<控制器>/baselines/` | HGT、HR-GP、GP-EI、RF-BO、随机测试等已有基线轨迹 |
| [protocol.json](protocol.json) | 原始场景采集协议 |
| [bank_audit.json](bank_audit.json)、[interface_audit.json](interface_audit.json) | 实测数据与执行接口核验 |
| [baseline_protocol.json](baseline_protocol.json)、[baseline_input_audit.json](baseline_input_audit.json) | 基线设置与输入核验 |

实测响应与执行耗时集中在 [data/scenarios](../../data/scenarios/)；第三个场景测试集也位于该目录。CG-BRT 比较见 [结果表](../baselines/cg_brt/comparison.md)，本文方法见 [GCRT](../gcrt/README.md)。
