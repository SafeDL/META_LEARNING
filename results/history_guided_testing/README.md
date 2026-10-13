# 历史数据与模型

本目录保存 CG-BRT 和基线实际使用的历史输入。

| 位置 | 内容 |
|---|---|
| [history/responses.npz](history/responses.npz) | 六来源在 2048 个共享坐标上的 12288 条响应 |
| [history/scenarios.json](history/scenarios.json) | 切入与前车急刹场景 |
| [models/](models/) | 五种子风险核、RAS 模型及训练记录 |
| [protocol.json](protocol.json) | 历史采集与模型训练设置 |

历史坐标固定按 1638/410 划分训练与验证。CG-BRT 只使用历史风险 `R`；当前六系统、两池、预算 200 的设置以[现行协议](../concordance_gated_bayesian_risk_transfer/protocol.json)为准。
