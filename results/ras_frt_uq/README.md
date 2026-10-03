# RAS-FRT-UQ 恢复证据

`model/` 保存 Git `cc7c1b32` 恢复的五个原模型、冻结参数、原 RAS 选例序列和[恢复记录](model/restoration.json)。`legacy_s01/` 保留原协议与评价出处，不作为另一个活动基线入口。

原 A/D 物理数据在[共享基准](../../benchmarks/s01/README.md)。D 含 116 次碰撞、33 个碰撞单元。原 RAS 五种子 F50/F100/F200 为 37.0/79.8/114.2，碰撞单元覆盖@200 为 100%。

[实现](../../methods/ras_frt_uq/README.md)保留原二值反馈和固定融合权重；全部新回放与[统一比较](../srd_tnp_bqd/comparison/README.md)汇合并核对原序列。原 D 曾用于融合开发，不能作为独立盲测证据。
