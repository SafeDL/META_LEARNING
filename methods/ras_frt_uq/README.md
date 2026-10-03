# RAS-FRT-UQ

作为本文基线恢复原 S01 方法：四维输入、五个历史 IDM 二值响应头、响应相似度局部修正、目标残差不确定性与覆盖融合。入口为 [s01.py](s01.py)，由 SRD-TNP-BQD 的统一 experiment 调用。

五个模型和固定参数直接恢复自 Git `cc7c1b32`，见[模型恢复记录](../../results/ras_frt_uq/model/restoration.json)。保持原二值碰撞反馈和融合权重，不重训为十一维连续风险方法。每次测量同时记录连续风险，供公共评价使用。

[统一比较](../../results/srd_tnp_bqd/comparison/README.md)核对五种子各 200 次选例与 Git 原序列一致。原 D 曾用于融合开发，属于同库开发证据。

这是本项目已有方法，不能声称有独立外部论文来源。不存在另一条训练/异质实验主链。
