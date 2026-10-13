# RAS-FRT-UQ 迁移基线

该基线根据历史系统的碰撞响应学习场景相似度，再融合目标反馈、覆盖度和不确定性选例。在线只接收已测试场景的碰撞标签；[CG-BRT](../concordance_gated_bayesian_risk_transfer/README.md) 则接收连续风险。

[unified.py](unified.py) 使用六来源历史库和固定训练划分。切入、急刹分别训练一个四维输入编码器，每个模型有六个碰撞预测头；两类场景间的相似度与协方差设为零。已有模型保存在[检查点目录](../../results/history_guided_testing/models/ras_frt_uq/)。

查看已有结果：

```powershell
conda activate metadrive
python -B -m experiments.concordance_gated_bayesian_risk_transfer.compare
```

该命令只重算比较，不训练模型或运行选例。结果与反馈差异见[比较表](../../results/concordance_gated_bayesian_risk_transfer/comparison.md)。
