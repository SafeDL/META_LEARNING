# RAS-FRT-UQ

RAS-FRT-UQ 是当前统一实验的历史迁移基线，沿用原四维网络、二值碰撞反馈和融合选例公式。

当前默认主链见 [history_guided_testing](../history_guided_testing/README.md)。[unified.py](unified.py) 将 RAS 加入两类场景统一比较：与主方法共享六源历史库、训练/验证坐标、目标库、五个种子和 200 次场景预算。切入、急刹各训练一个原四维结构的响应编码器，每个模型有六个二值碰撞预测头；跨类型相似度与物理协方差置零。

响应编码器使用 Adam（学习率 0.001）、批量 128，训练 300 轮，每 10 轮按历史验证 BCE 选择检查点。相似度尺度 0.3/0.25、校正正则项 0.1 沿用原历史实验固定设置，覆盖权重 0.2、新参数单元奖励 0.2、信息增益权重 0.05 和原融合更新保持不变。在线仅接收已查询场景的二值碰撞结果；其他统一基线接收连续风险。训练和选型均不读取最终目标标签。

统一实验模型保存在 [models/ras_frt_uq](../../results/history_guided_testing/models/ras_frt_uq/)，比较见 [统一报告](../../results/history_guided_testing/README.md)。原 S01 的五源模型、基准数据和独立结果已完整移入可恢复压缩包，见[恢复说明](../../archives/README.md)。默认统一入口会完成 RAS 训练、选例和统计：

```powershell
conda activate metadrive
python -B -m methods.history_guided_testing.run
```
