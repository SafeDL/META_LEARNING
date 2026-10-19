# 方法代码

| 目录 | 用途 |
|---|---|
| [gcrt](gcrt/README.md) | 本文方法：收益校正风险测试 |
| [baselines](baselines/README.md) | LRT、CG-BRT、DART、HGT 等对比方法 |
| [common](common/README.md) | 历史先验、候选特征、概率计算和训练数据配置 |

本文方法的三个入口为：

```powershell
python -B -m methods.gcrt.build
python -B -m methods.gcrt.evaluation
python -B -m methods.gcrt.report
```

分别用于构建模型、运行实验和汇总结果。实验设置与风险反馈接口在 [benchmark](../benchmark/README.md)，实测记录在 [data](../data/README.md)，产物在 [results](../results/README.md)。技术流程和代码对应见 [GCRT 说明](gcrt/README.md)。
