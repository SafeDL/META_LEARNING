# 方法代码

| 目录 | 用途 |
|---|---|
| [gcrt](gcrt/README.md) | 本发明方法：收益校正风险测试（GCRT） |
| [baselines](baselines/README.md) | LRT、CG-BRT、DART、HGT 等对比方法 |
| [common](common/README.md) | 历史先验、候选特征、概率计算和训练数据配置 |

在项目根目录、`metadrive` 环境中运行：

```powershell
python -B -m methods.gcrt.build
python -B -m methods.gcrt.evaluation
python -B -m methods.gcrt.report
```

以上命令分别构建模型、运行评价和汇总结果。发现效果、测试效率与分布校正实验的入口见 [GCRT 说明](gcrt/README.md)。

实验设置与风险反馈接口见 [benchmark](../benchmark/README.md)，实测记录见 [data](../data/README.md)，模型与报告见 [results](../results/README.md)。
