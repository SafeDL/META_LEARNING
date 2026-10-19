# 实验结果与模型

| 目录 | 内容 |
|---|---|
| [gcrt](gcrt/README.md) | 本文方法的模型、36 条完整轨迹、停止评价、周期更新与成本报告 |
| [baselines](baselines/README.md) | LRT、CG-BRT、DART、HGT 和响应基线的模型与正式结果 |
| [benchmark](benchmark/README.md) | 共用实验协议、数据核验与标准搜索轨迹 |

先看 [GCRT 完整报告](gcrt/comparison.md)。按类基线比较见 [LRT 按类实验](baselines/lrt/by_family/README.md)；两类共同使用 200 次预算的结果见 [LRT 联合实验](baselines/lrt/README.md)。

实测场景、风险和执行耗时统一放在 [data](../data/README.md)。本目录保存权重、训练记录、标签、查询轨迹和报告。原始源码快照保留当时的目录名称，核验程序负责对应到当前位置。

运行说明见 [methods](../methods/README.md)。
