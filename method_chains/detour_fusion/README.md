# Mining–DETOUR 融合方法链

该目录独立维护“本文 Risk Mining 思路 × DETOUR 历史失效层次”的 highway-env
多功能场景实验。它不是本文基础链，也不是 DETOUR 论文的独立复现。

```text
detour_fusion/
  config.py       融合实验唯一配置所有者
  fusion.py       DETOUR 层次先验与 Mining 后验的融合逻辑
  experiment.py   LOSO、固定预算实验、统计和静态图
  replay.py       由融合方法实际选中场景的 GIF 回放
  tests/          融合核心、协议、回放和依赖边界测试
```

## 维护边界

```text
highway_env_benchmark/（本文基础能力） ─────┐
                                 ├─> method_chains/detour_fusion/
replications/detour_highway_env/ ┘
```

- 融合专属逻辑、参数、入口和测试放在本目录，正式结果统一放在
  `results/method_chains/detour_fusion/`。
- `highway_env_benchmark/` 只提供通用 Cut-in 环境、Mining 低秩先验和诊断原语。
- `replications/detour_highway_env/` 只提供 DETOUR 的树、检索和特征实现。
- 上述两个依赖目录不得反向导入本融合包；边界测试会阻止这种耦合。
- 若修改共享接口，应先分别运行基础方法、独立 DETOUR 和本目录测试。

## 固定实验

- 6 个 highway-env SUT，240 个场景，5 种功能机制各 48 个。
- Leave-one-SUT-out；总预算 B=50，其中目标诊断预算 K=10。
- Random 重复 20 次；其余方法为确定性运行。
- DETOUR 层次权重固定为 0.10，不针对正式结果调参。

正式结果中，DETOUR-Guided Risk Diagnosis 的关键事件召回率为 72.58%，纯 Mining
为 72.25%，仅历史 DETOUR 为 52.34%。融合相对纯 Mining 的增益很小，不能解释为
统计显著或普遍优势；相对仅历史层次的提升则说明目标诊断对跨功能迁移有价值。

## 三车 passing 稀有事件实验

`passing_experiment.py` 复用 AdaTE 场景重构得到的三车 passing 布局，但保持本
方法链的 LOSO 发现任务不变。三个代理模型只提供历史，三个 AV 才作为目标。
在 240 个场景、B=50 下，Random、DETOUR 和 Risk Mining 的临界召回率分别为
21.48%、73.15% 和 100%。B=20 时直接融合为 32.56%，纯 Risk Mining 为 31.52%；
到 B=50 三种 Mining 方案均饱和，因此新场景强化了 Risk Mining 的效果证据，但
仍只显示很小的 DETOUR 融合增量。

## 验证与重建

```powershell
conda run -n metadrive python -m pytest method_chains/detour_fusion/tests -q -p no:cacheprovider

conda run -n metadrive python -m method_chains.detour_fusion.experiment

conda run -n metadrive python -m method_chains.detour_fusion.experiment --reuse-bank

conda run -n metadrive python -m method_chains.detour_fusion.replay

conda run -n metadrive python -m method_chains.detour_fusion.passing_experiment
```

默认输出固定到 `results/method_chains/detour_fusion/`。
