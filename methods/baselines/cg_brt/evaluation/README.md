# CG-BRT 实验

已完成六种驾驶控制器、两个场景测试集、五个随机种子的 60 条 CG-BRT 轨迹。每条轨迹共用两类场景的 200 次预算，包含 10 次初始化。

| 文件 | 用途 |
|---|---|
| [evaluate.py](evaluate.py) | 复用或补齐 CG-BRT 测试轨迹 |
| [compare.py](compare.py) | 汇总发现数量、覆盖率和代表性基线 |
| [verify.py](verify.py) | 核对风险反馈、来源更新、门控和源码快照 |
| [config.py](config.py) | 结果位置、种子与指标 |

在项目根目录、`metadrive` 环境中查看和核验已有结果：

```powershell
python -B -m methods.baselines.cg_brt.evaluation.compare
python -B -m methods.baselines.cg_brt.evaluation.verify
```

比较表包含五种子基线和种子 11 补充比较，共 1056 条唯一轨迹。结果见[结果目录](../../../../results/baselines/cg_brt/README.md)。按功能场景分别 200 次的比较另见 [LRT 按类实验](../../../../results/baselines/lrt/by_family/README.md)。
