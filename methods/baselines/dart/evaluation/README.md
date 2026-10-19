# DART 实验

已完成两组实验，随机种子均为 11、23、37、53、71：

| 范围 | 比较内容 | 轨迹数 |
|---|---|---:|
| 前两个场景测试集 | CG-BRT、DART 与五项消融，六种驾驶控制器 | 420 |
| 第三个场景测试集 | CG-BRT、DART、常规训练、无前瞻，六种驾驶控制器 | 120 |

第三个测试集的 12288 次物理测量已完成，数据位于 `data/scenarios/pool_2/`，当前复现直接使用这些记录。每条轨迹的 200 次预算由两类场景共同使用。

在项目根目录、`metadrive` 环境中汇总已有结果：

```powershell
python -B -m methods.baselines.dart.evaluation.report
python -B -c "from methods.baselines.dart.evaluation.report import main; main('confirmation')"
```

完整流程入口为 `python -B -m methods.baselines.dart.evaluation.pipeline`，依次处理训练、前两个测试集比较、快照、第三个测试集采集、比较与核验，已有产物直接复用。各阶段实现为 `evaluate.py`、`freeze.py`、`prepare.py`、`report.py`、`verify.py`。

两组实验的 H200 和覆盖率均达到相对 CG-BRT 保留至少 90% 的预设门槛。逐控制器结果、消融与查询记录见[结果目录](../../../../results/baselines/dart/README.md)。
