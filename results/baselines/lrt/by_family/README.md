# LRT 按功能场景实验

局部风险迁移（LRT）、CG-BRT、DART、HGT、GP-EI、RF-BO、均匀随机测试及 LRT 无动作价值学习消融，均按功能场景独立运行。

固定设置：切入、前车紧急制动各 1024 个候选；三个场景测试集、六种驾驶控制器、种子 11；每条轨迹 200 次测试，包含相同的 10 次初始化。已完成 **288 条轨迹、57600 次计费回放**，无新增物理仿真。

| 功能场景 | 场景测试集 | LRT H200 | CG-BRT H200 | LRT 覆盖率 | CG-BRT 覆盖率 |
|---|---|---:|---:|---:|---:|
| 车辆切入 | 前两个 | 121.33 | 125.08 | 88.70% | 89.53% |
| 车辆切入 | 第三个 | 117.83 | 123.50 | 87.12% | 88.88% |
| 前车紧急制动 | 前两个 | 49.00 | 48.83 | 87.45% | 87.08% |
| 前车紧急制动 | 第三个 | 48.67 | 48.83 | 86.48% | 86.85% |

表中按控制器等权平均。每类历史数据按 819 条训练、205 条验证划分；历史编码器每来源使用 128 条本类上下文。LRT 两类模型的完整构建耗时分别为 37.66、41.02 分钟，含分支标签、训练与验证。

## 查看与复现

在项目根目录、`metadrive` 环境中运行：

```powershell
python -B -m methods.baselines.lrt.evaluation.family_report
python -B -m methods.baselines.lrt.evaluation.family_verify
```

完整构建和评价入口为 `python -B -m methods.baselines.lrt.evaluation.by_family`，已有阶段直接复用。

## 结果文件

| 位置 | 内容 |
|---|---|
| [comparison.md](comparison.md)、`comparison.json` | 八种方法逐控制器比较、发现目标与复用成本 |
| `cutin/`、`brake/` | 独立模型、历史标签、候选缓存和正式轨迹 |
| 各类 `build_cost.json`、`historical_queries.json` | 完整构建耗时与历史回放次数 |
| 各类 `summary.json`、`discovery.png` | 按类效果、成本与发现曲线 |
| `full_scan_cost.json` | 相对全量执行 1024 个场景的发现能力与成本 |
| `protocol.json`、`frozen_inputs.zip` | 实验设置、源码和权重快照 |
| `verification.json`、`timing_review.json` | 查询、反馈、初始化和计时核验 |

成本计入模型构建、候选准备、在线计算和查询场景的档案执行耗时，报告复用 1、10、50、100、200 次。达到共同发现目标的配对成本见完整报告。

本实验两类各 200 次，合计 400 次；两类共同使用 200 次的结果见[上级目录](../README.md)。本文最新方法见 [GCRT 结果](../../../gcrt/README.md)。
