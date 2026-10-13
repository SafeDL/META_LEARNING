# 有限预算高风险驾驶场景发现

当前主方法为 **CG-BRT：一致性门控的贝叶斯风险迁移**（Concordance-Gated Bayesian Risk Transfer）。

方法先参考六个历史系统的风险记录，再用新系统的测试反馈更新预测和来源权重。若反馈支持迁移后的排序，就利用它选场景；证据不足时使用参考 GP。选例同时考虑风险、不确定性和剩余预算。

实验使用 highway-env 的六个被测系统、两个场景池和五个随机种子。每池包含切入、前车急刹各 1024 个场景；每条轨迹预算 **200 次**，含前 10 次初始化。主方法只接收已测试场景的风险分数 `R`，以 `R > 0.5` 判定高风险；碰撞单独评价。

当前平均发现 141.25 个高风险场景，覆盖率 72.695%，发现曲线面积 46.170%。完整结果与各方法的反馈差异见[比较表](results/concordance_gated_bayesian_risk_transfer/comparison.md)。

| 入口 | 内容 |
|---|---|
| [设计文档](docs/Failure_Discovery_Design.md) | 方法与实验设计 |
| [方法目录](methods/README.md) | 主方法和基线实现 |
| [实验入口](experiments/README.md) | 比较、核验和评价 |
| [数据与结果](results/README.md) | 响应池、模型和测试记录 |
| [被测算法](sut_algorithms/README.md) | 六个目标系统 |
| [恢复归档](archives/README.md) | 历史分支恢复 |
| [代码规范](docs/style.md) | 命名与实现要求 |

在项目根目录运行：

```powershell
conda activate metadrive
python -B -m experiments.concordance_gated_bayesian_risk_transfer.compare
python -B -m experiments.concordance_gated_bayesian_risk_transfer.verify
python -B -m pytest -q -p no:cacheprovider
```

`compare` 重算已有结果，`verify` 核对记录和冻结源码；均不运行新的选择轨迹或物理仿真。

Git 默认允许跟踪源码、配置、结果、实测数据和小型检查点，仅忽略缓存、凭据与明确列出的超大归档，见 [.gitignore](.gitignore)。
