# 历史响应迁移与联合反馈测试

本目录实现 **多源历史响应迁移与联合反馈适配框架（Historical Response Transfer with Joint Feedback）** 的离线表征学习、分来源联合后验构建、少样本在线适配、空间局部可信度选例及跨任务持续演化。

[总体方法流程图](../../figures/schematics/four_stage_method.png) 展示单一功能场景内的四个递进阶段（绘图入口：`python -B -m experiments.plotting.method_schematic`）。

---

## 一、模块与论文章节对应关系

| 文件 | 对应环节 | 核心实现职责 |
|---|---|---|
| [model.py](model.py)、[shared_private.py](shared_private.py) | §IV-A 历史风险先验构建 | 定义双输出联合核基类与共享–专属潜过程深度核 `SharedPrivateKernel`（含正定交叉协方差收缩因子与 RSS 观测噪声常量） |
| [inference.py](inference.py) | §IV-A / §IV-B 联合后验推断 | 实现“先 RSS 解析高斯条件化、后碰撞阻尼 Newton–Laplace 推断”两步联合后验 `JointPosterior`、来源摘要导出 `historical_prior` 与条件预测 `JointPrior` |
| [adaptation.py](adaptation.py) | §IV-B 在线目标残差适配 | 定义固定六标量十步配置 `FitSettings`／`OnlineSettings`，实现冻结深度表征下的目标核与原始特征核在线拟合 `adapted_online` |
| [training.py](training.py) | §IV-B 顺序元训练 | 拟合基础联合核（1,500 步）并在坐标严格互斥的逐源留出任务上重放前缀适配与动态调权，训练历史编码器（1,500 步） |
| [selection.py](selection.py) | §IV-B / §IV-C 基础序贯选例 | 实现全局查询前碰撞似然权重更新与贪心碰撞概率选例（`Ours v1`）、共同反馈预测评估及固定碰撞目标停止规则 |
| [lifelong_reliability.py](lifelong_reliability.py) | §IV-D 空间局部可信度与不确定性 | 实现带指数遗忘（`0.985`）、核平滑局部对数似然残差（收缩常数 `20`）与目标分支保底（`0.05`）的空间局部权重 `effective_weights`，以及 32 节点 Gauss–Hermite 概率方差矩 `logistic_moments` 与失配门控系数 `risk_beta` |
| [lifelong_selection.py](lifelong_selection.py) | §IV-D 增强主动发现选例 | 封装 `local_reliability`（`+LocalRel`）、`risk_ucb`（`+Risk-UCB`）与 `lifelong_full`（`Ours method`）三种配置的主动选例与逐次认知不确定性记录 |
| [lifelong_archive.py](lifelong_archive.py) | §IV-E 偏差感知归档 | 保留前 20 条无偏随机初始化前缀，按查询前碰撞概率对主动选例执行倾向去偏加权（下限 `0.15`）并锚定平滑基线碰撞率，构造加权联合后验 `WeightedJointPosterior` 与基于有效样本量（ESS）的来源快照 `source_snapshot` |
| [lifelong_training.py](lifelong_training.py) | §IV-E 锚定更新与 LOSO 门控 | 按 $\sqrt{\text{ESS}}$ 轮转抽样来源，在父模型参数上叠加 L2-SP 锚定正则（$\lambda_{\text{sp}}=0.01$、300 步、`lr = 3e-4`）微调，并通过零泄漏留一来源内部验证折（NLL 降幅 $\ge 0.002$）门控模型发布 `update_shared` |
| [reliability.py](reliability.py)、[lookahead.py](lookahead.py) | 辅助算子回归 | 保留基础核局部权重与联合反馈一步前瞻算子的独立数值回归实现 |

---

## 二、核心算法配置与在线推断约定

1. **分来源联合后验与顺序元训练**：
   - 每个历史来源独立保存候选库上的双输出后验均值 $(N, 2)$ 与候选内 $2 \times 2$ 协方差块 $(N, 2, 2)$。
   - 离线训练分两阶段（各 1,500 步、`lr = 0.001`、种子 `1301`，取末步最终模型）：第一阶段最大化多源联合边际似然训练 `target_kernel`；第二阶段轮流以单个历史 ADS 作为模拟目标，按物理坐标从其余来源中严格排除模拟目标整段序列的全部坐标，重放前缀适配与权重更新，以未见候选的碰撞交叉熵损失训练 `historical_encoder`。
2. **固定默认六标量十步在线适配**：
   - 由 `experiments.transfer_settings.native_settings()` 统一提供（float64 精度）。每次付费查询后冻结深度特征网络，从固定初始化用全部已付费碰撞与 RSS 联合反馈拟合 10 步。
   - 历史表征目标分支学习率 `0.003`；原始特征目标分支与 `Target-only GP` 共用学习率 `0.01`、初始碰撞潜均值 `-1.4`。
   - 初始分支先验固定为：历史来源组总权重 `0.675`（同权或按有效样本量分配）、历史表征目标分支 `0.225`、原始特征目标分支 `0.100`。`update_source_weights=False` 构成冻结权重对照（E03），两条目标 GP 分支仍继续执行十步适配。
3. **空间局部可信度与组件消融**：
   - `lifelong_selection.select` 支持 `configuration in ("local_reliability", "risk_ucb", "lifelong_full")`，在固定共同反馈顺序（E01/E03）下记录含局部可信度的留出集预测快照，在主动预算任务（E02/E04/E05）中驱动实际查询选择。
4. **同谱系版本回归与跨任务演化**：
   - 同谱系回归（E04）在四类场景内从两个真实前代的碰撞–RSS 响应训练 `SharedPrivateKernel`，各轮复测间将已付费前代参考记录追加为来源后验，并使用收敛模式隐式梯度（`implicit_evidence=True`）在线适配至评估端发出 $Q_{90}$ 停止信号。
   - 跨任务持续演化（E05）通过 `lifelong_archive.archive_weights`、`lifelong_archive.source_snapshot` 与 `lifelong_training.update_shared` 完成任务归档与门控共享表征更新。

---

## 三、单元测试与运行入口

运行本目录核心算法的单元测试（不触发原生闭环仿真）：

```powershell
python -B -m pytest tests/history_transfer -q
```

正式实验参数与验收判据见[统一实验计划](../../docs/experiment_plan.md)，完整运行与复算命令见 [experiments/README.md](../../experiments/README.md)，12-ADS 统一实证结果见 [results/lifelong_transfer/README.md](../../results/lifelong_transfer/README.md)。
