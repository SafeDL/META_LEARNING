# 测试基线

本目录实现论文正式对比使用的自动驾驶失效发现与回归排序基线，并将其训练、搜索与排序流程接入公开有限候选库和统一受限付费查询接口（`QueryLedger`）。

## 一、跨 ADS 风险预测与预算发现基线（E01／E02）

| 方法 | 目录 | 核心机制 |
|---|---|---|
| `Pooled historical GP` & `Random` | [response_baselines/](response_baselines/README.md) | 在冻结共享表征上合并同场景全部历史与已付费目标碰撞–RSS 反馈拟合分组单联合 GP（六标量十步、`lr = 0.003`、CUDA float64）；固定种子不放回随机选例 |
| `ScenarioFuzz` | [scenario_fuzz/](scenario_fuzz/README.md) | 依据历史档案训练节点／边双 GAT 场景图风险网络，结合预测过滤、驾驶质量反馈与先全局后局部变异调度候选种子 |
| 配对系统主动评估 | [paired_system_active_evaluation/](paired_system_active_evaluation/README.md) | 以历史核平滑风险代理配合 MC Dropout 目标代理执行局部控制变量残差校正，结合支持互信息过滤与聚类按批（每批 5 例）选例 |

> 注：`Target-only GP`（仅使用当前已付费目标反馈的原始特征双输出 GP）与 `Static prior`（固定 $K=0$ 历史先验排序、全程不根据目标反馈更新）由 [testing_methods/history_transfer/](../history_transfer/README.md) 与 [experiments/baselines.py](../../experiments/baselines.py) 统一提供。

## 二、同谱系策略版本回归基线（E04 固定套件）

| 方法 | 目录 | 核心机制 |
|---|---|---|
| `DETOUR` | [detour/](detour/README.md) | 基于归一化场景物理参数的 Ward 欧氏分层聚类与已知碰撞邻近度，在共同初始化后对固定套件一次性静态排序 |
| `Birchler` 贪心排序 | [birchler_greedy/](birchler_greedy/README.md) | 按未测候选到已选用例集合的平均标准化欧氏距离最大化规则逐步贪心排序 |
| `SDC-Scissor` | [sdc_scissor/](sdc_scissor/README.md) | 合并准入同谱系历史与已付费当前版本碰撞标签，每次查询前执行少数类平衡过采样并重拟合 L2 逻辑回归，选择预测碰撞概率最高的未测点 |
| `SPECTRE` | [spectre/](spectre/README.md) | 从准入历史与共同初始化的 5 近邻距离倒数加权估计碰撞概率、连续 RSS 缺口及 `(0.3, 0.9]` 潜在碰撞指示，结合公开交通属性驾驶需求与子集多样性，通过五目标 IBEA 进化搜索分块排定固定套件 |

---

## 三、实验覆盖与结果链接

- **E01（12-ADS 少反馈预测，48 任务）**：对比 `Target-only GP`、`Pooled historical GP`、`Ours v1` 与 `Ours method`（含 E03 `Frozen weights`）。
- **E02（12-ADS 预算内碰撞发现，48 任务）**：对比全部 6 项外部基线（`Target-only GP`、`Pooled historical GP`、`ScenarioFuzz`、配对系统主动评估、`Static prior`、`Random`）与 4 个本文消融变体（`Ours v1`、`+LocalRel`、`+Risk-UCB`、`Ours method`）。
- **E04（3 策略固定套件回归，36 任务）**：在 `PPO`、`Continual DQN` 与 `Social Attention DDQN` 的 `R0–R2` 版本上全面对比 `DETOUR`、`Birchler`、`SDC-Scissor`、交通属性 `SPECTRE`、`Ours v1` 与 `Ours method`。

所有实验均以仿真引擎原生碰撞判定真实失效。[统一 12-ADS 结果](../../results/lifelong_transfer/README.md) · [原 8-ADS E01](../../results/k_shot_adaptation/README.md) · [原 8-ADS E02](../../results/budget_discovery/README.md) · [PPO 回归](../../results/policy_regression/fixed_suite/README.md) · [Continual DQN 回归](../../results/dqn_continual_learning/policy_regression/README.md) · [Social Attention DDQN 回归](../../results/public_ads/policy_regression/README.md) · [实验计划](../../docs/experiment_plan.md)
