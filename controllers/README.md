# 被测驾驶控制器

本目录实现被测自动驾驶系统（ADS）控制器。控制器仅接收当前可见交通状态（隐藏未来事件时间和被遮挡参与者），经统一运动学限幅接口输出纵向加速度与前轮转向控制；场景采样与选例由 [testing_methods/](../testing_methods/README.md) 负责。

## 目录与文件职责

| 位置 | 职责 |
|---|---|
| [highway/catalog.py](highway/catalog.py) | 登记 8 个历史来源与 8 个同范式目标 ADS 的系统身份、控制范式、超参数与训练种子 |
| [highway/factory.py](highway/factory.py)、[highway/contracts.py](highway/contracts.py) | 构造已登记控制器实例，生成绑定源码、权重与时钟的运行时身份契约 |
| [highway/control.py](highway/control.py)、[highway/decision_interface.py](highway/decision_interface.py) | 定义纵向加速度与前轮转向限幅指令 `Control`，封装匿名可见世界接口 `DecisionInterface` |
| [highway/rule.py](highway/rule.py)、[highway/optimization.py](highway/optimization.py)、[highway/planning.py](highway/planning.py)、[highway/search.py](highway/search.py) | 实现规则控制（`IDM/MOBIL`、`FVDM`）、优化规划（`Linear MPC`、`Nonlinear trajectory`）、离散搜索（`UCT`、`Beam`）及 20 Hz 匀速外推轨迹评估 |
| [highway/learning.py](highway/learning.py)、[highway/train.py](highway/train.py) | 离散元动作映射（`DiscreteDriver`）及 `PPO`、`DQN` 策略加载与本地训练 |
| [highway/public_agents.py](highway/public_agents.py)、[highway/adapted_agents.py](highway/adapted_agents.py)、[highway/public_shield.py](highway/public_shield.py) | 加载外部公开跨架构策略 `OnlineMC PPO`（含 Maude 形式化安全屏蔽）、`Social Attention DDQN` 与 `Highway Decision Transformer` |
| [highway/continuous_sac.py](highway/continuous_sac.py)、[highway/interval_robust_planner.py](highway/interval_robust_planner.py) | 连续动作空间（纵向加速度 + 前轮转角）`Continuous SAC` 策略封装及区间鲁棒规划器接口 |
| [highway/checkpoints/](highway/checkpoints/) | 保存同范式学习型策略权重（[checkpoints/collision/](highway/checkpoints/collision/)）与跨架构策略冻结权重（[checkpoints/public/](highway/checkpoints/public/)） |
| [metadrive/](metadrive/) | MetaDrive 平台的控制器基类、IDM 控制器与注册表（预留用于后续跨平台扩展） |

---

## 一、同范式历史与目标控制器（8 历史 + 8 同范式目标）

`catalog.FORMAL_SUTS` 登记 8 个固定历史 ADS（`FORMAL_HISTORY_SUTS`，历史 DQN 使用 `history_dqn_native`）与 8 个同范式偏移目标 ADS（`FORMAL_TARGET_SUTS`）。仿真执行、遮挡裁剪和轨迹记录见 [simulator/](../simulator/README.md)。

- **规则、优化与搜索控制器**：规则控制器在首次换道和车身跨线期间同时考虑被占车道；FVDM 采用 `a = 0.6 * (V(s) - v) + velocity_gain * (v_lead - v)` 跟驰与独立换道规则组合。优化规划器对可见交通作 20 Hz 匀速外推并进行非线性车身与道路边界复核；UCT 和束搜索同样按 20 Hz 物理步评估候选动作序列。
- **离散强化学习控制器**：复现论文主队列使用 `experiments.paper_results.create_controller(sut_id)`。修订目标 DQN（`Continual DQN`）权重保存在 `checkpoints/collision/target_dqn_continual/`，基础与场景内持续学习协议见[修订 DQN 训练与评测记录](../results/dqn_continual_learning/README.md)。
- **能力核验与训练入口**：当前执行实现标识为 `ads-native-highway-v2`，绑定 20 Hz 物理／5 Hz 决策的源码、参数与权重（[原登记 ADS 能力轨迹](../results/k_shot_adaptation/qualification/) · [修订 DQN 能力轨迹](../results/dqn_continual_learning/base/qualification/)）。
  - 正式队列能力核验：`python -B -m experiments.history_transfer_experiment qualify`
  - 原生学习型控制器训练：`python -B -m controllers.highway.train native`（或指定 `<ADS_ID>`）

---

## 二、跨架构异质控制器（4 跨范式目标）

统一 12-ADS 基准包含 4 个跨范式目标：`OnlineMC PPO`、`Social Attention DDQN`、`Highway Decision Transformer` 与前瞻盲测准入的 `Continuous SAC`。[实验计划](../docs/experiment_plan.md) · [三项离散公开策略验收](../results/public_ads/qualification/README.md) · [Continuous SAC 验收](../results/lifelong_transfer/qualification/continuous_sac/acceptance.json)

### 1. 公开来源离散动作策略（`OnlineMC PPO`、`Social Attention DDQN`、`Decision Transformer`）

由 `controllers.highway.public_agents.create(sut_id)` 每幕独立实例化，采集前通过 `experiments.public_ads.qualify.verify_admitted(sut_id)` 核验。依赖与固定上游源码安装于 `cache/highway_agents/`：

```powershell
$adsPython = 'F:/anaconda3/envs/metadrive/python.exe'
$adsCache = Join-Path (Get-Location) 'cache/highway_agents'
$adsDeps = Join-Path $adsCache 'dependencies'
& $adsPython -B -m experiments.public_ads.prepare
& $adsPython -m pip install --target $adsDeps --no-deps maude==1.5.0 seaborn==0.13.2 numba==0.61.2 llvmlite==0.44.0
& $adsPython -m pip install --target $adsDeps transformers==4.26.1 numpy==2.2.6
$adsSource = (Get-ChildItem (Join-Path $adsCache 'upstream') -Directory | ForEach-Object { $_.FullName }) -join ';'
$env:PYTHONPATH = "$adsDeps;$adsSource;$adsCache/highway_dt"
& $adsPython -B -m experiments.public_ads.qualify --verify online_mc_ppo social_attention_ddqn decision_transformer
```

`experiments.public_ads.prepare` 恢复并校验固定上游版本；`experiments.public_ads.train <ADS_ID>` 以种子 `1301` 复现本地训练（拒绝覆盖 [checkpoints/public/](highway/checkpoints/public/) 中已登记权重）；`experiments.public_ads.qualify <ADS_ID>` 核验五项正式驾驶课程并导出四个开发点回放。

### 2. 连续动作策略（`Continuous SAC`）

由 `controllers.highway.continuous_sac.create()` 加载冻结权重 [checkpoints/public/continuous_sac/policy.pt](highway/checkpoints/public/continuous_sac/policy.pt)。策略在 `Box(-1, 1, shape=(2,))` 连续动作空间输出确定性均值动作，经仿射映射转化为纵向加速度 $a \in [-8.0, +3.0]\text{ m/s}^2$ 与速度约束前轮转角，并经 `DecisionInterface` 限幅。训练与准入核验入口为：

```powershell
python -B -m experiments.public_ads.continuous_sac verify
```

完整训练、准入与 12-ADS 持续迁移流程见[实验运行说明](../experiments/README.md#持续风险迁移)。
