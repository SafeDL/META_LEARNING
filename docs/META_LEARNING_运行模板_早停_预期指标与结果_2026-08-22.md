# META_LEARNING 最新运行模板：早停、预期指标与结果

**GPU 环境**：

```powershell
conda activate metadrive
```

**目标环境**：MetaDrive + PyTorch + 单张 RTX 4090  
**代码审查基线**：`main @ c467ccfcac9e29a9a3da0510cbd98a6ed18dcbb0`  
**原则**：核心算法冻结为 `HPTR + episode-level q(z|C) + Outer Hybrid PPO + Inner SAC`，后续重点是可靠训练和实验，不再频繁改框架。

**当前阶段状态（2026-08-22）**：

- Gate A 已按修正后的 aligned failure-disagreement 指标重新运行；
- 物理 episode：192；
- `mean_failure_disagreement = 0.0520833`；
- `mean_severity_distance = 0.100455`；
- `valid_rate = 1.0`；
- `failure_rate = 0.192708`；
- Gate A：**FAIL**；
- 因此当前仍应停在 Gate A，不进入 Inner / Posterior / Calibration / Outer；
- 由于 `valid_rate` 和 `failure_rate` 均处于健康区间，当前不应优先修改 scene parameter ranges；问题集中在 meta-train IDM profiles 的 failure boundary 仍不够分离。

---

## 1. 最新代码对应的正式流程

最新代码已经加入 10-D `PhysicalStateExtractor`、`InnerRiskReward`、统一 `OnlineMetaTest`、纯 IDM profile split 和新的 `INNER_CALIBRATION` 阶段。

因此建议实际运行顺序固定为：

```text
Preflight
  ↓
Gate A：Failure-Landscape Heterogeneity
  ↓
Inner SAC Pretraining
  ↓
Posterior Training
  ↓
Inner z-Conditioned Calibration
  ↓
Outer Hybrid PPO
  ↓
Validation Early Stopping
  ↓
Final Meta-Test
```

`light_joint` 仍默认关闭。

### 为什么现在需要 Inner Calibration？

Inner pretraining 时 posterior 基本处于 prior：

\[
z_0=0
\]

Posterior 学成以后，真实在线过程会出现：

\[
z_1,z_2,\ldots 
eq 0
\]

如果不 calibration，Inner 虽然结构上接收 \(z\)，但并没有真正学会利用 posterior-derived non-zero latent。因此最新代码新增 calibration 是合理的。

---

# 2. 运行目录

PowerShell：

```powershell
$CFG = "meta_testing/configs/mvr.yaml"
$TASKBOOK = "meta_testing/configs/idm_taskbook.json"
$RUN = "runs/mvr_c467ccfc_seed11"

New-Item -ItemType Directory -Force `
  "$RUN/00_preflight", `
  "$RUN/01_gateA", `
  "$RUN/02_inner", `
  "$RUN/03_posterior", `
  "$RUN/04_inner_calibration", `
  "$RUN/05_outer", `
  "$RUN/06_validation", `
  "$RUN/07_meta_test", `
  "$RUN/08_ablation"
```

推荐 checkpoint 命名：

```text
inner_block01.pt
inner_block02.pt
posterior_block01.pt
posterior_block02.pt
calibration_block01.pt
outer_block01.pt
outer_block02.pt
outer_final.pt
```

每个 `.pt` 都会有同名 `.json` sidecar，用于查看 stage、metrics 和 progress。

---

# 3. 当前默认配置的真实 simulator 预算

当前 `mvr.yaml` 主要预算：

```yaml
training:
  step_budget: 240

inner:
  episode_budget: 20

inner_calibration:
  episode_budget: 8
  support_episodes: 1

posterior:
  episode_budget: 20

outer:
  episode_budget: 20

evaluation:
  total_episode_budget: 20
  support_shots: [0, 1, 2, 4]
  seeds: [11, 22, 33]
```

注意不同阶段的 `episode_budget` 含义不同。

| 阶段 | 一次调用真实 physical episodes |
|---|---:|
| Inner | 20 |
| Posterior | 20 |
| Inner Calibration | 8 × (1 support + 1 query) = 16 |
| Outer | 12 train tasks × 20 = **240** |
| Final meta-test | 3 families × 3 seeds × 4 K × 20 = **720** |

meta-train 当前有：

```text
4 IDM profiles × 3 scenario families = 12 tasks
```

因此最容易低估的是 Outer：`outer.episode_budget=20` 是**每个 task 20 次**。

Gate A 默认还会运行：

```text
4 meta-train IDM profiles × 3 families × 16 configurations = 192 episodes
```

若只跑一个训练 block，完整流程约：

```text
Gate A             192
Inner               20
Posterior            20
Calibration          16
Outer               240
Final evaluation    720
-----------------------
Total              1208 physical episodes
```

因此单张 4090 下主要成本来自 simulator sampling，而不是网络规模。

---

# 4. Stage 0：Preflight

## 4.1 版本

```powershell
git rev-parse HEAD
```

当前模板对应：

```text
c467ccfcac9e29a9a3da0510cbd98a6ed18dcbb0
```

## 4.2 CUDA

```powershell
conda run -n metadrive python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

预期：

```text
True
NVIDIA GeForce RTX 4090
```

CUDA 不可用：**硬早停**。

## 4.3 单元测试

```powershell
conda run -n metadrive python -m pytest meta_testing/tests -q
```

预期：全部 PASS。

任何核心 test 失败：**硬早停，不开始训练**。

---

# 5. Stage 1：Gate A — IDM Failure-Landscape Heterogeneity

这是最重要的前置 Gate。

### 当前 Gate A 使用的 4 个 meta-train IDM profiles

| Profile | target speed (m/s) | lane change | distance wanted (m) | time headway (s) | ACC factor | DEACC factor |
|---|---:|---|---:|---:|---:|---:|
| `idm_cautious` | 7 | False | 24 | 3.0 | 1.20 | -2.0 |
| `idm_defensive` | 10 | False | 16 | 2.2 | 1.10 | -3.0 |
| `idm_normal` | 13 | True | 9 | 1.2 | 1.20 | -4.5 |
| `idm_assertive` | 18 | True | 3 | 0.45 | 1.80 | -8.0 |

Gate A 只对这 4 个 `meta_train` profiles 做硬判定；validation 的 `idm_fast_small_gap` 与 meta-test 的 `idm_late_response` 不参与 Gate A。

运行：

```powershell
conda run -n metadrive python -m meta_testing.scripts.audit_failure_landscape_heterogeneity `
  --taskbook $TASKBOOK `
  --output "$RUN/01_gateA/failure_landscape.json"
```

重点输出：

```json
{
  "pairwise_failure_disagreement": {},
  "pairwise_severity_distance": {},
  "dangerous_region_overlap": {},
  "mean_failure_disagreement": 0.0,
  "mean_severity_distance": 0.0,
  "gate": {
    "mean_failure_disagreement": 0.0,
    "mean_severity_distance": 0.0,
    "valid_rate": 0.0,
    "failure_rate": 0.0,
    "pass": true
  }
}
```

### 当前代码硬通过条件

当前实现的 Gate A 为：

\[
\operatorname{mean}_{i<j}\frac{1}{N}
\sum_{n=1}^{N}\mathbf{1}[F_i(x_n)\ne F_j(x_n)]\ge0.10
\]

并且：

\[
valid\_rate\ge0.80
\]

\[
0.02\le failure\_rate\le0.80
\]

`mean_severity_distance` 和 `dangerous_region_overlap` 当前作为诊断量，不参与代码硬 Gate。

### 建议研究级目标

| 指标 | 最低继续 | 理想 |
|---|---:|---:|
| valid_rate | ≥ 0.80 | ≥ 0.90 |
| failure_rate | 0.02–0.80 | 0.05–0.60 |
| failure disagreement | ≥ 0.10 | ≥ 0.15 |
| dangerous overlap | 不应全部≈1 | 至少若干 pair < 0.7 |

### 最新 Gate A 实际结果

当前最新结果：

```text
episodes                    = 192
mean_failure_disagreement   = 0.0520833
mean_severity_distance      = 0.100455
valid_rate                  = 1.000000
failure_rate                = 0.192708
Gate A                      = FAIL
```

pairwise failure disagreement：

| Pair | disagreement |
|---|---:|
| assertive ↔ cautious | 0.000000 |
| assertive ↔ defensive | 0.104167 |
| assertive ↔ normal | 0.000000 |
| cautious ↔ defensive | 0.104167 |
| cautious ↔ normal | 0.000000 |
| defensive ↔ normal | 0.104167 |

该结果的含义不是“测试空间不合适”。相反：

- `valid_rate = 1.0`：场景物理执行稳定；
- `failure_rate ≈ 0.193`：既不是全安全，也不是全失败；
- 真正的问题是 `assertive / cautious / normal` 在当前 48 个 aligned configurations 上得到完全相同的 failure pattern；
- 只有 `defensive` 与另外三个 profiles 表现出约 10.4% 的 failure disagreement。

因此当前阶段的修正方向应是**继续拉开 meta-train IDM profiles 的行为/失效边界**，而不是优先扩大或收缩 scene parameter ranges。

### 研究级接受建议

代码硬 Gate 只要求平均 disagreement ≥ 0.10，但论文级 task heterogeneity 不应只依赖平均值。建议同时检查：

1. `mean_failure_disagreement >= 0.10`；
2. 不应存在 3 个 meta-train profiles 在全部 aligned configurations 上 failure pattern 完全一致；
3. `valid_rate >= 0.80`；
4. `failure_rate` 保持在非退化区间；
5. severity / dangerous-region overlap 至少能显示多个 profile pair 的差异。

这些检查全部来自同一次 Gate A 输出，不额外增加实验。

### 早停

出现任一情况，不进入 Meta-RL 训练：

```text
failure_rate ≈ 0       → 测试空间太安全
failure_rate ≈ 1       → 所有 profile 都失败，无法区分
failure disagreement < 0.10 → profile failure landscape 不够异质
valid_rate < 0.80      → 场景执行本身不稳定
```

处理原则：

```text
如果 valid_rate / failure_rate 退化
→ 再考虑 scene parameter ranges

如果 valid_rate、failure_rate 健康，但 disagreement < 0.10
→ 只优先修改 IDM profiles，不动场景范围
```

当前最新结果属于第二种情况：**优先继续修改 IDM profiles；不要进入 Inner，也不要先改 scene parameter ranges。**

---

# 6. Stage 2：Inner SAC Pretraining

当前训练：

```text
map_encoder
shared_feature_encoder
option_embedding
inner_sac
```

Inner 已使用固定 10-D physical state 和 `InnerRiskReward`。

## Block 01

```powershell
conda run -n metadrive python -m meta_testing.scripts.train_inner `
  --config $CFG `
  --output "$RUN/02_inner/inner_block01.pt"
```

## Block 02

```powershell
conda run -n metadrive python -m meta_testing.scripts.train_inner `
  --config $CFG `
  --resume "$RUN/02_inner/inner_block01.pt" `
  --output "$RUN/02_inner/inner_block02.pt"
```

每个 block = 20 episodes。

预期 sidecar：

```json
{
  "stage": "inner_pretrain",
  "metrics": {
    "episodes": 20,
    "transitions": 0,
    "updates": 0,
    "last_loss": {
      "inner_actor_loss": 0.0,
      "inner_critic_loss": 0.0,
      "inner_alpha_loss": 0.0
    }
  },
  "progress": {"episodes": 20}
}
```

数值只是格式示例，不是预期绝对值。

## 不要用 SAC loss 单独早停

SAC actor/critic/alpha loss 不要求单调趋近 0。它们主要判断：

```text
finite
无 NaN / Inf
无持续爆炸
```

真正看物理指标：

```text
valid_episode_rate
invalid_rate
mean / median min_TTC
near_miss_rate
target_collision_rate
mean InnerRiskReward
option_effect_size
repeated_success_rate
```

当前 G4 最低：

```text
option_effect_size > 0
repeated_success_rate >= 0.80
```

建议研究目标：

```text
repeated_success_rate >= 0.80
option_effect_size >= 0.20
invalid_rate <= 0.15
```

### Inner 正常早停

满足：

1. G4 连续两个 block 通过；
2. validation risk/near-miss 指标连续两个 block改善 < 2%；
3. invalid_rate ≤ 0.15；
4. loss 全 finite。

建议最大：

```text
5 blocks = 100 episodes
```

100 episodes 后 G4 仍失败，不要继续烧 episode，检查 reward、state、action mapping、option semantics。

### 硬早停

```text
NaN / Inf
invalid_rate > 0.30
绝大多数 episode 立即 invalid
action 长时间饱和 -1/+1
风险指标完全不随训练变化
```

---

# 7. Stage 3：Posterior Training

从通过 Inner Gate 的 checkpoint 开始。

例如：

```powershell
$INNER = "$RUN/02_inner/inner_block03.pt"
```

## Block 01

```powershell
conda run -n metadrive python -m meta_testing.scripts.train_posterior `
  --config $CFG `
  --resume $INNER `
  --output "$RUN/03_posterior/posterior_block01.pt"
```

## Block 02

```powershell
conda run -n metadrive python -m meta_testing.scripts.train_posterior `
  --config $CFG `
  --resume "$RUN/03_posterior/posterior_block01.pt" `
  --output "$RUN/03_posterior/posterior_block02.pt"
```

当前输出：

```json
{
  "stage": "posterior",
  "metrics": {
    "episodes": 20,
    "updates": 0,
    "support_counts": [1, 2, 4],
    "posterior_loss": 0.0
  }
}
```

绝对 loss 不设硬阈值，必须结合 validation。

## Posterior early stopping

### 1. Validation held-out outcome loss

连续两个 block：

\[
|L_t-L_{t-1}|/|L_{t-1}|<1\%
\]

可视为 plateau。

### 2. G5 Few-shot Identifiability

最低：

\[
d_{between}>d_{within}
\]

推荐：

\[
d_{between}/d_{within}\ge1.2
\]

理想 ≥1.5。

### 3. K-shot 趋势

预期：

```text
K=0：prior
K=1：开始可分
K=2：明显增强
K=4：继续增强或趋于饱和
```

如果：

```text
K=0 ≈ K=4
```

Meta adaptation 没有成立。

### 4. G6 Posterior Causal Utility

最低：

\[
J(z_{correct})>\max(J(z_{swapped}),J(z_{zero}))
\]

推荐至少有 5% margin。

### Posterior 成功早停

```text
val loss plateau
AND G5 pass
AND G6 pass
```

建议最大：

```text
5 blocks = 100 physical episodes
```

100 episodes 后 G5/G6 仍失败：**不进入 Outer**。

---

# 8. Stage 4：Inner z-Conditioned Calibration

从最终 Posterior checkpoint：

```powershell
$POST = "$RUN/03_posterior/posterior_block03.pt"
```

运行：

```powershell
conda run -n metadrive python -m meta_testing.scripts.train_inner_calibration `
  --config $CFG `
  --resume $POST `
  --output "$RUN/04_inner_calibration/calibration_block01.pt"
```

当前配置：

```text
8 calibration iterations
× (1 support + 1 query)
= 16 physical episodes
```

预期格式：

```json
{
  "metrics": {
    "episodes": 16,
    "calibration_episodes": 8,
    "transitions": 0,
    "updates": 0,
    "last_loss": {}
  }
}
```

## Calibration early stop

比较同一 validation scene：

```text
Inner(correct z)
Inner(zero z)
Inner(swapped z)
```

成功条件：

\[
J_{inner}(correct)>\max(J_{inner}(zero),J_{inner}(swapped))
\]

研究目标：

```text
correct-z utility ≥ 5% improvement
invalid_rate 相比 calibration 前恶化不超过 0.05
```

推荐：

```text
先跑 1 block
最多 2 blocks
```

第二 block 仍无 z sensitivity，不继续扩大 calibration。

---

# 9. Stage 5：Outer Hybrid PPO

从 calibration checkpoint：

```powershell
$CAL = "$RUN/04_inner_calibration/calibration_block01.pt"
```

Block 01：

```powershell
conda run -n metadrive python -m meta_testing.scripts.train_outer `
  --config $CFG `
  --resume $CAL `
  --output "$RUN/05_outer/outer_block01.pt"
```

当前默认一次调用：

```text
12 tasks × 20 episodes = 240 physical episodes
```

预期 JSON：

```json
{
  "stage": "outer",
  "metrics": {
    "episodes": 240,
    "updates": 12,
    "outer_ppo_loss": 0.0
  }
}
```

## Outer 不以 PPO loss 作为 early stop

真正看 validation：

```text
cumulative_unique_failures
failure_discovery_auc
tests_to_first_valid_failure
tests_to_5_unique_failures
invalid_rate
unique_failures_per_episode
```

### 推荐 early stop

设 validation AUC 为 \(A_t\)。

若连续两个 Outer blocks：

\[
(A_t-A_{t-1})/\max(A_{t-1},\epsilon)<2\%
\]

并且 unique failures 无新增，则停止。

建议当前 POC：

```text
最多 3 Outer blocks
```

也就是最多约：

```text
720 Outer training episodes
```

3 blocks 后仍不优于 no-z / Sobol，不要继续单纯增加 PPO budget。

---

# 10. 当前早停所需的一个 P0 小改动

**当前 `evaluate_meta_test.py` 仍硬编码 meta-test split。**

正式运行前建议让 evaluate 支持：

```text
--split meta_validation
--profiles-key validation
```

期望命令：

```powershell
conda run -n metadrive python -m meta_testing.scripts.evaluate_meta_test `
  --config $CFG `
  --checkpoint "$RUN/05_outer/outer_block01.pt" `
  --split meta_validation `
  --profiles-key validation `
  --output "$RUN/06_validation/outer_block01.json"
```

最终：

```powershell
conda run -n metadrive python -m meta_testing.scripts.evaluate_meta_test `
  --config $CFG `
  --checkpoint "$RUN/05_outer/outer_final.pt" `
  --split meta_test `
  --profiles-key meta_test `
  --output "$RUN/07_meta_test/meta_test.json"
```

截至当前 commit，上述 `--split / --profiles-key` 还不是现成功能。

**禁止通过反复查看 `idm_late_response` 来选 checkpoint。**

---

# 11. 当前代码可直接执行的 Final Meta-Test

模型和所有超参数冻结后：

```powershell
conda run -n metadrive python -m meta_testing.scripts.evaluate_meta_test `
  --config $CFG `
  --checkpoint "$RUN/05_outer/outer_final.pt" `
  --output "$RUN/07_meta_test/meta_test.json"
```

该命令当前固定读取：

```text
meta_test = idm_late_response
```

总物理预算：

```text
3 families × 3 seeds × 4 K × 20 = 720 episodes
```

不要在训练过程中重复运行。

---

# 12. Final result JSON 重点

每个 meta-test task 都会有：

```text
K = 0 / 1 / 2 / 4
```

每个 K 下 3 个 seed，并报告：

```text
cumulative_unique_failures
failure_discovery_auc
tests_to_first_valid_failure
tests_to_5_unique_failures
tests_to_10_unique_failures
invalid_rate
unique_failures_per_episode
mean_failure_discovery_auc
mean_unique_failures
```

---

# 13. 最终应出现的结果趋势

## 13.1 Few-shot adaptation

核心趋势：

\[
K=0 < K=1 \lesssim K=2 \lesssim K=4
\]

不要求严格单调，但至少应有：

```text
K=2 或 K=4 明显优于 K=0
```

推荐论文级接受线：

```text
K=2/4 mean failure-discovery AUC
相对 K=0 提高 ≥ 10%
```

若：

```text
K=0 ≈ K=1 ≈ K=2 ≈ K=4
```

则 few-shot Meta-RL claim 不成立。

---

# 14. Failure discovery 推荐预期

固定 20 episode budget：

```text
invalid_rate <= 0.15
tests_to_first_valid_failure <= 5
```

`tests_to_5_unique_failures`：

```text
理想 <= 15
```

`tests_to_10_unique_failures`：

> 当前 POC 允许为 null，不作为硬门槛。

绝对 unique failure 数与当前 failure-signature 粒度高度相关，因此主结论应优先看相对 baseline 的提升。

---

# 15. 论文主体的 baseline 和成功标准

建议最终至少比较：

```text
Sobol / Random
Hierarchical RL without z
z → Outer only
z → Inner only
Full method
```

有余力再加入：

```text
CEM 或 BO
```

第一主指标建议：

\[
\boxed{\text{Failure Discovery AUC}}
\]

同时报告：

```text
Unique Failures @20
Tests-to-First
Tests-to-5
Invalid Rate
```

相对最强 non-meta baseline：

```text
AUC +10%：可以接受
AUC +15%：较有说服力
AUC +20%：强结果
```

同时应看到：

```text
unique failures ↑
tests-to-first ↓
tests-to-5 ↓
invalid_rate 不显著恶化
```

---

# 16. 四个核心 Scientific Gates

| Gate | 核心判断 | 推荐条件 |
|---|---|---|
| A Task Heterogeneity | IDM failure landscape 是否不同 | mean disagreement ≥0.10；valid≥0.80；failure rate∈[0.02,0.80] |
| B Inner Controllability | Inner 是否会制造有效风险 | repeated success≥0.8；invalid≤0.15 |
| C Few-shot Utility | z 是否可识别且有因果作用 | between>within；correct-z > zero/swapped |
| D Equal-budget Profitability | Full 是否真正省测试预算 | AUC > best baseline，目标 +10–15% |

**当前状态**：Gate A 仍为 FAIL，因此本轮正确动作仍是停止在 Stage 1；Gate B/C/D 尚未开始，不应提前解读。

Gate 失败时不继续后续昂贵阶段。

---

# 17. 总体 early-stop 表

| 阶段 | 成功早停 | 失败停止 | 推荐上限 |
|---|---|---|---:|
| Preflight | tests 全通过 | 任意核心 test fail | 1 次 |
| Gate A | landscape 异质且非退化 | disagreement<0.10 / 全安全 / 全失败 | 16 configs/family |
| Inner | G4 连续通过 + plateau | invalid>0.30 / 无 controllability | 100 ep |
| Posterior | val plateau + G5 + G6 | 100 ep 后仍 G5/G6 fail | 100 ep |
| Calibration | correct-z 有增益 | 2 blocks 仍无 z sensitivity | 2 blocks |
| Outer | val AUC 连续 2 blocks <2% 改善 | 3 blocks 仍不优于 no-z | 3 blocks |
| Meta-Test | 一次完成 | 禁止用于调参 | 720 ep |

---

# 18. 推荐完整命令序列

```powershell
# 0. Tests
conda run -n metadrive python -m pytest meta_testing/tests -q

# 1. Gate A
conda run -n metadrive python -m meta_testing.scripts.audit_failure_landscape_heterogeneity `
  --taskbook $TASKBOOK `
  --output "$RUN/01_gateA/failure_landscape.json"

# 2. Inner
conda run -n metadrive python -m meta_testing.scripts.train_inner `
  --config $CFG `
  --output "$RUN/02_inner/inner_block01.pt"

# 3. Posterior（将 inner_final 替换成实际通过 Gate 的 checkpoint）
conda run -n metadrive python -m meta_testing.scripts.train_posterior `
  --config $CFG `
  --resume "$RUN/02_inner/inner_final.pt" `
  --output "$RUN/03_posterior/posterior_block01.pt"

# 4. Inner calibration
conda run -n metadrive python -m meta_testing.scripts.train_inner_calibration `
  --config $CFG `
  --resume "$RUN/03_posterior/posterior_final.pt" `
  --output "$RUN/04_inner_calibration/calibration_block01.pt"

# 5. Outer
conda run -n metadrive python -m meta_testing.scripts.train_outer `
  --config $CFG `
  --resume "$RUN/04_inner_calibration/calibration_final.pt" `
  --output "$RUN/05_outer/outer_block01.pt"

# 6. Final Meta-Test：所有参数冻结后才运行
conda run -n metadrive python -m meta_testing.scripts.evaluate_meta_test `
  --config $CFG `
  --checkpoint "$RUN/05_outer/outer_final.pt" `
  --output "$RUN/07_meta_test/meta_test.json"
```

---

# 19. 每次运行后的记录模板

```text
Git SHA:
Config hash:
Date:
GPU:
Seed:

Stage:
Input checkpoint:
Output checkpoint:
Physical episodes:

actor loss:
critic loss:
alpha loss:
posterior loss:
PPO loss:
NaN/Inf: YES / NO

valid rate:
invalid rate:
failure rate:
near-miss rate:
collision rate:
mean min TTC:

K=0 AUC:
K=1 AUC:
K=2 AUC:
K=4 AUC:

K=0 unique failures:
K=1 unique failures:
K=2 unique failures:
K=4 unique failures:

within-z distance:
between-z distance:
correct-z utility:
zero-z utility:
swapped-z utility:

Decision:
[ ] PASS → next stage
[ ] EARLY STOP → accept checkpoint
[ ] FAIL → diagnose first

Reason:
```

---

# 20. 论文结果表模板

| Method | K | Unique Failures @20 ↑ | Discovery AUC ↑ | Tests-to-First ↓ | Tests-to-5 ↓ | Invalid Rate ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Sobol | 0 |  |  |  |  |  |
| Hierarchical no-z | 0 |  |  |  |  |  |
| z → Outer only | 1/2/4 |  |  |  |  |  |
| z → Inner only | 1/2/4 |  |  |  |  |  |
| **Full** | 1 |  |  |  |  |  |
| **Full** | 2 |  |  |  |  |  |
| **Full** | 4 |  |  |  |  |  |

期望主趋势：

```text
Full K=2/4
   >
Full K=0
```

并且 Full 应优于最强 non-meta baseline。

---

# 21. 当前代码的两个运行注意事项

## 21.1 README 已落后于最新 calibration commit

README 仍写：

```text
inner_pretrain → posterior → outer
```

但最新代码已经明确加入：

```text
INNER_CALIBRATION
```

对于 Full method，建议实际采用：

```text
inner_pretrain
→ posterior
→ inner_calibration
→ outer
```

## 21.2 Validation evaluator 是正式早停前的唯一 P0 小缺口

当前训练闭环基本已经可运行，但正式实验不能用 meta-test 选择 checkpoint。

因此下一步最值得做的不是改算法，而是：

```text
给 evaluation CLI 增加 validation split
```

完成后就可以严格执行本 Runbook。

---

# 22. 本版相对上一目标文档的更新

本版只同步最新代码与最新 Gate A 结果，没有改变主算法设计。

主要更新：

1. 代码基线从 `1845d0b6` 更新为 `c467ccfc`；
2. Gate A 已从旧的 `cross/within` 统计彻底切换为 aligned `failure disagreement`；
3. Gate A 只评估 4 个 meta-train IDM profiles；
4. 文档中的代码硬阈值修正为：
   - `mean_failure_disagreement >= 0.10`
   - `valid_rate >= 0.80`
   - `0.02 <= failure_rate <= 0.80`
5. 记录最新 Gate A 实际结果，并明确当前仍未通过；
6. 明确当前 failure/valid 分布健康，因此当前不建议先动 scene parameter ranges；
7. 修复 Scientific Gates 表中残留的旧 `cross/within` 条目；
8. 修复 Markdown 中 `\boxed{}` 的转义损坏；
9. Inner、Posterior、Calibration、Outer 的命令、预算和训练顺序保持不变；
10. validation evaluator 仍是正式 early stopping 前需要补齐的 P0 工程项。

---

# 23. 最终运行原则

不要问：

> “还要再训练多少 episode？”

统一问：

```text
Gate 是否通过？
validation 是否继续改善？
z 是否真的改变决策并带来收益？
```

决策链：

```text
Gate A fail
→ 停止

Gate A pass
→ Inner

G4 fail
→ 停止

G4 pass
→ Posterior

G5/G6 fail
→ 停止

G5/G6 pass
→ Calibration

correct-z 对 Inner 有用
→ Outer

Validation plateau
→ Early stop

全部冻结
→ Final meta-test 一次
```

**推荐核心运行策略：`block-wise training + Scientific Gate + validation early stopping`，而不是无条件跑完整训练链。**
