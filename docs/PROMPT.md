# Codex 执行提示词：FM²-FBT 单场景 S01 首轮实现与验证

> 执行约定更新：S01 使用完整 2,048 点固定候选库；六个历史 source 与目标 SUT 均在该库全量物理执行。方法探索新目标时每次只可查询 50 点。本轮训练和选择器各先运行一个固定随机种子。

## 0. 你必须先阅读的两份规格

把下面两份 Markdown 作为本轮实现的**唯一最新研究规格**：

1. `FM2_FBT_Research_Plan.md`  
   - 主规格。
   - 决定 FM²-FBT 的研究目标、三个组件、变维编码、历史记忆、target few-shot adaptation、DFE、两臂路由、B=50 与评价协议。

2. `FBRT_Scenario_Parameter_Space_V3.md`  
   - 场景参数契约。
   - 决定功能场景参数名称、语义、范围、维数、Sobol 候选与 per-family B=50 组织方式。


旧文件只能用于理解历史，发生冲突时以 **V7.2 + V3.1 B50** 为准。

---

# 1. 本轮唯一目标

先不要实现 8 个功能场景。

只在 **S01 邻车切入（Cut-In）** 上把 FM²-FBT 完整跑通，并验证它在同样有限预算下是否能够比当前强基线发现更多**目标 SUT 的真实 failure test cases**。

本轮不是最终论文 confirmation，而是：

> **single-family development pilot**

本轮可以用于方法开发和诊断，之后不能再把同一 target bank 当作独立 confirmation。

不要为了“让新方法赢”读取未查询 target labels 调模型、改场景、删困难样本或挑结果。  
目标是**验证是否存在清晰优势**；如果没有，就如实报告并定位根因，暂时不要扩展到 8 个场景族。

---

# 2. 运行环境与 GPU

所有 Python、训练、仿真和测试都在以下环境执行：

```bash
conda activate metadrive
```

首先记录：

```bash
which python
python --version
conda info --envs
git status --short
git rev-parse HEAD
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
PY
```

要求：

- FM² 神经网络训练默认使用 `cuda`；
- 不要静默退回 CPU；
- 如果 GPU / CUDA 不可用，先停止训练并输出诊断；
- highway-env 的物理执行本身不要求 GPU；
- 不要重新训练 PPO，本轮只测试现有 SUT。

所有环境信息保存到：

```text
results/method_chains/failure_memory_regression/fm2_s01_full_history/environment.json
```

---

# 3. 先审计当前本地代码，不要假定远端快照就是用户本地状态

当前可公开核验的 GitHub main 曾位于：

```text
68db2f767ef1b5495072ecaa94c5d48faea0a6e7
```

但用户本地可能已经更新。

首先检查是否存在：

```text
methods/failure_memory_regression/configs/scenario_parameter_space_v3.yaml
methods/failure_memory_regression/prepare_parameter_space_v3.py
```

并检查 S01 的 V3 4D 参数是否真的被 runner 使用：

```text
initial_clearance_m
lead_speed_mps / cut-in vehicle speed 对应字段
lane_change_time_scale_s
event_start_s
```

不要只看 YAML。

至少做 4 个极值 smoke tests，逐轴修改一个参数，确认：

- 初始化状态确实变化；
- lead actor speed 确实变化；
- lane-change duration 确实变化；
- event start 确实变化；
- 没有 initial overlap；
- event 实际发生；
- episode outcome 可用。

如果本地不存在 V3 YAML / generator，或者 S01 新轴实际上没有接到 runner：

> 停止正式 pilot，先补齐并测试场景执行契约。不要自己猜字段名后直接跑实验。

---

# 4. 为什么第一轮选择 S01

只做：

```text
S01 = fbrt_cutin
```

原因：

- 当前 unified runner 的 cut-in 路径成熟；
- S01 没有 S02 已知停车几何问题；
- 没有 S04/S09 功能恢复终点混杂；
- 没有 S05 自主换道能力契约的额外风险；
- 现有历史 FBRT 数据中 S01 已经存在 failure/pass 结构；
- V3.1 将 S01 定义为 4D，适合验证新的 schema-aware token encoder。

当前 S01 V3.1 参数契约必须以文档/YAML为准。

---

# 5. 候选池

优先复用本地已经冻结的 S01 V3 Sobol manifest。

若必须生成：

- scrambled Sobol；
- S01 共 2,048 个 candidate；
- seed 必须写入 protocol；
- 参数范围严格来自 V3.1；
- 不得看 target outcome 后重新生成候选；
- manifest 一旦 target 测量开始，禁止覆盖。

输出：

```text
results/method_chains/failure_memory_regression/fm2_s01_full_history/
    scenario_manifest.jsonl
    protocol.json
    freeze_audit.json
```

`protocol.json` 至少记录：

```text
family = S01
candidate_count = 2048
budget = 50
budget_checkpoints = [5, 10, 20, 30, 50]
manifest hash
scenario schema hash
runner hash
selector/model source hash
simulator seed / realization rule
target build
historical build list
```

---

# 6. 开发 target 与 historical SUT 组织

## 6.1 第一轮 target

本轮固定使用：

```text
target_build = merge_blind06
```

理由：

- 它与 S01 cut-in 的交互机制高度相关；
- 适合做算法 bring-up；
- 它是 development target，不是最终论文 confirmation target。

**固定后不要因为结果不够好而在同一个 pilot 中换 target。**

如果它在冻结 S01 candidate pool 上真实 failure 数为 0：

- 如实报告 zero-failure task；
- 当前 pilot 无法比较 failure discovery；
- 结束本轮；
- 另开新的 development protocol 再选择其他 target；
- 不把新 protocol 冒充原 pilot。

## 6.2 Historical SUTs

绝不能把 `merge_blind06` 的 target full-bank labels 放进历史 memory 或训练 episode。

先扫描现有 archive / response banks，优先使用对 S01 有真实历史执行记录的兼容 SUT。

建议预声明候选 source list：

```text
idm_ref
merge_brake2
slow_front_brake2
nl3_v0
nl3_v1
nl3_v2
```

处理规则：

1. target `merge_blind06` 永远排除；
2. 只使用真实执行过的历史结果；
3. 如果旧历史是 2D active parameters，但 S01 V3 是 4D：
   - 从旧 scenario 的 `active_parameters + fixed_context` 恢复有语义的 4D 参数 token；
   - 不把缺失轴伪造为 0；
   - 确实无法恢复时使用 missing mask；
4. source build identity 必须保留；
5. 不把多个 source SUT 的标签假装成一个系统。

上述六个 historical builds 都必须在同一份 2,048 点 S01 V3 candidate manifest 上全量执行，建立各自的历史 response bank；已有相同场景、种子和执行契约的实测结果可复用。历史物理执行不计入新目标的 50 次查询预算，但须写入成本账本。

---

# 7. 新增模块：只实现 S01 pilot 所需的最小完整 FM²

在：

```text
methods/failure_memory_regression/
```

新增：

```text
fm2_schema.py
fm2_features.py
fm2_memory.py
fm2_model.py
fm2_train.py
fm2_dfe.py
fm2_selector.py
fm2_s01_pilot.py
```

不要破坏：

```text
selector.py
pattern_memory.py
bayes_model.py
replay_utils.py
```

旧方法必须继续作为 baseline 可运行。

---

# 8. Component 1：Historical Failure Mechanism Memory

严格按照 V7.2。

## 8.1 Memory primitive

继续复用 `PatternCard` 的：

- failure executions；
- pass contrasts；
- component geometry；
- collision partner role；
- maneuver phase；
- TTC / clearance；
- source SUT；
- interaction semantics；
- evidence IDs。

不要只用 component center。

## 8.2 S01 schema-aware parameter tokens

S01 的每个 parameter 形成一个 token。

每个 token 至少包含：

```text
normalized numeric value / component center
component spread
pass-contrast signed difference
pass-contrast present mask
parameter-name embedding
semantic-group embedding
actor-role embedding（若适用）
```

token width：

```text
d_model = 128
```

要求：

- 参数 token 数可变；
- 参数顺序变化不改变最终结果；
- padding 只用于 batch，必须使用 attention mask；
- 不允许 positional embedding 表示“第几列”。

## 8.3 数值编码

使用 V7.2 的增强数值嵌入：

- normalized scalar；
- piecewise / bin-based numeric embedding；
- parameter semantic embedding。

不要退回“全部标量拼接后过一个 Linear”。

## 8.4 Pattern geometry set encoder

建议：

```text
d_model = 128
num_heads = 4
num_layers = 2
Pre-LN
dropout = 0.1
```

parameter-token set：

```text
[n_param, 128]
```

经 set self-attention + attention pooling：

```text
[1, 128]
```

## 8.5 Mechanism metadata

再融合：

- failure count；
- pass-contrast count；
- TTC summary；
- clearance summary；
- collision role；
- maneuver phase；
- event type；
- source SUT family；
- missing masks。

最终：

```text
PatternCard -> mechanism memory token [128]
```

## 8.6 Historical memory set

多个 memory tokens 再经过 memory-set self-attention：

```text
[K, 128] -> [K, 128]
```

不要先无条件平均成一个历史向量。

候选应该对 memory set 做 cross-attention：

```text
candidate query -> relevant historical memories
```

保存 attention weights 作为 audit artifact，但不要把 attention weight解释为因果重要性。

---

# 9. Component 2：Target-Evidence-Conditioned Few-Shot Adaptation

这是本轮核心创新实现。

## 9.1 Candidate encoder

S01 candidate 也使用相同的 schema-aware parameter tokenizer：

```text
scenario parameters
    -> variable-length parameter tokens
    -> set encoder
    -> query embedding [128]
```

只允许 query 前可知数据。

## 9.2 Target support token

每次真实 target query 后形成：

```text
scenario embedding
PASS / FAIL label
min_ttc
min_clearance
collision role
maneuver phase
completed
missing masks
```

所有 rollout-derived 字段只能在执行后进入 support。

## 9.3 历史-only baseline prediction cache

会话开始时，为所有 candidate 保存：

```text
p_history_0(x)
```

它只能由历史 memory 产生。

当真实 target label 到来时，记录：

```text
history_target_residual = y - p_history_0(x)
```

这是 observed mismatch signal，不是因果标签。

## 9.4 Target-conditioned memory gating

不要只在最终 MLP 后面把 history 和 target support 粗暴拼接。

实现：

1. target support 先对 historical memory 做 evidence conditioning；
2. 为每个历史 memory token 产生 candidate-dependent / support-dependent gate：
   ```text
   g_k(x, D_t) in [0,1]
   ```
3. candidate-to-history attention 使用：
   ```text
   similarity
   + evidence gate
   + source balancing
   ```
4. target support 还单独通过 cross-attention进入预测器。

目标是：

> target 的真实 PASS/FAIL 证据可以改变“当前 candidate 应该继续相信哪些历史 failure patterns”。

不要把 `g_k` 命名为“历史正确概率”；它只是 learned retrieval/fusion coefficient。

## 9.5 Predictor

输出：

```text
failure_logit
failure_probability
```

用于 ranking。

网络建议：

```text
d_model = 128
history attention = 4 heads
support attention = 4 heads
fusion MLP = 256 -> 128 -> 64 -> 1
```

可以参考 V7.2 中提到的轻量参数共享多分支 head，但不要为了模型“新”无限扩大网络。

---

# 10. Episodic Training

仅使用 historical SUTs。

## 10.1 pseudo-target

每个 training episode：

- 固定 family = S01；
- 从 historical SUT list 中留一个作为 pseudo-target；
- 其余 source SUTs 构造 historical failure memory；
- pseudo-target 提供少量 support；
- 预测其余 query failures。

## 10.2 support size

因为最终 B=50，训练不能只覆盖 0~10。

随机采样：

```text
n_support in [0, min(49, available-1)]
```

并提高以下关键 support sizes 的采样概率：

```text
0, 1, 3, 5, 10, 20, 30, 40
```

## 10.3 loss

使用：

```text
BCE / weighted BCE
+
pairwise failure-vs-pass ranking loss
```

形式严格按 V7.2。

不要使用 target `merge_blind06` 调超参数。

模型选择只能依靠 historical pseudo-target validation。

## 10.4 seeds

至少固定：

```text
training_seeds = [7319]
```

本轮先训练 1 个 seed。后续增加种子需要另行记录，不将单种子结果解释为稳定性估计。

主 pilot 可以选择 validation 最好的 checkpoint，但选择规则必须在真正读取 `merge_blind06` target full-bank评价前完成并写入 protocol。

---

# 11. Stage A：先完成 FM²-NoDFE

在实现 DFE 前，先完成：

```text
FM²-NoDFE
```

只允许：

```text
Historical Mechanism Memory
+
Target Few-Shot Adaptation
+
Global failure ranking
```

每轮选择：

```text
argmax p_target_failure(x)
```

在冻结的 target bank 上 replay B=50。

如果 FM²-NoDFE 完全不比旧 FBRT 有竞争力：

- 先诊断 component 1/2；
- 不要立即用 DFE 掩盖问题。

---

# 12. Component 3：Diverse Failure Expansion（DFE）

只有 Stage A 代码正确后加入。

## 12.1 不做 grid search

S01 V3 candidate 是不规则 Sobol pool。

不能使用：

```text
grid_index
4-neighbor
```

DFE 必须直接在 normalized 4D S01 Sobol candidate coordinates 中工作。

## 12.2 failure components

新 target failures 按局部近邻形成 target failure components。

局部尺度只用 candidate geometry 预先计算，不读取 target truth。

建议先计算：

```text
h = median nearest-neighbor distance of S01 candidate pool
```

开发起点：

```text
component radius = 1.5 * h
local kernel sigma = 2.0 * h
```

这些只是 development defaults。

## 12.3 representative seeds

每个 component 最多保留 4 个 farthest-spread failure representatives。

不要每个失败都创建独立 expansion process。

## 12.4 local score

严格实现 V7.2：

```text
target failure probability
× locality around real target failure
× direction novelty
```

direction 只在 S01 的 normalized numeric parameter space 中计算。

## 12.5 pass barrier

Local ray 出现 pass 后：

- 降低同方向更远候选的 local priority；
- 不从 global pool 删除；
- 不把它们标成安全；
- 不继续做高成本精确边界二分。

---

# 13. Global / Local Router

实现两个 candidate proposer：

```text
G = global top-risk candidate
L = DFE local candidate
```

不是两个模型。

使用 V7.2 中的折扣 Beta-Bernoulli Thompson router。

基本规则：

- 无 target failure seed -> 必须 G；
- 无合法 local candidate -> 必须 G；
- G 与 L 返回同一个 candidate -> 只执行一次；
- target FAIL -> 所选 arm reward=1；
- valid PASS -> reward=0；
- inconclusive -> query 仍计费，reward=0，但不加入二值 support；
- 使用折扣使旧收益逐渐衰减。

将每一步保存：

```text
selected_arm
theta_global
theta_local
candidate_global
candidate_local
actual_candidate
outcome
posterior alpha/beta before/after
```

---

# 14. 目标 response bank 与信息隔离

为了计算 ground truth，必须完整执行 `merge_blind06` 在 2,048 个固定 S01 candidate 上的结果。

但是：

```text
full_response_bank.jsonl
```

必须 evaluator-only。

selector 只能通过：

```python
TargetOracle.query(scenario_id)
```

获得一个真实 target label。

严禁：

- 训练前读取 target failure count；
- 根据 target bank 调超参数；
- 根据 target truth 决定 history memory；
- 根据 target truth 修改 DFE neighborhood；
- 根据 target truth选择模型 checkpoint。

完整 bank 只用于最后计算：

```text
GT failures
D@5
D@10
D@20
D@30
D@50
Recall@50
HitRate@50
```

---

# 15. Baselines

同一个 S01 candidate pool、同一个 target、同一个 B=50。

必须实现/复用：

```text
Random
FailureDistance-v2
HistoryRank-UCB-v2（S01 single-family 适配）
FBRT-Memory-Exploit-v3（target-failure reward 适配）
FM²-NoDFE
FM²-FBT
```

如果 IFR-DSB adapted 已经容易复用，也加入：

```text
IFR-DSB-adapted
```

但不要因为 IFR 适配工程量阻塞第一轮 FM² pilot。

旧 FBRT 必须使用：

```text
valid target failure
```

作为本轮 reward，不再要求 parent-pass regression。

所有 baseline 也禁止提前看 target full bank。

---

# 16. Logical repeats

物理 response bank 只执行一次。

对带随机性的 selector 使用：

```text
selector_repeats = 1
```

这些重复：

- 只 replay 同一 frozen target bank；
- 不增加独立物理样本；
- 本轮只报告该种子的实际发现数，不估计跨种子方差；
- Random 也只使用 1 个固定 seed。

不要将 selector repeat 当成独立车辆试验。

---

# 17. 本轮主结果

只做一张主表：

| Method | D@5 | D@10 | D@20 | D@30 | D@50 | Recall@50 | HitRate@50 |
|---|---:|---:|---:|---:|---:|---:|---:|

另做一张图：

```text
Cumulative confirmed target failures vs query budget
```

每个点都是**真实 target failure**，不是模型预测。

附加输出：

```text
GT failure count / 2048
first failure rank
queries until 5 failures
global/local query counts
global/local failure yields
```

---

# 18. “卓越能力”的判断方式

本轮是 development pilot。

目标是验证 FM² 是否表现出**清晰且可解释的优势**，不是预先保证它赢。

禁止：

- 看 target truth 后改 target；
- 调参数直到 target 上获胜；
- 删除 FM² 表现差的候选；
- 只展示对 FM² 有利的随机 seed。

开发 gate：

1. FM²-NoDFE 至少应对当前 FBRT-v3 有竞争力；
2. FM²-FBT 应检验 DFE 是否进一步增加 D@50 或更早的 D@20/D@30；
3. 最重要的是比较：
   ```text
   FM²-FBT vs strongest existing baseline
   ```
4. 如果没有优势：
   - 输出 root-cause report；
   - 分析历史 memory retrieval、support adaptation、DFE yield；
   - 不自动扩展到其他 7 个 family。

如果出现优势，也只能称：

> **S01 development evidence**

不能写成跨场景、跨SUT的一般结论。

---

# 19. 必须新增的测试

至少添加：

## schema / feature tests

```text
test_parameter_order_invariance
test_padding_mask_invariance
test_s01_four_dimensional_schema
test_missing_historical_axis_mask
test_no_target_rollout_feature_before_query
```

## memory tests

```text
test_patterncard_to_memory_token
test_memory_permutation_invariance
test_no_history_representation
test_source_target_exclusion
```

## target support tests

```text
test_support_only_after_query
test_pass_and_fail_support_both_used
test_history_target_residual_uses_frozen_history_prediction
test_target_support_changes_candidate_scores
```

## DFE tests

```text
test_dfe_same_family_only
test_dfe_no_duplicate_query
test_direction_novelty
test_pass_barrier_only_penalizes_local
test_multiple_failure_components
```

## protocol tests

```text
test_budget_exactly_50
test_full_bank_not_selector_visible
test_target_build_excluded_from_training
test_all_methods_share_same_manifest
```

---

# 20. 输出目录

不要覆盖任何旧结果。

使用：

```text
results/method_chains/failure_memory_regression/fm2_s01_full_history/
```

至少生成：

```text
environment.json
protocol.json
freeze_audit.json
scenario_manifest.jsonl

history/
    source_inventory.json
    pattern_cards.jsonl
    cost_ledger.json

model/
    config.json
    training_log.csv
    checkpoint.pt
    validation_summary.json

target/
    full_response_bank.jsonl
    cost_ledger.json

evaluation/
    queries_<method>.jsonl
    summary.csv
    repeats.csv
    figure_failures_vs_budget.png
    retrieval_audit.jsonl
    router_audit.jsonl

report.md
```

---

# 21. 最终 report.md 必须回答

1. S01 2,048 个 candidate 中真实 target failures 有多少？
2. FM² 在 5/10/20/30/50 次预算时分别发现多少？
3. 最强 baseline 是谁？
4. FM²-NoDFE 是否超过旧 FBRT？
5. 加 DFE 后是否进一步提高 failure discovery？
6. target feedback 是否实际改变了 historical-memory retrieval？
7. Global / Local 各用了多少预算、各自命中了多少 failure？
8. 如果没有优势，瓶颈在哪里？
9. 所有 target full-bank 泄漏检查是否通过？
10. 下一步是否值得扩展到第二个 family？

---

# 22. 实现风格

- 优先复用现有代码，不复制一套平行 simulator；
- 不修改旧冻结结果；
- 不破坏旧 baseline；
- 新代码写清类型、docstring、seed；
- 所有超参数进入 config；
- 不硬编码 target truth；
- 不把模型预测当作已确认 failure；
- 不把 attention weight 当作因果解释；
- 所有 target execution 都必须计费；
- 如果发现现有研究方案与实际代码接口冲突，优先报告冲突并做最小、可审计的修复，不静默改变研究协议。

---

# 23. 执行顺序

严格按以下顺序完成，不要一次性盲跑：

```text
A. environment/GPU audit
B. local V3 S01 execution-contract audit
C. freeze 2,048-candidate S01 manifest
D. historical source inventory
E. build S01 historical PatternCards
F. implement schema-aware memory encoder
G. implement target-conditioned few-shot predictor
H. unit tests
I. historical episodic training on GPU
J. freeze selected checkpoint before target full-bank evaluation
K. build target full response bank
L. replay FM²-NoDFE + strong baselines at B=50
M. only then add DFE/router
N. replay FM²-FBT
O. generate report and figure
```

完成每一阶段后记录状态；如果关键审计失败，不要绕过。

---

# 24. 最终交付

请直接修改仓库并运行上述 pilot。

最终向用户交付：

1. 修改/新增文件列表；
2. 精确运行命令；
3. GPU与环境记录；
4. 测试结果；
5. `report.md` 路径；
6. 主表；
7. 主图；
8. FM² 是否相对最强 baseline 出现清晰优势；
9. 若无优势，明确指出下一步应该改 Component 1、Component 2 还是 Component 3；
10. 不要用“理论上应该更好”替代真实结果。
