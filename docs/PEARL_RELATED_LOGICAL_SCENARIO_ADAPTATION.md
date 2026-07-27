# PEARL 第二阶段训练目标（Codex Implementation Specification v2）

## 1. 阶段目标

本阶段目标：

> 验证 PEARL 是否能够在固定 SUT、统一动作空间和统一奖励下，通过少量 support episode 适配未见逻辑场景，并优于普通多任务 SAC 和从头训练 SAC。

当前代码基础：
- Task/Case 分离；
- LogicalScenarioTaskSpec；
- Adapter 环境接口；
- PEARL-SAC；
- task replay buffer；
- few-shot evaluation。

本阶段重点不是继续扩展网络，而是修正任务定义和实验闭环。

当前代码的GPU训练环境在系统的:
conda activate metadrive

---

## 2. Task定义

Task必须表示逻辑场景模板：

\[
T_i=(G_i,C_i,R_i,\pi_{SUT})
\]

其中：
- G_i：道路拓扑；
- C_i：冲突区域和交互结构；
- R_i：规则和优先权；
- SUT保持固定。

Task差异来源：

- on_ramp_merge；
- lane_drop_merge；
- bottleneck_merge；
- y_merge。

禁止使用SUT参数作为主要task变化。

---

## 3. Adapter要求

必须独立实现：

```
OnRampMergeAdapter
LaneDropMergeAdapter
BottleneckMergeAdapter
YMergeAdapter
```

每个Adapter必须拥有：

- 独立地图构造；
- 独立route；
- 独立conflict geometry；
- 独立spawn规则。

禁止：

```
lane_drop_merge -> BottleneckMergeAdapter
y_merge -> OnRampMergeAdapter
```

否则无法证明跨逻辑场景泛化。

---

## 4. Case定义

Case只表示同一逻辑任务下的初始条件：

包括：

- adversary speed；
- SUT speed；
- initial gap；
- arrival offset；
- traffic seed。

Case不能改变逻辑场景。

---

## 5. PEARL训练目标

任务后验：

\[
z_i=q_\phi(z|C_i)
\]

策略：

\[
a_t=\pi_	heta(a_t|o_t,z_i)
\]

训练流程：

1. sample meta tasks；
2. prior rollout；
3. context inference；
4. posterior rollout；
5. SAC update。

Context：

\[
(o_t,a_t,r_t,o_{t+1})
\]

保持：

- actor使用detach(z)；
- context encoder接收Q gradient；
- KL regularization；
- meta-test不更新网络。

---

## 6. Few-shot评估修正

必须：

### K=0

直接：

```
z = prior
evaluate query
```

不能提前收集support。


### K>0

循环：

```
evaluate query

collect support episode

update posterior

continue
```

输出：

```
Success@0
Success@1
Success@2
Success@5
Success@10
```

---

## 7. Observation要求

增加消融：

### PEARL-full

```
dynamic state
+
topology descriptor
+
latent z
```

### PEARL-no-topology

```
dynamic state
+
latent z
```

禁止：

- task_id；
- logical_type label；
- template index。

目的：

证明收益来自任务推断，而不是标签泄漏。

---

## 8. 必须增加Baseline

正式PEARL前实现：

### Per-task SAC

验证不同task存在策略差异。

### Pooled Multi-task SAC

验证普通多任务是否不足。

### Scratch SAC

同环境预算重新训练。

### PEARL

比较：

```
PEARL@0
PEARL@5
PEARL@10
```

---

## 9. Task Heterogeneity Audit

新增：

```
audit_task_heterogeneity.py
```

生成：

- policy transfer matrix；
- per-task SAC性能；
- pooled SAC gap。

如果：

```
Pooled SAC ≈ Per-task SAC
```

则task过于简单，不进入正式PEARL实验。

---

## 10. 指标

分别报告：

### Collision

```
target_collision_rate
```

### Critical

```
critical_success_rate
min TTC
```

### Validity

```
valid_critical_strict_rate
invalid_rate
```

禁止只使用collision rate。

---

## 11. 正式训练条件

必须满足：

- 四个Adapter通过topology audit；
- train/test task无泄漏；
- support/query无泄漏；
- posterior随support更新；
- no-gradient adaptation通过；
- baseline完成。

---

## 12. 最终目标

证明：

> PEARL不是简单识别场景类别，而是通过少量交互学习逻辑场景中的风险模式，使场景挖掘器能够在未见逻辑场景上快速发现有效安全关键测试场景。


