# 最新研究 Idea 与实现目标
## Population-Referenced Bayesian Regression Vulnerability Testing
### 面向自动驾驶系统版本迭代的群体参考式贝叶斯回归脆弱性测试

> 版本：2026-09-13  
> 适用代码库：`SafeDL/META_LEARNING`  
> 当前核心目标：停止继续优化原始 DIVA-Mine / PR-DVM v1，转向一个与现有实验结果、历史 SUT 数据价值和原始研究动机更一致的问题——**快速发现新 SUT 相对于历史 SUT population 新增或异常放大的 vulnerability**。

---

# 1. 当前结论：哪些方向已经不值得继续

经过当前 Highway-env 的多轮验证，可以明确停止以下主线：

- 不再继续把 **普通 `CriticalScore@20` 最大化** 作为 target-specific few-shot adaptation 的主要目标；
- 不再使用固定 `K=4` 的“先诊断、后挖掘”两阶段流程；
- 不再继续优化 `diagnostic_support_indices()`、boundary-aware information gain、Teacher、Transformer；
- 不再把 Meta-RL / SAC / PPO 作为核心方法；
- 不再通过增加网络容量、增加 K、继续堆 acquisition term 来挽救原方法。

原因已经明确：

1. 修复后的 E6 中，Shared Prior 对普通 failure mining 已接近 outcome oracle；
2. 当前普通 `CriticalScore@20` 的理论提升空间极小；
3. PR-DVM v1 虽然显著提高 differential score，但显著牺牲 raw critical score；
4. 因此，“target adaptation 用于普通 failure count 最大化”缺乏足够收益空间。

---

# 2. 最新研究问题

## 2.1 不再问“新 SUT 哪里危险”

历史多个 SUT 的测试档案已经能告诉我们大量公共高风险区域。

如果新 SUT 在这些区域再次发生 collision，这对于“寻找新漏洞”来说信息价值有限。

因此最新问题改为：

> **给定多个历史 SUT 的 vulnerability landscape，当一个新 SUT / 新版本到来时，如何用极少测试预算，优先发现该系统相对于历史 population 新出现、异常增强或此前少见的 failure-prone region？**

这不是普通 failure mining，而是：

## Few-Shot Regression Vulnerability Discovery

或更正式地：

## Population-Referenced Bayesian Regression Vulnerability Testing

适用场景包括：

- ADS 软件版本迭代；
- 控制器更新；
- 规划模块升级；
- 参数更新；
- 第三方回归测试；
- 历史测试资产复用。

---

# 3. 核心研究动机

原始研究动机可写成：

\[
R_u(x)=R_{\text{shared}}(x)+\Delta_u(x)
\]

其中：

- \(R_{\text{shared}}(x)\)：历史多个 SUT 的公共 vulnerability；
- \(\Delta_u(x)\)：目标 SUT 相对于历史 population 的 residual / differential vulnerability。

当前实验已经说明：

- \(R_{\text{shared}}\) 对普通高风险区域预测非常强；
- 因此真正值得 target adaptation 的不是整个 \(R_u\)，而是 \(\Delta_u\)。

所以最新方法的核心不是：

> 重新学习 target risk。

而是：

> **利用历史 population 作为 reference，在线推断 target 相对于 population 的异常 vulnerability。**

---

# 4. 最新方法定义

暂定方法名称：

## PR-BRVT

**Population-Referenced Bayesian Regression Vulnerability Testing**

也可以暂时使用更短的：

## PR-RVD

**Population-Referenced Regression Vulnerability Discovery**

---

# 5. 方法核心结构

整个方法只保留三个组件：

1. **Population Reference**
2. **Target Residual Posterior**
3. **Regression-Failure Acquisition**

不再需要额外的 diagnosis policy。

---

# 6. Population Reference

对于 source SUT：

\[
u=1,\dots,M
\]

对于场景：

\[
x\in\mathcal X
\]

已有 vulnerability response：

\[
f_u(x)
\]

定义历史 population vulnerability 均值：

\[
\mu(x)=\frac{1}{M}\sum_{u=1}^{M}f_u(x)
\]

同时定义历史 failure prevalence：

\[
\pi_{\text{src}}(x)
=
\frac{1}{M}
\sum_{u=1}^{M}
\mathbf 1[r_u(x)>0]
\]

其中：

\[
r_u(x)=
\begin{cases}
1,& collision\\
0.5,& critical\ near\ miss\\
0,& otherwise
\end{cases}
\]

含义：

- \(\pi_{\text{src}}(x)\approx1\)：历史系统普遍失败，属于公共危险；
- \(\pi_{\text{src}}(x)\approx0\)：历史系统大多安全，如果新 SUT 在此失败，则更可能是 regression vulnerability。

为了减少小样本 source 数带来的硬 0/1，建议保留 Beta smoothing：

\[
\tilde\pi_{\text{src}}(x)
=
\frac{n_{\text{fail}}(x)+1}{M+2}
\]

---

# 7. Target Residual Prior

继续复用当前 low-rank prior：

\[
f_u(x)=\mu(x)+b(x)^\top z_u+\epsilon
\]

也可写成：

\[
\Delta_u(x)=b(x)^\top z_u+\epsilon
\]

其中：

- \(b(x)\)：历史 population 中学到的 vulnerability basis；
- \(z_u\)：目标 SUT 的 latent vulnerability profile；
- \(\Delta_u(x)\)：target-specific residual vulnerability。

当前代码中的 centered SVD 本质上已经实现：

```python
centered = responses - mean
```

因此：

**当前 `LowRankPrior` 不需要推翻。**

---

# 8. Target Posterior

继续复用当前 Bayesian posterior：

\[
p(z_{u^*}\mid D_t)
=
\mathcal N(m_t,\Sigma_t)
\]

其中：

\[
D_t=\{(x_i,y_i)\}_{i=1}^{t}
\]

关键变化：

- 不再固定 `K=4`；
- 不再划分 support / mining 两个阶段；
- 每一次 target test 都会：
  1. 产生真实 safety outcome；
  2. 更新 posterior；
  3. 参与最终 testing utility。

---

# 9. Regression Failure Probability

当前 posterior 对场景 x 给出：

\[
f_{u^*}(x)\mid D_t
\sim
\mathcal N(
\hat\mu_t(x),
s_t^2(x)
)
\]

其中：

\[
\hat\mu_t(x)
=
\mu(x)+b(x)^\top m_t
\]

\[
s_t^2(x)
=
b(x)^\top\Sigma_t b(x)+\sigma^2
\]

当前 vulnerability semantics 中，critical boundary 取：

\[
\gamma=0.75
\]

因此 target critical probability：

\[
p_t(x)
=
P(f_{u^*}(x)\ge\gamma\mid D_t)
\]

即：

\[
p_t(x)
=
1-\Phi
\left(
\frac{\gamma-\hat\mu_t(x)}
{s_t(x)}
\right)
\]

这里 posterior uncertainty 已经自然进入概率计算。

因此：

**不再额外加入 `+ beta * sigma` uncertainty bonus。**

---

# 10. Regression Vulnerability Acquisition

最终 acquisition 简化为：

\[
\boxed{
a_t(x)
=
p_t(x)
\left[
1-\tilde\pi_{\text{src}}(x)
\right]
}
\]

解释：

- target 越可能 failure，分数越高；
- 历史系统越少 failure，分数越高；
- 只有“target 高风险 + source population 低风险”时，才成为高价值 regression candidate。

这就是整个方法的核心。

---

# 11. 为什么取消 additive uncertainty bonus

PR-DVM v1 使用过类似：

\[
a(x)=
\hat f(x)(1-\pi_{\text{src}}(x))
+0.1\hat f(x)
+0.25\sigma(x)
\]

该结构的问题是：

> 很不确定但实际并不危险的场景，也可能因为 uncertainty bonus 被过度选择。

这会造成：

- differential score 上升；
- 但 raw failure discovery 大幅下降。

新方法中 uncertainty 不再单独奖励，而是进入：

\[
P(f\ge\gamma)
\]

因此 uncertainty 只在“它会改变 failure probability”时才有意义。

---

# 12. Sequential Testing

整个流程统一为：

\[
x_t
\rightarrow
y_t
\rightarrow
p(z\mid D_t)
\rightarrow
p_{\text{reg}}(x)
\rightarrow
x_{t+1}
\]

没有：

- fixed K；
- diagnosis phase；
- freeze posterior；
- exploitation phase。

每个 test 都同时承担：

1. 发现 regression failure；
2. 更新 target posterior。

---

# 13. 最新实现目标

## 13.1 保留

继续保留：

```text
mvr/highway/envs/cutin_env.py
mvr/highway/data/response_bank.py
mvr/highway/sut/idm_profiles.py
mvr/highway/diva/low_rank_prior.py
mvr/highway/diva/posterior.py
```

继续复用：

```python
adapt_posterior(...)
```

## 13.2 降级为历史 baseline

以下不再属于主方法：

```text
variance_support_indices
boundary_weight
boundary_aware_support_index
diagnostic_support_indices
oracle_support_indices
diagnostic_mining
highest_risk_support_mining
PR-DVM v1 additive acquisition
```

保留用于：

- 旧实验复现；
- appendix；
- baseline。

---

# 14. 需要新增/修改的代码

建议新增：

```text
mvr/highway/diva/regression_reference.py
mvr/highway/diva/regression_probability.py
mvr/highway/diva/regression_mining.py
mvr/highway/experiments/run_loso_regression_mining.py
mvr/highway/scripts/run_pr_brvt_mvp.py
```

---

# 15. `regression_reference.py`

职责：

- 计算 source-only population failure prevalence；
- 进行 Beta smoothing；
- 严格 LOSO，禁止 target leakage。

建议接口：

```python
@dataclass(frozen=True)
class RegressionReference:
    mean_vulnerability: np.ndarray
    failure_prevalence: np.ndarray
    smoothed_failure_prevalence: np.ndarray
```

构建：

```python
def build_regression_reference(
    source_vulnerability,
    source_collisions,
    source_near_misses,
    alpha=1.0,
    beta=1.0,
) -> RegressionReference:
    ...
```

---

# 16. `regression_probability.py`

职责：

根据当前 posterior 计算：

\[
p_t(x)=P(f_{u^*}(x)\ge0.75\mid D_t)
\]

建议核心函数：

```python
def critical_probability(
    prior,
    posterior,
    threshold=0.75,
):
    ...
```

输出：

```text
shape = [num_anchors]
```

---

# 17. `regression_mining.py`

主方法：

```python
for t in range(total_budget):

    posterior = current target posterior

    p_target = critical_probability(...)

    regression_score = (
        p_target
        * (1.0 - reference.smoothed_failure_prevalence)
    )

    choose highest untested scenario

    reveal target outcome

    update posterior
```

禁止：

- additive uncertainty；
- shared-risk bonus；
- fixed diagnosis count。

---

# 18. 主评价目标

最新主任务已经不再是普通 failure count。

因此主指标改为：

## Regression Critical Score

\[
RCS@B
=
\sum_{t=1}^{B}
r_{u^*}(x_t)
\left[
1-\tilde\pi_{\text{src}}(x_t)
\right]
\]

它衡量：

> 测试预算中发现了多少“历史 population 不常见，但 target 实际发生”的 critical events。

---

# 19. 辅助指标

仍然报告：

### 19.1 Target-Specific Failure Count

\[
TSF@B
=
\#\{
x_t:
r_{u^*}(x_t)>0,
\pi_{\text{src}}(x_t)\le0.2
\}
\]

### 19.2 Raw CriticalScore@B

继续报告：

```text
collision = 1
near-miss = 0.5
otherwise = 0
```

但：

**Raw CriticalScore 不再作为 pass/fail gate。**

原因：

> regression testing 的目标不是重复发现所有历史版本都已经知道的公共 failure。

---

# 20. 最小 baseline

只保留：

## Random

回答：不使用历史知识时如何？

## Shared Prior

回答：只找公共高风险场景时如何？

## Population-Novelty Only

\[
a(x)=
\mu(x)
(1-\tilde\pi_{\text{src}}(x))
\]

回答：不使用 target posterior，仅用历史 population 是否已经足够？

## Proposed：Bayesian Regression Vulnerability

\[
a_t(x)
=
P(f_{u^*}(x)\ge0.75\mid D_t)
(1-\tilde\pi_{\text{src}}(x))
\]

回答：target feedback 是否真正提高 regression vulnerability discovery？

---

# 21. 唯一主 Gate

## Gate BRVT-1

要求：

\[
RCS_{\text{Proposed}}
>
RCS_{\text{Population-Novelty}}
\]

并且：

\[
RCS_{\text{Proposed}}
>
RCS_{\text{Shared}}
\]

同时至少：

```text
4 / 6 target SUT
Proposed RCS@20 >= Population-Novelty Only
```

辅助要求：

```text
mean TSF@20 proposed
>=
mean TSF@20 population-novelty
```

不再设置 RawScore 的 90% 护栏。

---

# 22. 最新的停止规则

如果：

\[
RCS_{\text{Proposed}}
\le
RCS_{\text{Population-Novelty}}
\]

则说明：

> target Bayesian adaptation 没有带来额外 regression-testing utility。

此时：

**停止 few-shot target adaptation 主线。**

不再：

- 加 CNP；
- 加 Transformer；
- 调 threshold；
- 加 acquisition bonus；
- 扫 beta；
- 加更多 K。

---

# 23. 本轮实验资源原则

第一轮仍然：

```text
新增 Highway-env episode = 0
新增 GPU training = 0
新增 response bank = 0
```

直接复用：

```text
results/diva_highway/cutin_mvp_e6_action_fix/response_bank_highway_e6.npz
```

先验证方法定义是否成立。

---

# 24. 代码验收

必须新增单元测试：

## 24.1 target leakage

target row 不参与：

```text
population prevalence
population mean
prior basis
latent covariance
```

## 24.2 critical probability

posterior mean 越高，`P(f >= 0.75)` 应越高。

posterior uncertainty 对 threshold 附近概率产生合理影响。

## 24.3 regression ranking

若：

```text
x0:
target failure probability = 0.9
source prevalence = 0.9

x1:
target failure probability = 0.8
source prevalence = 0.1
```

则必须：

```text
score(x1) > score(x0)
```

## 24.4 sequential reveal

第 `t+1` 次选择只能使用前 `t` 次 target outcomes。

## 24.5 no duplicate queries

每个 anchor 最多执行一次。

---

# 25. 论文定位

最新问题不再建议写：

> few-shot adaptive failure mining for arbitrary unseen ADS

更推荐：

> **Population-Informed Regression Vulnerability Testing for Updated Autonomous Driving Systems**

或者：

> **Few-Shot Bayesian Regression Vulnerability Discovery for Autonomous Driving System Updates**

---

# 26. 应用故事

现实中的使用场景：

```text
已有：
ADS v1.0
ADS v1.1
ADS v1.2
ADS v1.3

每个版本都已有 historical testing archive

现在：
ADS v1.4 发布

目标不是：
重新证明所有版本在明显危险 Cut-in 下都会失败

而是：
尽快找出 v1.4 新增的、异常放大的、
或者历史版本没有暴露的 failure region
```

这时 historical SUT population 有不可替代的意义：

- 定义什么叫“已知公共 vulnerability”；
- 定义什么叫“新出现 vulnerability”；
- 为新版本建立 prior；
- 用极少 target tests 在线校准。

---

# 27. 与原始研究材料的关系

原始 motivation 的核心：

\[
R_U=R_{\text{shared}}+\Delta_U
\]

旧 DIVA-Mine：

```text
学习 ΔU
但最终仍优化总风险 RU
```

PR-DVM v1：

```text
开始强调 differential risk
但 additive uncertainty 导致过度探索
并且 RawScore gate 与 regression objective 冲突
```

最新 BRVT：

```text
直接建模 target critical probability
× historical rarity

只优化 regression vulnerability
```

---

# 28. 当前证据如何解释

截至目前：

- Shared Prior 对普通 failure mining 很强；
- 普通 `CriticalScore@20` 不适合作为 target-specific adaptation 的主任务；
- PR-DVM v1 的 differential score 显著高于 Shared Prior；
- PR-DVM v1 也高于 Population-Novelty Only；
- 说明 target feedback 对 differential discovery 平均存在额外价值；
- 但 additive uncertainty 与 raw-score trade-off 使 PR-DVM v1 不适合作为最终方法。

因此最新方法不是从零重新猜测，而是针对当前实验结果暴露出的核心问题做结构性收缩。

---

# 29. 最终实现目标

本轮只实现：

- [ ] `regression_reference.py`
- [ ] `regression_probability.py`
- [ ] `regression_mining.py`
- [ ] `run_loso_regression_mining.py`
- [ ] `run_pr_brvt_mvp.py`
- [ ] 对应单元测试
- [ ] 使用现有 action-fix E6 bank
- [ ] 输出：
  - `RCS@20`
  - `TSF@20`
  - `Raw CriticalScore@20`
- [ ] 按 `Gate BRVT-1` 给出 continue / stop

---

# 30. 最终原则

> **不再为了证明 few-shot 而强行诊断；不再为了提高普通碰撞数而做 target adaptation；不再用 additive uncertainty 奖励“仅仅不确定”的场景。**

最新研究主张应收敛为：

> **历史 SUT population 定义“已知公共风险”，目标 SUT 的在线 Bayesian posterior 定义“新版本真实风险”，两者之差用于快速发现 regression vulnerability。**

如果该最小方法仍不能稳定优于 Population-Novelty Only，则应停止 population-to-target few-shot adaptation 主线。

这将是当前项目最清晰、最可证伪、也最符合现有实验事实的研究终点。
