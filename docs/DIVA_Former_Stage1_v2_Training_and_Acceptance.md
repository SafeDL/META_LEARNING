# DIVA-Former Stage 1-v2 训练前机制修复与验收协议

**版本**：Stage 1-v2  
**数据范围**：冻结的 4 个 source SUT × 64 common anchors  
**新增仿真预算**：0  
**上游记录**：Stage 1-v1 的 `stop_before_validation_sut` 结论保持不变。

## 1. 目标与边界

Stage 1-v2 修复 Stage 1-v1 中 terminal `FormalScore@20` 饱和和 counterfactual Q 信号退化的问题。证据链改为：

\[
\text{headroom}
\rightarrow\text{diagnosis}
\rightarrow\text{formal alignment}
\rightarrow\text{early mining utility}.
\]

禁止访问 `idm_fast_small_gap` 和 `idm_late_response`，禁止新增 MetaDrive rollout，禁止在 G1-A～G1-D 全部通过前实现或训练 Transformer。正式 failure oracle 与 vulnerability response 合同均不得修改。

## 2. 双预算协议与指标

- `B=10` 是 Teacher 可辨识性和早期收益的主要预算。
- `B=20` 保持与原研究问题可比，用于最终非劣和完整 AUC 验收。
- 必须报告 `Score@5/8/10/20`、`AUC-Formal@10/20`、first-critical-step 和完整累计曲线。

定义：

\[
AUC_B=\sum_{t=1}^{B}FormalScore@t,
\qquad
Headroom_B=Oracle_B-BestBaseline_B.
\]

最强 baseline 必须按聚合指标预先选择，禁止逐 replay 事后选择。

## 3. Formal-aligned surrogate

解析 Teacher 使用仅由当前 fold training sources 拟合的三分类校准器：

\[
P(normal),P(near\ miss),P(collision).
\]

输入仅包含 scenario/candidate 特征及当前 vulnerability posterior 的均值和方差，不得包含 SUT identity。正式即时效用为：

\[
\widehat U_F=0.5P(near\ miss)+P(collision).
\]

formal mining 使用 `U_F × evaluability² × novelty`。连续 vulnerability posterior 继续服务于信息增益、边界辨识和 dense auxiliary value。

## 4. 时间敏感 Teacher

主 Teacher 标签改为 AUC return：

\[
Q^F_{AUC}(s,a,b)=b\,r(a)+\max_{\pi}G^F_{AUC}(\pi,s',b-1).
\]

vulnerability auxiliary Q 使用相同时间权重。保存 raw Q 与除以 `b(b+1)/2` 的标准化 Q；terminal return 只作为审计字段。当 `b=1` 时必须严格满足 `Q^F=r`、`Q^V=v`。

continuation portfolio 同时包含 vulnerability mining 和 formal mining 的 mine-now、diagnose-once、diagnose-twice 路径。强 baseline 至少包括 current fixed-K4、mine-now、formal-mine-now、fixed-K4-formal 和 highest-shared-risk。

## 5. Gates

### G1-A Headroom

在 event-bearing sources 上同时要求：

- `OracleScore@10 - BestBaselineScore@10 >= 0.5`；
- oracle 的 `AUC@10` 与 `AUC@20` 均严格高于最强 baseline；
- 至少 2/3 event-bearing SUT 的 fold-aggregate AUC headroom 为正。

`idm_cautious` 保留在报告和稳定性检查中，但不进入 formal-yield 聚合。

### G1-B Diagnosis

冻结的 source LOSO 结果必须继续满足：diagnostic K=4 vulnerability NDCG 相对 K=0 增益至少 0.05；相对 random K=4 的 paired 95% CI 下界大于 0；RMSE 相对 random 的退化不超过 5%。

### G1-C Formal Alignment

event-bearing held-out folds 必须满足：diagnostic K=4 formal NDCG@10 相对 K=0 增益至少 0.05；相对 random K=4 的 paired bootstrap 95% CI 下界大于 0。全部 folds 的三分类 Brier score相对 K=0 不退化超过 5%，且所有概率有限、非负、和为 1。

### G1-D Mining Utility

Teacher 相对最强非 Teacher baseline 必须满足：

- `Score@10` 不劣，且绝对增益至少 0.5 或相对增益至少 5%；
- `AUC@10` 与 `AUC@20` 均相对提升至少 5%；
- `Score@20` 不劣，first-critical-step 中位数不劣；
- AUC@20 的 fold-aggregate paired gain 至少 3/4 为正；
- 在 `b>=4` 且仍有未发现 formal-positive 的状态中，至少 30% 的 formal 标准化 Q `P90-P10>=0.025`，至少 50% 的 vulnerability 标准化 Q `P90-P10>=0.05`。

## 6. 停止规则

G1-A～G1-D 任一失败，必须输出 `stop_before_transformer`，不得训练 DIVA-Former，也不得访问 validation/test SUT。全部通过只能解锁后续 belief/distillation/policy 工程，不能直接解锁 unseen validation；进入 validation 仍要求完整 Stage 1-v2 gate 全通过。
