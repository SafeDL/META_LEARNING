# 有限预算危险场景发现

唯一设计文档为 [Failure_Discovery_Design.md](../../docs/Failure_Discovery_Design.md)。本目录保留两条可执行路线：Student-t 风险预测，以及在线反馈支持的选择纠正。当前没有经独立验证的显著优势。

## 实现与入口

| 文件 | 职责 |
|---|---|
| `config.py`、`pools.py` | 六个固定 SUT、两类场景、共享场景池与 B=200 的配置 |
| `prepare.py`、`interface_audit.py`、`validate_banks.py` | 物理仿真、接口检查与实测池核对 |
| `risk_task_prior.py`、`student_risk_prior.py`、`risk_event_response.py` | 风险均值校正、Student-t 尺度后验与风险到事件概率的映射 |
| `frozen_coupling.py` | 独立参照回放与实际发现计数保护；Student-t 排程沿用该阶段冻结实现 |
| `censored_margin_response.py`、`ep_margin_response.py` | 历史数值余量模型及联合 EP 推断 |
| `event_report_response.py` | 仅用碰撞符号更新的事件模型；失败的风险顺序约束已删除 |
| `adaptive_margin_prior.py` | 按查询前预测持续更新模型可信度；已删除十次后固定解释与分场景共享分支 |
| `decision_support.py`、`discovery_supported_coupling.py` | 联合二元发现损失、候选选择与参照回退 |
| `evaluation.py`、`feedback_audit.py` | 唯一查询披露、固定指标与实际反馈核查 |
| `evaluate_student_risk.py`、`evaluate_feedback_correction.py` | 两条保留路线的评价入口 |
| `baselines.py`、`summarize.py` | 14 个冻结基线的执行与指标重算 |

在项目根目录使用 `conda activate metadrive`。按需要单独运行：

```powershell
python -B -m research.representative_sut_testing.validate_banks
python -B -m research.representative_sut_testing.summarize
python -B -m research.representative_sut_testing.evaluate_student_risk
python -B -m research.representative_sut_testing.evaluate_feedback_correction
```

`prepare` 会执行物理仿真；评价入口读取已有实测池，只按实际查询披露 R/C。初始化十次计入总预算 200，禁止重复计费。清理过程中没有启动新实验。

清理后的评价分别写入 `results/student_risk/` 和 `results/feedback_correction/`，执行时才创建目录，避免覆盖清理前的源码快照和曲线。反馈纠正默认采用全局在线权重、历史数值 EP 与仅符号模型；数值 EP 的既有结果作为直接控制。

## 数据与结果

- [实测协议](results/protocol.json)、[接口审计](results/interface_audit.json)、[实测池核对](results/bank_audit.json)：六 SUT × 两池，共 24576 次物理执行。
- [全部基线](results/baseline_comparison.md)、[逐池指标](results/baseline_details.md)：14 个基线、840 条五种子曲线。
- [当前重点方法比较](results/current_method_comparison.md)：同一选择种子、同预算、同初始化；开发均值，不与五种子表混用。
- [Student-t](results/student_risk_coupling/report.md)、[高斯任务均值](results/earned_discovery_credit_coupling/report.md)、[事件评分](results/event_evidence_prior_coupling/report.md)、[仅事件符号](results/safe_risk_order_coupling/event_sign/report.md)：保留较强版本及必要控制的原始轨迹。
- [阶段记录](results/stage_report.md)、[清理记录](results/cleanup/report.md)、[删除清单](results/cleanup/manifest.json)。

旧方法与原结果的目录名、SUT ID、选择种子均属于冻结记录，保留原名以便核对。新增源码按职责命名，不使用数字版本号；旧 `safe_risk_order_coupling/event_sign/` 是仅符号控制的历史目录，不再提供风险顺序实验入口。

## 已退役分支

历史收益学习、旧前瞻、响应通道扩展、独立二元核、风险顺序约束与证据共享范围等失败尝试已退出活动代码。其源码、配置和全部原始结果完整归档到 [retired_experiments.zip](results/retired_experiments.zip)，各阶段保持原相对路径。61 个退役源码和测试文件另有逐字验证的 [清理前源码快照](results/cleanup/source_inputs.zip)。归档是证据保存，不是活动兼容层。

实测池、840 条基线日志、已有历史训练数据和当前必要对照均保留。旧完整主比较 [comparison.md](results/comparison.md) 仍是冻结历史结果，不代表最新方法已经胜出。

池目录里的旧 `mechanisms`、`local_mechanisms`、`probit_mechanisms` 及风险关系纠正日志也已完整转入同一退役归档。较强的 `risk_relation_mechanisms/pooled_reference_*.json` 共 60 条继续保留，它使用全部历史坐标，按额外历史数据工程参照单列。当前状态只维护在 `results/stage_status.json`；完整阶段历史见阶段记录和归档。
