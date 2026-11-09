# RF-EI：随机森林与期望改进

RF-EI 用已查询响应拟合风险评分，估计剩余候选的不确定性，按期望改进选择下一场景。

依据 Huang、Sun、Tian 的 *A Bayesian Optimization Method for Finding the Worst-Case Scenarios of Autonomous Vehicles*，IEEE T-ITS 26(1):529–543，2025，[DOI](https://doi.org/10.1109/TITS.2024.3490616)，[核对全文](https://drive.google.com/file/d/1qljUztG9aLFge_M0YrlZD3QoggAcw80D/view)。本实现按正文 III–VI、公式和表 I 核对，并通过独立数值例验证。

| 原文 | 实现 | 作用 |
|---|---|---|
| III，式(1)–(2) | [risk](method.py)、[disclose_impact](../../../experiments/domain_risk.py) | 碰撞时使用首次接触的等质量撞击量，安全时使用负最小 TTC |
| IV-B.2 | `RandomForestEI._forest` | 拟合风险回归模型 |
| IV-B.3，表 I | `RandomForestEI.tune` | 用已付费样本的交叉验证 RMSE 调参 |
| IV-C，式(8) | `expected_improvement` | 计算风险最大化的期望改进 |
| IV-C，式(9)–(11) | `jackknife_variance`、`fit/predict` | 通过删折重训练和伪值估计方差 |
| IV-D | `run` | 在未查询候选上精确最大化 EI |

## 候选库适配

碰撞风险为 `0.5 * ||v_ego - v_partner||`，由评估端从本次执行的首次碰撞帧计算，并通过 `impact_delta_v_mps` 返回。安全风险为负最小 TTC，按 10 秒观察窗截尾。TTC 使用二维定速矩形重叠定义，覆盖切入和障碍交互。

原文的连续差分进化搜索在此替换为有限候选上的 EI 枚举，同分按场景 ID。共同初始化和总预算由统一协议传入；输入默认是场景归一化特征，也可指定 `feature_fn=domain.features`。

默认随机森林为 100 棵树、深度 10、最小分裂数 2。每 50 条已付费反馈执行 20 次 TPE 试验，含当前参数和 10 次启动试验；搜索树数 10–300、深度 3–30、最小分裂数 2–10，使用 10 折平均 RMSE。Jackknife 折数为 `min(10, 已付费样本数)`，调参复用已有响应。

## 接入与验证

入口为 `run(candidates, oracle, initial_indices=..., budget=200, seed=11)`。适配、调参和选例耗时随查询保存，由 [experiments.pilot](../../../experiments/pilot.py)接入。

~~~powershell
python -B -m pytest tests/experiments/test_av_bo_rf_ei.py tests/experiments/test_domain_risk.py -q
~~~

测试覆盖伪值方差、EI 数值积分、调参范围、风险量和反馈隔离。结果见[论文报告](../../../artifacts/runs/joint_response/paper/reports/results.md)。
