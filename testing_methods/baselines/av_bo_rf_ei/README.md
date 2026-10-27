# RF-EI：随机森林与期望改进

本方法根据已查询场景的反馈拟合随机森林，预测剩余候选的风险与不确定性，再按期望改进选择下一场景。查询后更新模型，直到用完预算。

原文：Yan Huang、Jian Sun、Ye Tian，*A Bayesian Optimization Method for Finding the Worst-Case Scenarios of Autonomous Vehicles*，IEEE T-ITS 26(1):529–543，2025，[DOI](https://doi.org/10.1109/TITS.2024.3490616)。全文优先取自[用户目录中的 PDF](https://drive.google.com/file/d/1qljUztG9aLFge_M0YrlZD3QoggAcw80D/view)。复现依据为全文第 III–VI 节、公式和表 I，主要步骤与实现对应如下。

## 原文与实现对应

| 原文 | 实现 | 核对内容 |
|---|---|---|
| III，式(1)–(2) | [risk](method.py)、[disclose_impact](../../../experiments/domain_risk.py) | 碰撞风险为等质量假设下的 `0.5 * ||v_ego - v_partner||`，使用首次碰撞记录的速度和航向；非碰撞为负最小 TTC |
| IV-A，步骤2–8 | [run](method.py) | 共同初始化付费；只使用已查询反馈；每轮更新 SSD，再选择下一项 |
| IV-B.2 | `RandomForestEI._forest` | RF 回归风险，单线程、固定随机种子 |
| IV-B.3，表 I | `RandomForestEI.tune` | 每50个已付费样本调参；10折平均 RMSE；TPE 搜索树数10–300、深度3–30、最小分裂数2–10 |
| IV-C，式(8) | `expected_improvement` | 对最大风险的高斯 EI，含零方差极限 |
| IV-C，式(9)–(11) | `jackknife_variance`、`RandomForestEI.fit/predict` | 完整已查询数据模型加 K 个删折重训练模型，按伪值公式估计方差 |
| IV-D | `run` 中候选 EI 排序 | 将连续 DE 最大化替换为有限剩余候选的精确最大化；保留 RF、Jackknife 与 EI |

## 已声明的任务适配与设置

本实现使用 **RF-EI 候选库适配**：保留随机森林、Jackknife 方差和期望改进，将搜索范围设为本项目的公开候选库。

- 原文初始化为9倍参数维度；本项目固定共同随机初始化8次、总预算128次。初期使用 `K=min(10, 已付费样本数)`，样本达到10后为原文的10折；不为调参额外查询目标。
- 公开特征由本项目 `experiments.scenarios.features` 从候选参数与类别生成，不接收目标身份、档案、未查询标签或轨迹。
- 连续参数的 DE 搜索由剩余候选上的精确 EI 最大化替代，同分按公开场景标识确定。每个库坐标最多查询一次。
- 原文式(3)为纵向前车距离/速度差。本项目使用共同执行接口的二维定速矩形重叠 TTC，涵盖切入和切出障碍交互；TTC 超过10秒（包括无穷大）按10秒仿真观察窗截尾。该二维 TTC 定义与风险函数一并登记。
- 撞击量在评估进程从**本次已付费执行**的首次碰撞帧导出，只通过 `SafetyMeasurements.impact_delta_v_mps` 披露标量。碰撞时缺失该量直接报错，不以碰撞 Boolean 或间距替代撞击严重度。等质量假设也用于障碍碰撞的风险标量。
- 原文未给出可直接核对的初始 RF 参数和 TPE 最大试验次数。本项目登记初值为100棵树、深度10、最小分裂数2；每次20个试验，含当前参数、10次启动试验，其余使用独立 TPE。调参发生在已付费样本数50、100，使用已查询数据的10折平均 RMSE。初始化8次时不进行不可定义的10折调参。
- `Config` 固定所有设置；Optuna 5.0.0 已写入[环境配置](../../../environment.yml)，RF 使用项目已有 scikit-learn 1.7.2。模型训练、调参和候选选例时间分别进入逐次记录的适配/选例时间，真实执行时间保留原账本值。

## 实现核对与使用入口

复现记录中未定位到可核对的作者实现，因此实现核对采用论文公式、独立数值计算和预算隔离测试。

[数值与隔离测试](../../../tests/experiments/test_av_bo_rf_ei.py)以手算伪值、独立数值积分、树间方差对照、调参范围和隔离查询账本验证关键机制。[风险反馈测试](../../../tests/experiments/test_domain_risk.py)覆盖静止对象、同向、正交、对向、首次接触、JSON 往返和不一致记录拒绝。开发仿真库首项及独立回放核查由[阶段 A 入口](../../../experiments/domain_rf_ei_validation.py)执行，输出位于 `experiment_outputs/av_bo_rf_ei/collision/discovery/domain_comparison/stage_a/`。

```powershell
conda activate metadrive
python -B -m pytest tests/experiments/test_av_bo_rf_ei.py tests/experiments/test_domain_risk.py -q -p no:cacheprovider
python -B -m experiments.domain_rf_ei_validation
```

开发期和独立验证均已完成。RF-EI 在全来源条件下有200项独立运行，严格留家族条件复用这些运行；它只依据已查询的目标反馈更新模型。完整范围与结果见 [独立验证报告](../../../experiment_outputs/benchmark/collision/discovery/independent_validation/report.md)。
