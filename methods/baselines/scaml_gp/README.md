# ScaML-GP 碰撞分类改编

本目录按[ScaML-GP原论文](https://proceedings.mlr.press/v238/tighineanu24a.html)的多来源先验结构实现固定候选库碰撞分类。来源系数通过已查询目标反馈学习，所有位置共用同一组系数。这里使用Bernoulli似然、共享FSBO表示及碰撞概率贪心，名称为`scaml_classifier`，不表述为原连续响应贝叶斯优化实验的原样复现。

## 先验与目标适配

每个历史来源保留固定的碰撞潜在后验`μ_s、C_s`。先验定义为

\[
m(x)=m_t+\sum_s a_s[\mu_s(x)-m_H],\qquad
K(x,x')=K_t(x,x';\ell,v)+\sum_s a_s^2C_s(x,x').
\]

`m_H`来自历史来源教师的分类均值，调用时显式提供；`m_t`来自目标FSBO适配。减去共同历史均值，再加入目标截距，是本项目在碰撞潜在尺度上的中心化设置。整个先验是一个GP，来源系数不归一为概率权重，协方差必须使用系数平方。原论文的多来源公式及来源独立训练见[定理1与算法1](https://proceedings.mlr.press/v238/tighineanu24a/tighineanu24a.pdf)。

目标表示每次从同一FSBO基础模型重置，只用已付费碰撞适配10步Adam、学习率0.003。随后固定表示和目标均值，以目标Bernoulli/Laplace边际证据学习残差核的长度尺度、信号方差及来源系数。该分步适配是本项目的信息匹配设置；新网络的宽度、历史监督量和目标标签权限均不增加。

来源系数限制为正，采用独立Gamma(1,1)先验，对应目标函数中的`Σ a_s`。长度尺度和信号方差各限制在`[1e-4, 1e2]`，来源系数数值下界为`1e-8`。分类残差核由边际似然优化，不继承连续响应尺度的核先验或高斯观测噪声先验。无目标反馈时来源系数均为`1/H`；每次标量优化从该系数和当前FSBO核参数重新开始，不使用前次系数作热启动。

标量优化器为L-BFGS-B，最多100步、500次函数评估，`ftol=1e-9、gtol=1e-5、maxls=30`。异常终止、非有限解或目标值升高会直接停止；达到迭代/评估上限的有限解单列为未收敛，不将停止标志解释成全局最优。

## 分类推断与数值检查

历史准备中发现，直接使用旧分类器固定10次的无阻尼Newton模式更新，会在该新优化范围的部分参数邻域中振荡。失败输入和扰动结果保存在`results/baselines/scaml_gp/collision/preparation/undamped_failure*`，没有产生开发查询。

新基线改用`methods/common/gp_laplace.py`：在白化潜在坐标中做阻尼Newton更新，以目标下降决定步长，最多100次，模式梯度上限为`1e-10`。边际证据和预测使用同一个收敛模式。它保持已冻结的`gp_classifier.py`、`gp_joint.py`和既有实验原样；不能把数值推断变化带来的差异单独解释为来源迁移收益。

13项专门检查及81项相关合并回归已通过，覆盖独立模式优化的边际证据、有限差分梯度、来源协方差的平方系数、零来源系数退化、正系数优化、无反馈初始化、预算、仅付费标签、来源排列、辅助安全量、隔离查询、CPU/CUDA及大方差对立先验的振荡回归。32组历史完整库概率、目标值和梯度现已全部核对通过：最大概率差4.97e-12、目标值差8.34e-11、梯度差4.09e-8。31次标量优化达到投影梯度条件，1次按相对目标变化停止；最大投影梯度8.60e-5。全部内层模式满足1e-10阈值，不声称标量超参数取得全局最优。

```powershell
python -B -m pytest tests/baselines/test_scaml_search.py -q
python -B -m benchmark.discovery_scaml_audit
python -B -m benchmark.discovery_scaml_audit correction
```

实际选择入口为`methods.baselines.scaml_gp.search.select`。参数仅为公开候选、隔离查询接口、基础模型、允许的历史来源后验及参考均值、共同初始化、预算和设备。返回每次查询前的系数、核参数、目标均值、概率、标量优化停止原因和分项时间。

数值修正另用固定失败参数及一维解析模式方程核对，见[图和对应数据](../../../results/baselines/scaml_gp/collision/preparation/numerical_correction.json)。PNG/SVG/PDF已生成，PNG已目视检查；该证据不涉及新的开发场景或查询。

另提供`physical_scaml_classifier`，在GP预测概率外按0.90/0.10混合独立物理GP概率；物理GP只用已付费碰撞，重置后适配10步Adam、学习率0.01。这是本项目对固定物理分量的匹配控制，与另外两个同推断目标控制共同进入独立160项矩阵，见[ScaML开发协议](../../../docs/collision_discovery_scaml.md)。

后续加入`scaml_target`及`physical_scaml_target`，保持相同FSBO适配、收敛模式及残差核优化，完全省去显式来源后验和来源系数项。它们调用时只能传入`sources=None、source_reference=None`，不增加训练标签。新增32组历史完整库、证据与梯度核对通过。四模式现有17项专门检查、98项相关合并回归通过；原13/81项与32项来源数值检查仍是前序记录。

四模式各40项、共160项现已完成；20,480条反馈/逐步摘要、480次独立全库选点及10项损坏控制通过，首轮计入160项。两种来源版本平均召回90.5785%和94.4997%，对应无来源控制为88.3990%和93.7603%，均未超过既有最强参考98.5857%。全部34种方法的汇总、曲线和预算点另经独立重算，图表已目视检查。该比较不更换既定主方法或启动C/D/E，详见[完整报告](../../../results/benchmark/collision/discovery/scaml/report.md)。
