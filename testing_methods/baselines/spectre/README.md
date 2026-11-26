# SPECTRE 固定套件适配版

SPECTRE 用旧版 ADS 的测试响应选择和排序新版的复测场景。PPO 与修订 DQN 主比较采用论文第 2 节的碰撞、接近碰撞程度、驾驶需求和多样性属性，以及作者比较中表现最好的指标型进化算法（IBEA）。每次选择的单位为一个完整交通场景。[论文与公式](https://chengjie-lu.github.io/files/Lu2021_Chapter_Search-BasedSelectionAndPriori.pdf) · [作者目标函数](https://github.com/simplexity-lab/SPECTRE/blob/83613463676a0c3fa629cb1399541830499d48bf/ScenarioSearching/Problems/problem.py) · [作者搜索配置](https://github.com/simplexity-lab/SPECTRE/blob/83613463676a0c3fa629cb1399541830499d48bf/ScenarioSearching/Solutions/IBEA_Selection.py)

| 论文环节 | 固定套件的实现 |
|---|---|
| 碰撞 COL、接近碰撞程度 CPT（§2.1） | `attributes` 从批准开放的同谱系历史和共同前 20 次付费反馈中取原生碰撞 0/1、缺口 `clip(-RSS裕度,0,1)`、缺口落入 (0.3,0.9] 的 0/1 指示。每个候选按物理参数域归一化后的欧氏距离，取最近 5 条记录作距离倒数加权，边界并列全取，距离为零只平均精确匹配。得到预计碰撞概率、预计连续缺口和潜在碰撞区间的预计命中概率。区间指示先在每条已执行记录上计算。 |
| 驾驶需求 DEM（§2.1） | `demand_count` 按公开交通属性计分。跟车制动与受限切入各有四项，满足至少三项为高需求；切出露障碍与侧向侵入各有三项，满足至少两项为高需求。标定单位、方向、含等号阈值和每个场景的字段见[登记表](../../../docs/experiment_plan.md#指标与验收条件)。左右方向只用于近邻及距离。 |
| 多样性 DIV（式 3–5、9） | 用声明范围归一化各数值属性，以 0/1 表示方向是否相同。任意两例的距离为这些属性绝对差的均值。位置评分使用该例到套件内全部其他用例的平均距离。子集评分使用子集内所有无序用例对的平均距离；在固定长度下与距离总和同向，并使得分处于 [0,1]。 |
| 五个最大化目标（§2.2） | `fitness` 累加预计碰撞数、预计潜在碰撞数、高需求例数，另计算子集多样性和位置加权得分。位置效用对预计碰撞概率、预计连续缺口、原始需求计数、逐例多样性各赋 1/4 权重。按作者代码从 (L+1)/L 到 2/L 线性加权再除以块长 L，后两项目标保留三位小数。 |
| 搜索与决策（§3） | `survive`、`offspring`、`select_block` 使用加性 ε 指标和二元锦标赛。种子 11，每个搜索块种群与子代均为 100，总计 30,000 次目标计算，κ=1。整数 SBX 交叉概率 0.9，多项式变异逐位概率 1/L，分布指数均为 20。交叉转整数时截断，变异取最近整数（正中取偶）。[变异参考](https://github.com/jMetal/jMetalPy/blob/v1.5.5/jmetal/operator/mutation.py) |
| 固定套件排序 | `prioritize` 在共同前缀后一次排定其余 1,980 例，每块最多 200 例。变异和交叉在按位置效用排列的候选 ID 上进行，重复项无放回修复。块内按未舍入的位置效用降序，同分按套件序号。种群包含位置效用最大的可行子集。最终非支配解按预计碰撞数、位置得分、预计潜在碰撞数、高需求数、子集多样性依次最大化。末块全部入选时直接排序。 |

原研究从已有执行属性排序。这里的历史坐标与新版套件隔离，因此用近邻估计与执行结果有关的三项数值。驾驶需求与多样性可以在执行前直接计算。位置项保留原始需求计数，跟车与切入的范围为 0–4，切出与侧向侵入为 0–3，其余三项均为 0–1。因此，即使四项系数相同，驾驶需求的变化对位置得分也可能有更大影响。当前版本的全套件碰撞标签归评价端管理。本文分别报告种子 11 下四条 PPO 链和四条修订 DQN 链各 12 对结果。[实现](method.py)

运行 `python -B -m experiments.regression.run_fixed replay --policy ppo --method spectre --workers 4`。各场景–版本的 `queries/<版本>/spectre_traffic_attributes/` 保存属性设置、准入历史指纹、完整后缀排序指纹和付费账本。`python -B -m experiments.regression.run_fixed report --policy ppo` 从准入历史与共同前缀重建完整排序，逐笔核对扣费，再按封存真值复算指标。[PPO 结果](../../../results/policy_regression/fixed_suite/README.md)

修订 DQN 使用相同入口，将 `--policy ppo` 替换为 `--policy dqn`。2026-11-22 授权追加的 12 条回放使用相同交通属性设置与停止规则，全部排序及账本独立核对后纳入五方法矩阵。[修订 DQN 结果](../../../results/dqn_continual_learning/policy_regression/README.md)
