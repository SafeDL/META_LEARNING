# SAMOTA：多目标代理搜索

SAMOTA 根据已查询的碰撞、间距和 TTC 训练代理集成，交替进行全局与局部搜索，寻找违反安全要求的候选。

依据 Haq、Shin、Briand 的 *Efficient Online Testing for DNN-Enabled Systems using Surrogate-Assisted and Many-Objective Optimization*，ICSE 2022，[DOI](https://doi.org/10.1145/3510003.3510188)，[核对全文](https://drive.google.com/file/d/1voE2CULyRJ9YKYvxzSvvsuTEtiBbFOmm/view)。作者实现核对使用[官方复现包](https://doi.org/10.6084/m9.figshare.16468530)第 4 版。

| 原文与作者实现 | [pool.py](pool.py) | 作用 |
|---|---|---|
| §2.2.4，`ensemble.py` | `Committee` | RBF、二阶多项式、Kriging 三成员；按验证 MAE 加权，成员最大差表示不确定性 |
| §5.2，算法2 | `PoolSearch` | 维护已查询数据库、未覆盖目标和搜索阶段 |
| §5.2.1，算法3 | `PoolSearch.propose` 全局分支 | 按目标顺序提出最佳预测与最不确定候选 |
| §5.2.2，算法4 | `local_clusters` 与局部分支 | 各目标取前 20% 响应，至少 5 条，聚类后分别训练局部 RBF |
| §6.2.1，`RBF.py` | `RadialNetwork/RadialSurrogate` | 训练高斯基中心、宽度和线性输出 |

## 候选库适配

三个响应目标为 `1-collision`、截断至 100 米的间距除以 100、截断至 10 秒的 TTC 除以 10，覆盖阈值为 0、0、0.2。全部目标覆盖后清空覆盖集合，保留数据库并继续搜索至预算结束；重启与覆盖状态逐次记录。

全局搜索枚举剩余候选，局部搜索枚举聚类包围盒内的候选；盒内无剩余点时，选择距离该区域最近的可用候选。提案按多目标偏好排序、合并并去重。归一化连续距离与类别差异用于聚类，也可通过 `domain` 指定参数定义。

RBF 使用 10 个高斯基，初始宽度参数为 3，以 Adam、MAE、批量 8 训练 1,000 轮。集成采用 80%／20% 训练验证划分，输出限制在 [0,1]。scikit-learn 的 HDBSCAN 取 `min_samples=6`，对应作者实现中不计自身的 5；全为噪声时使用前 20% 样本作为局部区域。

Kriging 的预测坐标与训练使用相同标准化；成员验证误差均为零时等权集成。这两项明确修正作者代码中的坐标不一致和零除问题。

## 接入与验证

入口为 `run(candidates, oracle, initial_indices=..., budget=200, seed=11)`，由 [experiments.pilot](../../../experiments/pilot.py)接入。

~~~powershell
python -B -m pytest tests/experiments/test_samota_pool.py -q
~~~

测试覆盖代理训练、目标顺序、并列去重、局部区域和预算重启。[维护回放](../../../artifacts/runs/joint_response/reports/samota_cleanup_replay_check.json)另核对原任务的 200 次查询顺序和反馈。结果见[论文报告](../../../artifacts/runs/joint_response/paper/reports/results.md)。
