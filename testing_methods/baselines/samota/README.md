# SAMOTA：多目标代理搜索

本方法将已查询的碰撞、间距和 TTC 反馈作为多个优化目标，训练代理模型集成，交替执行全局与局部搜索，选择可能违反安全要求的场景。

原文：Fitash Ul Haq、Donghwan Shin、Lionel Briand，*Efficient Online Testing for DNN-Enabled Systems using Surrogate-Assisted and Many-Objective Optimization*，ICSE 2022，[DOI](https://doi.org/10.1145/3510003.3510188)。全文取自[用户目录](https://drive.google.com/file/d/1voE2CULyRJ9YKYvxzSvvsuTEtiBbFOmm/view)，并核查[作者官方复现包](https://doi.org/10.6084/m9.figshare.16468530)第4版、文件34050833中的源码与安全要求补充材料。

## 原文与实现对应

| 原文及作者文件 | [候选库实现](pool.py) | 保留内容 |
|---|---|---|
| §2.2.4；`lib/ensemble.py` | `Committee` | RBF、二阶多项式、Kriging三成员；80%/20%划分，按验证MAE的互补权重集成；最大成员差作为不确定性 |
| §5.2、算法2 | `PoolSearch` | 已付费数据库、未覆盖目标、全局/局部交替、真实反馈更新 |
| §5.2.1、算法3 | `PoolSearch.propose`全局分支 | 对每个未覆盖目标选择最佳预测及最不确定候选，多目标偏好优先 |
| §5.2.2、算法4；`utils.py:generate_clusters_using_database` | `local_clusters`及局部分支 | 每目标前20%，至少5条；HDBSCAN分区，每个区训练独立局部RBF并在有界区域优化 |
| §6.2.1；`lib/RBF.py` | `RadialNetwork/RadialSurrogate` | 10个高斯基，随机样本中心、初始β=3，中心、β及线性输出均训练；Adam、MAE、1000 epoch、批量8 |
| `lib/PR.py`、`lib/Kriging.py` | `Committee` | 标准化后的二阶普通最小二乘及可优化的常数×RBF GP；各成员输出裁剪至[0,1] |

当前候选库实现使用 PyTorch 训练高斯基的中心、宽度和线性输出；代理前向与训练机制分别核对作者代码。多目标偏好排序与代理、聚类和查询流程统一位于 [pool.py](pool.py)。

## 固定128次预算与目标重启

原文目标是覆盖不同安全要求，覆盖后从未覆盖集合移除。本项目使用三个响应目标：`1-collision`、截断至100米的间距/100、截断至10秒的TTC/10，阈值分别为0、0、0.2。这三个目标适配当前场景；原文 CARLA 任务使用六项安全要求。

固定预算实验中，全部目标覆盖后清空覆盖集合并重启目标，保留已查询数据库及全局、局部交替，继续至128次预算。重启时机、查询数、覆盖状态与重启次数均写入运行记录；这项规则用于将原文的覆盖目标流程适配到固定查询预算。

共同随机初始化8次计入128次总预算；不使用额外目标先验。选择器只取得公开候选及已付费Boolean、间距、TTC。候选池内全局代理搜索采用精确枚举取代MOSA的200次代理进化；局部搜索在历史局部包围盒内精确枚举，盒内无剩余点时投影到距离盒最近的可用候选区域。代理集成、多目标偏好、不确定性、全局/局部互补及聚类保持可核对。

作者原场景以16个混合属性表示，本项目以公开物理参数及类别表示。局部HDBSCAN距离使用原始物理范围归一化后的连续差值与类别不一致指标的平均值。scikit-learn的`min_samples`含自身点，因此以6对应作者hdbscan的5。所有点被判为噪声时，与作者代码相同，使用这一组前20%样本作为局部训练区。

作者Kriging源码在预测时算出标准化变量却将未标准化变量传给模型；本实现预测使用标准化变量，与训练坐标保持一致。成员误差同时为0时采用相同权重，避免原代码的0/0。

## 机制核查与入口

[机制测试](../../../tests/experiments/test_samota_pool.py)核对可训练高斯基、混合距离、不同局部区域分离，以及完整覆盖后的付费重启、全局/局部阶段和不重复查询。

```powershell
conda activate metadrive
python -B -m pytest tests/experiments/test_samota_pool.py -q -p no:cacheprovider
python -B -m experiments.domain_samota_validation
```

[核查入口](../../../experiments/domain_samota_validation.py)从官方固定文件读取作者PR和RBF前向函数，在相同输入下核对普通多项式预测及高斯基前向，再运行一个开发库的128次隔离回放。训练时间与候选选择时间分别记录。结果位于 `experiment_outputs/samota/collision/discovery/domain_comparison/stage_a/`。开发比较和独立验证均已完成。SAMOTA 在全来源条件下有200项独立运行，严格留家族条件复用同一批运行。比较范围与结果见 [独立验证报告](../../../experiment_outputs/benchmark/collision/discovery/independent_validation/report.md)。
