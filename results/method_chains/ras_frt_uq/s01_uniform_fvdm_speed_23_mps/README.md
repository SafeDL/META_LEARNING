# 候选库 D：固定 FVDM 的预算化碰撞发现

## 场景与判定

A 中已有 2,048 个 S01 切入场景、五组同模型不同参数 IDM 的物理响应及冻结的历史响应模型。D 是另外 2,048 个 S01 场景，A 与 D 没有完全相同的参数坐标。

D 的被测对象是完整 FVDM 控制器 `fvdm_safety_speed_23_mps`，目标速度 23 m/s、最大制动 8 m/s²、期望间距 8 m、敏感度 0.6、速度增益 1.0、过渡间距 8 m。该配置在抽取 D 前根据控制器校准确定。D 的 2,048 次物理执行全部有效，其中 **116 次自车碰撞（5.66%）**。

本实验将 `ego_collision=True` 判为碰撞，完成且未碰撞判为通过。`min_clearance` 仅供诊断，不设 surrogate safety measure 阈值；下文的“碰撞场景”均指真实自车碰撞。覆盖率统计已发现碰撞在四维场景参数网格中占据的单元，不代表独立事故机理。

## 九方法比较：每种方法最多查询 200 次

表中为找到的真实碰撞场景数。HistoryRank-adapted 为单次确定性运行，其余方法使用五个固定随机种子，表中报告均值；全库分母为 116。

| Method | @10 | @30 | @50 | @100 | @150 | @200 |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.2 | 1.2 | 2.4 | 4.6 | 7.8 | 10.8 |
| Farthest-First | 2.0 | 4.2 | 5.8 | 9.6 | 12.4 | 16.0 |
| HistoryRank-adapted | 2 | 3 | 4 | 32 | 72 | 77 |
| History-only | 6.2 | 17.0 | 28.2 | 54.0 | 79.2 | 92.6 |
| Residual risk-only | 6.8 | 20.0 | 33.8 | 65.6 | 83.8 | 95.6 |
| RAS-FRT | 4.8 | 16.6 | 30.4 | 62.2 | 83.0 | 93.2 |
| Transfer-UQ | 0.0 | 9.6 | 28.6 | 70.0 | 103.0 | 114.2 |
| Target-GP-UCB | 0.6 | 15.6 | 31.8 | 71.8 | 106.8 | **115.4** |
| RAS-FRT-UQ | 6.0 | **20.0** | **37.0** | **79.8** | **109.6** | 114.2 |

查询到 100 次时，RAS-FRT-UQ 的碰撞命中率为 **79.8%**，全库碰撞召回率为 **68.8%**，四档网格碰撞覆盖率为 **31.8/33＝96.4%**。查询到 200 次时，Target-GP-UCB 平均找到 115.4/116 个碰撞，RAS-FRT-UQ 与 Transfer-UQ 各找到 114.2/116 个；三者平均均覆盖全部 33 个碰撞网格单元。五个种子共用同一候选库，不能当成五个独立数据集。

从仓库根目录运行 `conda run -n metadrive python -m methods.ras_frt_uq.experiment` 可重放已有 A、D 上的九方法比较。此命令不重新执行 FVDM；200 次是每种选择器可见的标签预算。主要文件为 [`budget_200_protocol.json`](budget_200_protocol.json)、[`budget_200_method_replay.json`](budget_200_method_replay.json) 和 [`budget_200_method_evaluation.json`](budget_200_method_evaluation.json)。

## 早期 100 次实验的来源记录

最初的冻结比较包含 Random、HistoryRank-adapted、RAS-FRT、Transfer-UQ；随后在同一 A、D 上探索 RAS-FRT-UQ。两轮 100 次回放的协议、逐次轨迹和预测分数仍保留，供核对融合方法来源。上表的 100 次列已用九方法统一口径给出当前比较，不重复展示旧表。

## 数据可见性与局限

D 的 2,048 个 FVDM 标签先由物理仿真完整测得。离线回放时，每个选择器只接收自己查询到的标签；完整标签只用于最后的真值评价。该设置比较的是**固定库上的选例策略**，不能把 200 次预算理解为只需 200 次物理执行就能获得完整 D 真值。

RAS-FRT 的初始配置在 A 内部选择；RAS-FRT-UQ 的机制与权重曾在查看 D 的结果后确定。因此当前 D 上的融合结果属于开发比较，尚不是独立盲测确认，也不能直接推广到其他候选库。完整 D 上的 AP 包含已查询场景，只作诊断；主要结果是实际找到的碰撞数、查询碰撞率、全库召回率与四档网格覆盖率。

## 文件索引

- [`candidate_manifest.jsonl`](candidate_manifest.jsonl)、[`scenario_template.json`](scenario_template.json)：D 的 2,048 个场景及生成模板。
- [`fvdm_safety_speed_23_mps.jsonl`](fvdm_safety_speed_23_mps.jsonl)：全部 FVDM 物理响应。
- [`all_fvdm_dangerous_scenarios.jsonl`](all_fvdm_dangerous_scenarios.jsonl)：116 个真实自车碰撞场景；文件名沿用冻结标识。
- [`budget_200_protocol.json`](budget_200_protocol.json)、[`budget_200_method_replay.json`](budget_200_method_replay.json)、[`budget_200_method_evaluation.json`](budget_200_method_evaluation.json)：当前九方法比较的协议、逐次查询与汇总。
- `protocol.json`、`method_protocol.json`、`method_replay.json`、`method_evaluation.json`、`method_predictions/`：D 生成协议和最初的 100 次比较。
- `fusion_protocol.json`、`fusion_replay.json`、`fusion_evaluation.json`、`fusion_predictions/`：融合方法在同一 D 上的早期开发记录。
- `source_snapshot/`：早期 100 次实验的 `d_experiment.py` 与 `fusion_experiment.py` 源码字节；当前九方法回放执行 `methods/ras_frt_uq/` 中的代码。既有协议中的源码哈希保留为历史记录，活动回放不再逐次校验。
