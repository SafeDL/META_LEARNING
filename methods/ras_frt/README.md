# RAS-FRT：使用 A 的 IDM 历史测试 D 中的 FVDM

历史库 A 包含 2,048 个 S01 切入场景，每个场景由同一个 IDM 模型的五组参数分别执行。候选库 D 包含另外 2,048 个场景，以及固定 FVDM 控制器在这些场景上的完整物理响应。D 使用 `FBRTUnifiedEnv`、`highway-env` 1.9.1 和 20 Hz 仿真频率。实验只使用 A、D 两库。

## 九方法比较

```powershell
conda run -n metadrive python -m methods.ras_frt.d_budget_200_experiment
```

入口读取已经冻结的 A、D 文件，在 D 上比较 `Random`、`Farthest-First`、`HistoryRank-adapted`、`History-only`、`Residual risk-only`、`RAS-FRT`、`Transfer-UQ`、`Target-GP-UCB` 和 `RAS-FRT-UQ`。每种方法最多看见 200 个 FVDM 标签，在 10、30、50、100、150、200 次查询处统计发现数与覆盖率。HistoryRank 运行一次，其余方法各用五个固定随机种子。完整 D 标签仅在回放完成后用于评价。

这是对已完整执行的 D 库进行离线查询回放；200 次是选择器可见的标签预算，不是获得全库真值所需的物理执行次数。详细结果与局限见 [D 库 README](../../results/method_chains/ras_frt/s01_uniform_fvdm_speed_23_mps_confirmation/README.md)。融合方法的设置曾参考 D 的结果，因此其 D 上优势属于开发阶段证据。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `protocol.py`、`banks.py` | 读取并校验已经冻结的 A 库及五组 IDM 响应 |
| `training.py`、`response_encoder.py` | 在 A 上训练历史响应模型；D 的回放读取已冻结权重 |
| `target_profile_confirmation.py` | 生成 D 候选场景并执行固定 FVDM；当前回放直接读取已有结果 |
| `coverage_selector.py`、`transfer_uncertainty.py`、`fusion_selector.py`、`comparison_selectors.py` | RAS-FRT、融合方法及所需对比选择器 |
| `d_budget_200_experiment.py` | 当前九方法、200 次预算的回放与汇总入口 |
| `d_experiment.py`、`fusion_experiment.py` | 早期 100 次比较及融合开发记录；200 次入口复用前者，后者保留用于核对融合设计来源 |

共享构建指纹与 S01 参数规则分别位于 `highway_sim_env/build_spec.py` 和 `highway_sim_env/s01_parameters.py`。A、D 冻结协议中的旧源码哈希通过各结果目录的 `source_snapshot/` 核对；活动代码已使用新模块名称。

历史库 A 见 [A 库 README](../../results/method_chains/ras_frt/historical_idm/README.md)。完整的九方法轨迹、协议及汇总指标见 [D 库 README](../../results/method_chains/ras_frt/s01_uniform_fvdm_speed_23_mps_confirmation/README.md)。
