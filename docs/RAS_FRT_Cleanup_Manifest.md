# RAS-FRT-UQ 代码与结果清理清单

当前论文实验只使用历史 IDM 库 A、FVDM 候选库 D，以及 D 上的九种选例方法。当前主方法 RAS-FRT-UQ 与九方法回放位于 `methods/ras_frt_uq/`，既有结果位于 `results/method_chains/ras_frt_uq/`。本清单记录已从 `methods/` 和 `results/method_chains/` 移至 [`archives/retired_research_chains/`](../archives/retired_research_chains/README.md) 的旧研究链；逐文件原路径及大小见 [`RAS_FRT_Cleanup_File_Manifest.tsv`](RAS_FRT_Cleanup_File_Manifest.tsv)。

| 原位置 | 文件数 | 归档原因 |
| --- | ---: | --- |
| `methods/detour_fusion/` | 13 | 独立旧方法链；不参与九方法比较 |
| `methods/function_conditioned_routing/` | 16 | 独立旧方法链；不参与九方法比较 |
| `methods/function_posterior_search/` | 7 | 独立旧方法链；不参与九方法比较 |
| `methods/core_mine/` 首轮归档文件 | 84 | 旧实验入口、分析与测试 |
| `methods/failure_memory_regression/` 首轮归档文件 | 66 | 旧 FBRT 实验入口、分析与配置 |
| `results/method_chains/core_mine/` | 694 | 旧方法结果 |
| `results/method_chains/detour_fusion/` | 18 | 旧方法结果 |
| `results/method_chains/failure_memory_regression/` | 5,282 | 旧方法结果 |
| `results/method_chains/function_conditioned_routing/` | 27 | 旧方法结果 |
| `results/method_chains/function_posterior_search/` | 15 | 旧方法结果 |

上表是首轮归档，合计 **6,222 个文件，702,369,559 字节**。其中旧结果目录有 6,035 个 Git 已跟踪文件。`core_mine/` 与 `failure_memory_regression/` 中的 14 个原有未提交修改文件也已按原路径结构归档，并另存[补丁备份](RAS_FRT_Legacy_Worktree_Changes.patch)。本轮再迁移原先留下的共享依赖：八个旧源码及包标识文件归档，活动实现以新职责命名；首轮逐文件清单保持原始记录。

## 必须保留的依赖

- `methods/ras_frt_uq/`：主方法 RAS-FRT-UQ、原版 RAS-FRT 及 Random、Farthest-First、HistoryRank-adapted、History-only、Residual risk-only、Transfer-UQ、Target-GP-UCB；早期 100 次实验的源码快照和结果保留在结果目录。
- `results/method_chains/ras_frt_uq/historical_idm/`：A 的场景、五组 IDM 物理响应、冻结模型与协议。
- `results/method_chains/ras_frt_uq/s01_uniform_fvdm_speed_23_mps/`：D 的场景、2,048 条 FVDM 响应、116 条碰撞清单、九方法 200 次回放及前期 100 次来源记录。
- `highway_sim_env/build_spec.py`、`highway_sim_env/s01_parameters.py`、`highway_sim_env/configs/scenario_parameter_space.yaml`：共享的构建标识、S01 坐标和有效标签规则。
- `sut_algorithms/highway_env/reference_profiles.py`、`sut_algorithms/highway_env/local_fault_idm.py`：注册表、适配器和仿真测试使用的控制器配置与车辆实现。原 `core_mine`、`failure_memory_regression` 中保留的 pilot 和旧契约源码也已归档，活动 `methods/` 不再依赖旧方法包。
- `results/method_chains/ras_frt_uq/{historical_idm,s01_uniform_fvdm_speed_23_mps}/source_snapshot/`：保存迁移前的源码字节，以核对 A、D 已冻结协议中的源码 SHA-256；冻结协议及结果未改写。

历史方案文档中的旧相对路径仍表示归档前的冻结位置；查阅时在该路径前加 `archives/retired_research_chains/`。注册表中的旧 PPO 构建指向归档检查点，该构建不属于当前 A→D 实验。

**状态：首轮清单文件及随后保留的旧源码均已归档；活动 `methods/` 仅保留 `ras_frt_uq/`，正式结果目录同步为 `results/method_chains/ras_frt_uq/`。**

## 共享仿真代码整理

| 原活动路径 | 整理后位置 |
| --- | --- |
| `highway_sim_env/envs/fbrt_unified_env.py` | `highway_sim_env/envs/unified_env.py`，类名为 `UnifiedHighwayEnv` |
| `highway_sim_env/envs/fbrt_metrics.py` | `highway_sim_env/envs/safety_metrics.py` |
| `highway_sim_env/envs/fbrt_scripted_vehicle.py` | `highway_sim_env/envs/scripted_vehicle.py` |
| `sut_algorithms/highway_env/fbrt_adapters.py` | `sut_algorithms/highway_env/policy_adapter.py` |
| `methods/ras_frt_uq/target_profile_confirmation.py` | `methods/ras_frt_uq/target_bank.py` |

旧 `fbrt_env.py`、`fbrt_scenarios.py`、`fbrt_training_env.py` 和 `fbrt_parameters.py` 已无活动引用，按原路径移入 `archives/retired_research_chains/highway_sim_env/`。D 结果子目录统一为 `s01_uniform_fvdm_speed_23_mps/`；冻结协议、模型权重、响应与源码快照保留原始内容。

S01 参数读取移除了旧字段别名、跨层查找和未使用的对齐掩码；重复查询由 `TargetOracle` 统一检查。FST 与 ScenarioFuzz 新运行记录文件路径，移除了仅用于来源记录的文件、源码与模型哈希；场景去重和确定性划分所需的哈希继续保留。

本轮验证：180 项活动测试通过；九方法、200 次预算的完整回放与既有协议、查询记录和汇总结果逐字节一致。另重跑 3 个 FVDM 场景及五组 IDM 各 2 个场景，物理响应与 A、D 记录一致；FST 的无文件哈希记录流程也在临时目录完成选例与评价。
