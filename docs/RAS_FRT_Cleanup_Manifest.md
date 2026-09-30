# RAS-FRT 代码与结果清理清单

当前论文实验只使用历史 IDM 库 A、FVDM 候选库 D，以及 D 上的九种选例方法。九种方法的实现和逐次回放均位于 `methods/ras_frt/` 与 `results/method_chains/ras_frt/`。本清单记录已从 `methods/` 和 `results/method_chains/` 移至 [`archives/retired_research_chains/`](../archives/retired_research_chains/README.md) 的旧研究链；逐文件原路径及大小见 [`RAS_FRT_Cleanup_File_Manifest.tsv`](RAS_FRT_Cleanup_File_Manifest.tsv)。

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

- `methods/ras_frt/`：RAS-FRT、RAS-FRT-UQ 及 Random、Farthest-First、HistoryRank-adapted、History-only、Residual risk-only、Transfer-UQ、Target-GP-UCB；其中 100 次试验脚本与文件也是 200 次回放的冻结来源。
- `results/method_chains/ras_frt/historical_idm/`：A 的场景、五组 IDM 物理响应、冻结模型与协议。
- `results/method_chains/ras_frt/s01_uniform_fvdm_speed_23_mps_confirmation/`：D 的场景、2,048 条 FVDM 响应、116 条碰撞清单、九方法 200 次回放及前期 100 次来源记录。目录名是已冻结标识，论文称为 D。
- `highway_sim_env/build_spec.py`、`highway_sim_env/s01_parameters.py`、`highway_sim_env/configs/scenario_parameter_space.yaml`：共享的构建标识、S01 坐标和有效标签规则。
- `sut_algorithms/highway_env/reference_profiles.py`、`sut_algorithms/highway_env/local_fault_idm.py`：注册表、适配器和仿真测试使用的控制器配置与车辆实现。原 `core_mine`、`failure_memory_regression` 中保留的 pilot 和旧契约源码也已归档，活动 `methods/` 不再依赖旧方法包。
- `results/method_chains/ras_frt/{historical_idm,s01_uniform_fvdm_speed_23_mps_confirmation}/source_snapshot/`：保存迁移前的源码字节，以核对 A、D 已冻结协议中的源码 SHA-256；冻结协议及结果未改写。

历史方案文档中的旧相对路径仍表示归档前的冻结位置；查阅时在该路径前加 `archives/retired_research_chains/`。注册表中的旧 PPO 构建指向归档检查点，该构建不属于当前 A→D 实验。

**状态：首轮清单文件及随后保留的旧源码均已归档；活动 `methods/` 仅保留 `ras_frt/`。**
