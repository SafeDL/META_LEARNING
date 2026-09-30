# 退出活动目录的研究链

本目录按原相对路径保留本轮清理的旧方法实现与对应结果。原来的 `methods/...` 和 `results/method_chains/...` 路径分别位于本目录下同名子目录。逐文件清单见 [`RAS_FRT_Cleanup_File_Manifest.tsv`](../../docs/RAS_FRT_Cleanup_File_Manifest.tsv)。

| 旧研究链 | 实现 | 结果 |
| --- | --- | --- |
| CoRe-Mine | [方法说明](methods/core_mine/README.md) | [结果说明](results/method_chains/core_mine/README.md) |
| FBRT／FM²-FBT | [方法说明](methods/failure_memory_regression/README.md) | [结果说明](results/method_chains/failure_memory_regression/README.md) |
| DETOUR 融合 | [方法说明](methods/detour_fusion/README.md) | [结果目录](results/method_chains/detour_fusion/) |
| 功能条件路由 | [方法说明](methods/function_conditioned_routing/README.md) | [结果目录](results/method_chains/function_conditioned_routing/) |
| 功能后验搜索 | [方法说明](methods/function_posterior_search/README.md) | [结果目录](results/method_chains/function_posterior_search/) |

按清单迁入 6,222 个原始文件，合计 702,369,559 字节。14 个原有未提交代码改动随文件保留；另有[补丁备份](../../docs/RAS_FRT_Legacy_Worktree_Changes.patch)。旧冻结协议可能记录原位置的路径；如需重跑，应先把对应文件恢复到原路径并检查依赖。当前论文方法与结果分别位于 [`methods/ras_frt/`](../../methods/ras_frt/README.md) 和 [`results/method_chains/ras_frt/`](../../results/method_chains/ras_frt/README.md)。

本地归档保留全部文件；其中 11 个超过 10 MiB 的旧结果文件列在根目录 [`.gitignore`](../../.gitignore) 中，不随常规 Git 提交上传。其余归档文件可以提交。
