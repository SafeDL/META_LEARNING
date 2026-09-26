# 方法结果

本目录按方法保存正式实验结果。目录名沿用冻结实验清单中的历史路径；当前实现位于仓库根目录的 `methods/`。

| 目录 | 保留内容 |
| --- | --- |
| [`failure_memory_regression/`](failure_memory_regression/README.md) | FBRT 的冻结测量银行、完整六方法回放、留一簇分析和独立交互验证 |
| [`core_mine/`](core_mine/README.md) | CoRe-Mine 的原始数据划分及不同研究问题的正式结果 |
| `detour_fusion/` | Risk Mining 与 DETOUR 融合实验及可视化 |
| `function_conditioned_routing/` | 功能条件化路由的对齐与功能迁移基准 |
| `function_posterior_search/` | 功能后验搜索的独立确认结果 |

不同方法的实验目标和数据契约各自独立，不把结果混成同一排名。已被正式结果替代的试跑和无效中间版本移至 [`archives/experimental_results/`](../../archives/experimental_results/README.md)，不在本目录展示。
