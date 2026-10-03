# 退出活动目录的研究链

本目录按原相对路径保留此前退出活动目录的独立旧研究链。原来的 `methods/...` 和 `results/method_chains/...` 路径分别位于本目录下同名子目录；下表为本地仍保留的内容。

| 旧研究链 | 实现 | 结果 |
| --- | --- | --- |
| CoRe-Mine | [方法说明](methods/core_mine/README.md) | [结果说明](results/method_chains/core_mine/README.md) |
| FBRT／FM²-FBT | [方法说明](methods/failure_memory_regression/README.md) | [结果说明](results/method_chains/failure_memory_regression/README.md) |
| DETOUR 融合 | [方法说明](methods/detour_fusion/README.md) | [结果目录](results/method_chains/detour_fusion) |
| 功能条件路由 | [方法说明](methods/function_conditioned_routing/README.md) | [结果目录](results/method_chains/function_conditioned_routing) |
| 功能后验搜索 | [方法说明](methods/function_posterior_search/README.md) | [结果目录](results/method_chains/function_posterior_search) |
| AdaTE、DETOUR、FST、ScenarioFuzz 独立复现 | [复现说明](replications/README.md) | [结果说明](results/highway_replications/README.md) |

首轮迁入记录为 6,222 个原始文件、702,369,559 字节，14 个原有未提交代码改动随文件保留。旧冻结协议可能记录原位置的路径；如需重跑，应先把对应文件恢复到原路径并检查依赖。当前论文方法与结果分别位于[SRD-TNP-BQD实现](../../methods/srd_tnp_bqd/README.md)和[当前结果](../../results/srd_tnp_bqd/README.md)。原始 A/D 已移至[共享基准库](../../benchmarks/s01/README.md)。

共享仿真的历史依赖统一保存在 [`highway_sim_env/`](highway_sim_env) 和 [`sut_algorithms/`](sut_algorithms)；包括旧场景／低秩挖掘接口、外部策略、历史控制器构建，以及此次收敛前的原物理执行器与配置。独立复现的 3,634 个结果文件完整迁入，未删除其模型或有效结果。原环境配置见 [environment.yml](environment.yml)。当前实验只运行 S01；根目录同时恢复保留其他 ADS 与通用执行接口，见 [仿真说明](../../highway_sim_env/README.md)。

复核独立复现时应使用本归档目录内的冻结代码和协议。旧记录可能包含原工作目录路径，须先检查并恢复相应实验上下文；活动目录不提供旧接口兼容层。当前默认测试不发现本目录中的历史测试。

本地归档保留全部文件；其中 11 个超过 10 MiB 的旧结果文件列在根目录 [`.gitignore`](../../.gitignore) 中，不随常规 Git 提交上传。其余归档文件可以提交。
