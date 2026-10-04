# 独立论文复现

本目录保留 AdaTE、DETOUR、FST、ScenarioFuzz 的独立复现代码、冻结仿真与控制器依赖，以及对应实验结果：

|内容|位置|
|---|---|
|复现代码|[replications](replications/README.md)|
|复现实验结果|[results/highway_replications](results/highway_replications/README.md)|
|冻结平台与控制器|`highway_sim_env/`、`sut_algorithms/`|

CoRe-Mine、FBRT／FM²-FBT、DETOUR 融合、功能条件路由、功能后验搜索五条退出主链的试验代码已清理；旧结果已压缩保存，恢复说明见 [归档索引](../README.md)。当前默认入口仅运行[历史引导风险测试](../../methods/history_guided_testing/README.md)，报告见[统一实验结果](../../results/history_guided_testing/README.md)。
