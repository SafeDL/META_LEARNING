# 方法链

`method_chains/` 保存组合方法及其独立研究链。当前主线是 FBRT；其他方法链和
CoRe-Mine 历史实验各自保留，不与 FBRT 的数据或结果合并。正式结果统一位于
`results/method_chains/`。

| 目录 | 当前定位 |
| --- | --- |
| `failure_memory_regression/` | 当前 FBRT 回归测试主链；使用历史失败边界为目标版本挑选测试场景 |
| `detour_fusion/` | Risk Mining 与 DETOUR 的独立融合研究链 |
| `function_conditioned_routing/` | 功能条件化历史迁移研究及双基准验证 |
| `function_posterior_search/` | 经独立物理确认的自适应功能后验搜索 |
| `core_mine/` | CoRe-Mine 历史实验；FBRT 复用其中 IDM 参考配置与受控修改实现 |

FBRT 的职责、实验设置和运行命令见
[`failure_memory_regression/README.md`](failure_memory_regression/README.md)；当前代码修复方案与最新缓存回放分别见
[`docs/FBRT_CODE_FIXES_ZERO_SIM_CODEX_PLAN.md`](../docs/FBRT_CODE_FIXES_ZERO_SIM_CODEX_PLAN.md) 和
[`results/method_chains/failure_memory_regression/repair_20260926/repair_report.md`](../results/method_chains/failure_memory_regression/repair_20260926/repair_report.md)。
标准对齐结果和 `memory_v2` 数据库保留作基准及回放输入，不与最新修复报告混作同一结果版本。

依赖方向为：

```text
highway_env_benchmark/ + replications/  ->  method_chains/*
method_chains/core_mine/ IDM reference and local faults  ->  failure_memory_regression/
```

基础实现和独立复现不得反向导入方法链。FBRT 只复用 CoRe-Mine 的 IDM 参考配置
与受控修改实现，不读取其历史实验结果。各方法链自行拥有实验入口和专属基准；
功能后验搜索的 SUT 与场景因子设计集中在其 `benchmark.py`，避免复制控制器、
场景生成器或评价条件。

每个目录至多保留一个 `README.md`，在该文件中集中记录职责、入口和结果位置。已由完整对齐实验确认无效、且没有成为研究主张的中间方法不在本目录保留。
