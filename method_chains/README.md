# 本文方法链

`method_chains/` 只保存由基础方法或论文复现组合形成、且已完成有效验证的方法链。
源码目录仅包含实现、配置、测试和说明；正式结果统一位于
`results/method_chains/`。

当前保留三条方法链：

- `diva_detour_fusion/`：DIVA-Mine 与 DETOUR 历史层次的固定融合对照；
- `diva_function_conditioned_routing/`：功能条件化历史先验路由及双基准验证；
- `function_posterior_search/`：通过独立物理确认的自适应功能后验搜索。

依赖方向固定为：

```text
diva_highway_env/ + replications/  ->  method_chains/
```

基础实现和独立复现不得反向导入方法链。各方法链自行拥有实验入口和专属基准；
功能后验搜索的 SUT 与场景因子设计集中在其 `benchmark.py`，避免复制控制器、
场景生成器或评价条件。

已由完整对齐实验确认无竞争力、且没有成为论文主张的中间方法，不在本目录保留。
