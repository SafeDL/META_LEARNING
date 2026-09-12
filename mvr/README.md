# Transferable Scenario Mining

当前活跃方法按仿真器隔离。MetaDrive 的 DIVA-Mine Cut-in 实现位于
`mvr/metadrive/`，其中包含场景、Frenet、物理约束、控制、failure analysis、
SUT 适配层及其脚本和测试；Highway-env 的快速机制验证实现位于
`mvr/highway/`。两个实现不共享运行时代码。

MetaDrive 工件位于 `results/metadrive/`；Highway-env MVP 的可复现实验工件
位于 `results/diva_highway/cutin_mvp/`，以 `_highway` 后缀明确区分仿真器。

配置与执行说明见 [`docs/DIVA_Mine_CutIn_Technical_Design.md`](../docs/DIVA_Mine_CutIn_Technical_Design.md)。

```powershell
conda run -n metadrive python -m pytest mvr/metadrive/tests mvr/highway/tests -q
conda run -n metadrive python -m compileall -q mvr/metadrive mvr/highway
conda run -n metadrive python -m mvr.highway.scripts.run_diva_highway_mvp --rebuild-bank
conda run -n metadrive python -m mvr.highway.scripts.render_diva_highway_mvp
```
