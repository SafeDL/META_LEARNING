# Transferable Scenario Mining

当前活跃方法是 DIVA-Mine Cut-in。实现位于 `mvr/diva/`；共享的场景、Frenet、物理约束、控制、failure analysis 和 SUT 适配层位于其相邻模块。

配置与执行说明见 [`docs/DIVA_Mine_CutIn_Technical_Design.md`](../docs/DIVA_Mine_CutIn_Technical_Design.md)。

```powershell
conda run -n metadrive python -m pytest mvr/tests -q
conda run -n metadrive python -m compileall -q mvr
```
