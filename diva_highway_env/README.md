# DIVA highway-env implementation

本目录维护本文方法在 highway-env 上的轻量实现，并提供当前复现实验共用的
Cut-in 场景、SUT 和响应库。AdaTE、DETOUR 和融合链可以读取这些稳定接口，但
不得把各自的算法、配置或结果写回本目录。

主要目录：

- `envs/`：可直接执行的 highway-env 场景；
- `sut/`：IDM/FVDM 被测控制器；
- `data/`：候选场景和响应库；
- `diva/`：低秩先验、诊断采样和后验更新；
- `tests/`：共享仿真契约。

```powershell
conda run -n metadrive python -m pytest diva_highway_env/tests -q -p no:cacheprovider
conda run -n metadrive python -m diva_highway_env.data.generate_anchor_bank
conda run -n metadrive python -m diva_highway_env.data.response_bank
```
