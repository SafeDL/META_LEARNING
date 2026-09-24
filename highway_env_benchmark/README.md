# Mining highway-env implementation

本目录维护本文方法在 highway-env 上的轻量实现，并提供当前复现实验共用的
Cut-in 场景和响应库。SUT 统一位于根目录 `sut_algorithms/highway_env/`。
AdaTE、DETOUR 和融合链可以读取这些稳定接口，但
不得把各自的算法、配置或结果写回本目录。

主要目录：

- `envs/`：可直接执行的 highway-env 场景；
- `data/`：候选场景和响应库；
- `mining/`：低秩先验、诊断采样和后验更新；
- `tests/`：共享仿真契约。

```powershell
conda run -n metadrive python -m pytest highway_env_benchmark/tests -q -p no:cacheprovider
conda run -n metadrive python -m highway_env_benchmark.data.generate_anchor_bank
conda run -n metadrive python -m highway_env_benchmark.data.response_bank
```
