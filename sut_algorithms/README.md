# Systems under test

本目录统一保存实验中实际控制 ego 车辆的被测驾驶算法。仿真环境、测试方法和
正式结果分别保留在 `highway_env_benchmark/`、`metadrive_benchmark/`、
`replications/` 和 `results/`，不得在这些目录中复制控制器实现。

- `highway_env/`：Highway-env 的 IDM/FVDM profiles，以及筛选后保留的
  IDM+MOBIL、VI-TTC、MCTS-CV 和 PPO-ECE。
- `metadrive/`：MetaDrive 的黑盒 SUT 接口、IDM adapter 和 profile registry。

外部策略权重不进入源码目录。需要复跑 PPO-ECE 时，使用
`python -m replications.highway_sut_selection.cli fetch` 下载并校验临时缓存。

