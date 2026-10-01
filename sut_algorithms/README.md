# 被测驾驶算法

本目录统一保存实验中实际控制 ego 车辆的被测驾驶算法。仿真环境、测试方法和
正式结果分别保留在 `highway_sim_env/`、`metadrive_sim_env/`、
`replications/` 和 `results/`，不得在这些目录中复制控制器实现。

- `highway_env/`：Highway-env 的 IDM/FVDM profiles，以及筛选后保留的
  IDM+MOBIL、VI-TTC、MCTS-CV 和 PPO-ECE。
- `metadrive/`：MetaDrive 的黑盒 SUT 接口、IDM adapter 和 profile registry。

`highway_env/reference_profiles.py` 保存共享仿真器与测试使用的参考 IDM、FVDM 配置；`highway_env/policy_adapter.py` 将这些控制器接入统一执行器。`highway_env/local_fault_idm.py` 保存旧构建仍支持的局部故障车辆类。当前 RAS-FRT-UQ 实验使用完整 FVDM。

PPO-ECE 权重保存在 `highway_env/checkpoints/`，可随项目提交到 Git。
需要重新获取时，使用 `python -m replications.highway_sut_selection.cli fetch`
下载并校验原始模型文件。
