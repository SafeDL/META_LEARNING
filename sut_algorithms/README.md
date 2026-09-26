# 被测驾驶算法

本目录统一保存实验中实际控制 ego 车辆的被测驾驶算法。仿真环境、测试方法和
正式结果分别保留在 `highway_sim_env/`、`metadrive_sim_env/`、
`replications/` 和 `results/`，不得在这些目录中复制控制器实现。

- `highway_env/`：Highway-env 的 IDM/FVDM profiles，以及筛选后保留的
  IDM+MOBIL、VI-TTC、MCTS-CV 和 PPO-ECE。
- `metadrive/`：MetaDrive 的黑盒 SUT 接口、IDM adapter 和 profile registry。

PPO-ECE 权重保存在 `highway_env/checkpoints/`，可随项目提交到 Git。
需要重新获取时，使用 `python -m replications.highway_sut_selection.cli fetch`
下载并校验原始模型文件。
