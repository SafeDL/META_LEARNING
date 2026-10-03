# 被测驾驶算法

本目录保留当前 S01 所需控制器和此前复现的其他 ADS 实现。暂不参加实验的控制器及其权重也保留，供后续研究使用。

|位置|内容|当前使用|
|---|---|---|
|`highway_env/idm_profiles.py`、`registry.py`|IDM/FVDM 公式、历史配置及构建注册|S01 使用五个原 IDM 源和固定 FVDM 目标|
|`highway_env/idm_mobil.py`|IDM + MOBIL|暂不使用|
|`highway_env/value_iteration.py`|TTC 状态上的 Value Iteration|暂不使用|
|`highway_env/mcts_cv.py`|恒速预测的 MCTS|暂不使用|
|`highway_env/ppo_ece.py`、`checkpoints/ppo_ece/`|已有 PPO 实现及原 checkpoint|暂不使用|
|`metadrive/`|MetaDrive 原控制器、配置与策略|暂不使用|

Highway-env 的其他 ADS 可由 `registry.policy_factory()` 创建，原外部策略执行接口也已恢复。独立论文复现的冻结源码和协议仍在[历史归档](../archives/retired_research_chains/README.md)。

当前目标 `fvdm_safety_speed_23_mps` 仍直接读取[原目标协议](../benchmarks/s01/s01_fvdm/protocol.json)，不因恢复其他注册项而切换目标。源期望速度为 27 m/s、目标为 23 m/s，ego 初始速度均为 25 m/s。当前实验设置见[汇总](../results/srd_tnp_bqd/report.md)。
