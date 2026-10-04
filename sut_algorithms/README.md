# 被测驾驶算法

研究对象是 highway-env / MetaDrive 中的驾驶规划与控制算法。当前目标为带安全停车间距和 0.15 s 连续感知延迟的 FVDM，参数在 [当前方法 config.py](../methods/history_guided_testing/config.py)，结果见 [实验报告](../results/history_guided_testing/README.md)。

|位置|内容|当前使用|
|---|---|---|
|highway_env/idm_profiles.py|IDM/FVDM 跟驰控制公式|六历史源和 FVDM 目标的基础控制器|
|highway_env/perception.py|连续感知延迟、基于速度的预测、安全停车间距 FVDM|当前历史源与目标|
|highway_env/policy_adapter.py|统一执行接口|当前仿真|
|highway_env/idm_mobil.py|IDM + MOBIL|可作为后续被测目标|
|highway_env/value_iteration.py|Value Iteration 规划|可作为后续被测目标|
|highway_env/mcts_cv.py|恒速预测 MCTS 规划|可作为后续被测目标|
|highway_env/ppo_ece.py、checkpoints/ppo_ece|PPO 策略和已有权重|可作为后续被测目标|
|metadrive|MetaDrive 控制器、配置与策略|保留研究平台|

历史源包括常规 IDM/FVDM、0.6 s 感知延迟 IDM/FVDM、制动受限 IDM 和预测制动 IDM。所有历史源期望速度 23 m/s、初速 25 m/s；预测读取当前位置与速度。
