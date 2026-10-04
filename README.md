# 驾驶规划与控制算法的场景测试

研究在 highway-env / MetaDrive 中如何用有限测试预算发现驾驶规划与控制算法的失效。当前主链为**历史引导 GP 风险测试**，实测使用 highway-env；`metadrive` 是运行用的 Conda 环境。

历史场景库与独立目标库均包含切入 1024 个、前车急刹 1024 个。六个历史 SUT 分为反应、感知延迟、制动受限和预测制动四组；历史坐标只分训练/验证约 80%/20%。目标为带安全停车间距和 0.15 s 感知延迟的 FVDM，期望速度 23 m/s、最大制动 8 m/s²。每方法五个种子、每次 200 次查询。

```powershell
conda activate metadrive
python -B -m methods.history_guided_testing.run
python -B -m pytest -q -p no:cacheprovider
```

|目录|用途|
|---|---|
|[methods/history_guided_testing](methods/history_guided_testing/README.md)|唯一默认主链、九个基线（含 RAS-FRT-UQ）与组件消融|
|[当前实验结果](results/history_guided_testing/README.md)|设置、五种子对比、反馈精度、图与清理记录|
|[highway_sim_env](highway_sim_env/README.md)|当前仿真平台与其他驾驶算法执行接口|
|[metadrive_sim_env](metadrive_sim_env/README.md)|可用于后续研究的 MetaDrive 平台|
|[sut_algorithms](sut_algorithms/README.md)|IDM、FVDM、MOBIL、VI、MCTS、PPO 等被测算法|
|[tests/history_guided_testing](tests/history_guided_testing)|当前主链的数值、控制器与基线检查|
|[methods/ras_frt_uq](methods/ras_frt_uq/README.md)|统一实验中的 RAS-FRT-UQ 基线|
|[archives](archives/README.md)|独立论文复现与旧实验恢复归档|
|[docs/style.md](docs/style.md)|当前代码规范|

已移除无额外收益的神经历史特征 h、旧全局线性校准、均值读出及默认混合 QD 采集。方法采用实测历史先验和风险空间 GP，默认核由历史验证选择，采集固定为纯风险优先。最后一次方法简化后重新实测独立目标库，最终目标测试不参与选型。

旧主链失败诊断保存在结果目录的 `development_summary.json`。废弃试验分支、过期控制器注册、重复的原 S01/RAS 实验及缓存已退出活动目录；旧结果完整压缩保存于 `archives/retired_experiments.zip`，经逐文件解压比对验证，可恢复原路径。清理记录见 `results/history_guided_testing/cleanup.json`，文件清单与备份核验见 `cleanup_review.json`。MetaDrive 平台、规划／控制算法和独立论文复现保留。
