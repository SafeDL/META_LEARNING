# FBRT 失效记忆回归测试

当前方法是 `FBRT-Memory-Exploit-v3`：从历史碰撞及附近通过记录建立失效模式，在每次目标查询后更新风险后验，按预测的失败概率选择下一场景。它与五种保留的对照方法共用候选库、目标结果和查询预算。方法名称中的版本标识属于已冻结实验协议；活动源码与配置文件使用职责名称。

`selector.py` 实现选择器和只在查询后揭示目标结果的 `TargetOracle`；`pattern_memory.py`、`bayes_model.py` 分别构造模式特征和概率模型。`catalogue.py` 编译场景，`experiment.py` 使用项目的 `highway_sim_env/` 与 `sut_algorithms/highway_env/` 测量物理结果，`replay.py` 读取已测银行并完成六方法离线比较，`benchmark.py` 进行留一物理种子簇的覆盖次数分析。`archive.py`、`schema.py` 和 `replay_utils.py` 提供共同的数据契约；`report.py` 生成物理实验报告，`audit.py` 检查初始状态场景。

本方法的四份配置集中在 [`configs/`](configs/)；当前交互场景使用 `interaction_catalogue.yaml`。`interaction.py` 是物理测量入口。`interaction_holdout_catalogue.yaml` 仅保留已冻结的 age080 场景定义；`prepare_holdout.py` 和 `validate_holdout.py` 对应其独立验证，结果中目标回归池为零。历史数据目录中的版本名与文件名是既有数据契约，清理源码时不改写其内容。

正式结果索引见 [`results/method_chains/failure_memory_regression/README.md`](../../results/method_chains/failure_memory_regression/README.md)：`repair_exploit_v3_fullbank/` 保存六方法完整冻结银行回放，`current_method_statistical_comparison.md` 记录按物理种子簇的比较，`memory_exploit_v3/` 保存覆盖次数留一分析，`interaction_holdout_age080/` 保存独立交互验证。已淘汰原型的结论在结果索引中概述，不进入活动代码。

在项目根目录运行离线回放；它不会新增物理回合：

```powershell
conda run -n metadrive python -m methods.failure_memory_regression.replay --offline-only --paired-repeats 10 --output results/method_chains/failure_memory_regression/memory_exploit
```
