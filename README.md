# SRD-TNP-BQD：S01 事故场景测试

面向黑盒驾驶控制器的测试研究。当前唯一论文方法为 **SRD-TNP-BQD**；实验恢复为原 RAS-FRT-UQ 的单一 S01 cut-in 基准：A/D 各 2048 个场景、五个历史 IDM、固定 FVDM 目标、每条轨迹 200 次查询。

默认方法已统一为匹配历史源、独立均值读出、联合均值校准、冻结 h／差异核和原混合采集。原 A/D、目标、ego 初始条件、五源及冻结模型不变；均值使用原 A 上补充的期望速度23 m/s 的 IDM 源。正式比较包含本文和九个基线：Uniform random、Farthest-first、kNN historical-risk ranking、GP-UCB、GP-EI、BOP-Elites、BAS、RF surrogate BO、RAS-FRT-UQ，另列同源 kNN 数据对照。RAS 保留原二值反馈，其余方法使用连续风险。

|目录|用途|
|---|---|
|[methods/srd_tnp_bqd](methods/srd_tnp_bqd/README.md)|本文、通用和文献基线的统一实验入口|
|[methods/ras_frt_uq](methods/ras_frt_uq/README.md)|原四维、二值 RAS-FRT-UQ 对照|
|[benchmarks/s01](benchmarks/s01/README.md)|共享原 A/D、五源、目标响应与实验协议|
|[当前比较](results/srd_tnp_bqd/comparison/README.md)|默认新主线、九个基线、同源 kNN 与完整真值|
|[highway_sim_env](highway_sim_env/README.md)|当前 S01 仿真器及保留的外部 ADS 执行接口|
|[metadrive_sim_env](metadrive_sim_env/README.md)|保留的 MetaDrive 仿真平台，当前实验暂不使用|
|[sut_algorithms](sut_algorithms/README.md)|共享被测控制器|
|[results](results/README.md)|正式结果索引|
|[tests](tests/srd_tnp_bqd)|当前 S01 方法、基线和信息边界的回归测试|
|[archives](archives/README.md)|退出主链的历史研究与独立论文复现|
|docs|[代码规范](docs/style.md)；当前方法、基线出处与执行协议合并在方法 README|

使用现有 `metadrive` 环境。唯一执行主链为 `audit → train → experiment → evaluate`，命令见[方法说明](methods/srd_tnp_bqd/README.md)。原 D 含 116 次碰撞和 33 个碰撞单元；曾参与 RAS 开发，成绩属于同库开发证据。

异质多场景、弱 FVDM、新增 IDM 目标及其试验结果已退出当前方法链。后续实验设置只允许在单一 S01 内讨论 ego 初始条件、场景参数空间、源 SUT 类型和参数；固定目标保持不变。

当前实验只运行 S01 主链。`metadrive_sim_env/`、`sut_algorithms/metadrive/` 和此前复现的 Highway-env ADS 控制器及 PPO 权重完整保留，暂不用于当前比较。AdaTE、DETOUR、FST、ScenarioFuzz 的独立复现与历史结果保留在归档。默认 pytest 运行 `tests/`，保留的 MetaDrive 测试可单独执行。

第一批完整轨迹、旧消融与[事故 GIF](results/srd_tnp_bqd/reference/first_batch/visualizations/README.md)作为[历史参照](results/srd_tnp_bqd/reference/first_batch/README.md)保留。当前导出器读取默认新主线，只导出 GIF。已有冻结数据中的场景编号和执行契约保留原标识，用于追溯原实验。
