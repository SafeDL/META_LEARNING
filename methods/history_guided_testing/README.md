# 历史引导风险测试

用已有驾驶算法的实测风险定位危险区域，再用新算法的少量反馈修正历史经验，选择下一次测试场景。研究对象是 MetaDrive 或 highway-env 中的驾驶规划与控制算法；当前实验在 highway-env 1.9.1 中完成。

当前主链在独立目标库上用 200 次查询找全 84 个碰撞，五个种子均覆盖全部 46 个碰撞参数单元。反馈校正使总体风险预测 RMSE 相对历史先验降低 31.76%。完整比较和组件证据见 [实验报告](../../results/history_guided_testing/README.md)。

## 运行

```powershell
conda activate metadrive
python -B -m methods.history_guided_testing.run
python -B -m pytest -q -p no:cacheprovider
```

固定实验参数写在 [config.py](config.py)，方法、基线和组件实验共用一个入口。结果集中在 [results/history_guided_testing](../../results/history_guided_testing/README.md)。已有测量、检查点和选例轨迹会复用；修改实验参数时需删除受影响的产物再运行。

## 被测算法和场景

目标是 FVDM 纵向速度规划/跟驰算法。它依据车身净间距确定期望速度，并用与前车的速度差修正加速度。目标期望速度 23 m/s，初始速度 25 m/s，最大制动 8 m/s²，停车间距 8 m，连续感知延迟 0.15 s；灵敏度 0.5、速度差增益 0.8、间距过渡尺度 8 m。切入车与急刹前车按场景脚本运动，目标根据观测状态驾驶 ego。

历史源为常规 IDM、常规 FVDM、延迟 IDM、延迟 FVDM、制动受限 IDM、预测制动 IDM，形成直接反应、延迟、制动限制和预测四组。延迟源具有 0.6 s 连续观测延迟；受限源最大制动 3 m/s²；预测源使用 2 s 匀速预测与 TTC 触发制动。

仅维护历史库与目标库两个场景库。每个库均含切入 1024 个、前车急刹 1024 个，合计 2048 个。历史库六源共享坐标，统一划分训练 1638 / 验证 410；目标库使用独立 Sobol 坐标。仿真持续 12 s、20 Hz、双车道；参数范围和场景清单见实验报告。

## 方法

1. **历史先验。** 对每个来源取八个最近邻，用高斯距离权重估计风险，再在机制组内平均、对四组等权平均。
2. **跨算法训练。** 将一个历史算法当作待测对象，排除其所属机制组，用其余来源建立先验，学习少量反馈如何修正历史差异。五个种子的历史验证 RMSE 选择默认核，各参考核独立选择检查点。
3. **反馈校正。** 在归一化物理场景参数上使用差异 GP，核由局部通道、三个 Matérn 尺度和场景尺度门组成。两类场景分别建模；默认保留随距离衰减的相关性，取消紧支撑截断。实现内部对应 `global_feedback` 核。
4. **预期风险选例。** 每次反馈后执行 Gaussian 条件更新，选择尚未查询场景中预期风险最大的一个，直至 200 次。在线冻结核参数。

风险 R 为 TTC、DRAC、车身间距三项归一化风险的均方根，再取全轨迹峰值；R>0.5 表示高风险。本文测试器只收到所查询场景的连续风险；RAS-FRT-UQ 沿用二值碰撞反馈。碰撞由环境独立判定，在选例结束后统一用于评价。

默认方法使用历史先验、风险反馈、多尺度物理核和纯风险采集。紧支撑核、单尺度、固定尺度权重、单源先验、常数先验与混合 QD 采集在同一实现内承担组件对照。神经历史特征 h、全局线性校准与旧均值读出已退出主链。

## 基线

|基线|固定候选库实现|
|---|---|
|Uniform random|无放回随机抽样|
|Farthest-first|归一化参数的最远优先遍历|
|kNN historical-risk ranking|与本文相同的历史先验，固定排序|
|GP-UCB|Matérn-5/2 GP 与上置信界；[Srinivas 等](https://arxiv.org/abs/0912.3995)|
|GP-EI|相同 GP 与预期改进；[Jones、Schonlau、Welch](https://link.springer.com/article/10.1023/A:1008306431147)|
|RF surrogate BO|50 棵树、最大深度 10、删除折 Jackknife 方差与 EI|
|BOP-Elites|已知场景参数描述符，按参数单元档案计算 EI；[Kent 等](https://arxiv.org/abs/2307.09326)|
|BAS|单保真 Bayesian adaptive sampling，阈值事件的预期方差下降；[Sinha 等](https://proceedings.mlr.press/v270/sinha25a.html)|
|RAS-FRT-UQ|共享六源历史数据；两类四维响应编码器、响应相似度校正与残差不确定性融合；二值碰撞反馈|

GP/RF 基线共用各种子的 10 个随机初始点，计入 200 次预算。GP 均值 0.5、方差 0.25、长度尺度 0.2、噪声 0.001。本文方法、历史排序和 RAS 共同使用离线历史库，其余七个基线从目标反馈开始建模。历史物理执行成本与在线查询预算分别报告。

RAS 在每类历史训练集上训练 300 轮，每 10 轮按历史验证 BCE 选择检查点，使用相同六源的碰撞标签。原网络结构与融合权重保持不变，类型之间不传播反馈；相似度参数沿用原历史实验的固定值。当前 RAS 比较是对两类场景和六源数据的适配，说明见 [RAS 实现](../ras_frt_uq/README.md)。在线统一的是场景调用预算，各方法的反馈信息差异在结果报告中注明。

## 文件职责和接口

|文件|职责|
|---|---|
|config、scenarios、prepare|固定设置、场景和实测数据准备|
|risk、measurements|被动风险测量与环境执行|
|history、kernel、train|历史先验、物理核和历史训练/验证|
|gp、archive、search|风险反馈、参数单元档案和测试会话|
|baseline、experiment、../ras_frt_uq/unified|九个基线、组件实验和目标反馈逐次披露|
|evaluate、run|评价、图表和默认入口|

输入为归一化四维物理参数加场景类型，形状 n×5。调用 `next_index()` 获得场景编号，执行模拟，将实测风险传给 `observe(risk)`；传入 `None` 表示无有效风险，仍消耗一次预算。

```python
from methods.history_guided_testing.search import TestingSession

session = TestingSession(numeric_candidates)
index = session.next_index()
session.observe(measured_risk)
```

整理状态见 [cleanup.json](../../results/history_guided_testing/cleanup.json)，开发阶段关键负结果见 [development_summary.json](../../results/history_guided_testing/development_summary.json)。
