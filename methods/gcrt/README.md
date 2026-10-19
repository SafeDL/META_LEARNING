# 收益校正风险测试

收益校正风险测试（GCRT）是本文方法。车辆切入与前车紧急制动分别构建模型、运行和停止。

## 怎样工作

1. **根据历史记录预测风险。**历史响应关联编码器用注意力机制学习场景之间的风险联系。联合风险后验融合六个历史来源和一个目标独立分量，形成当前驾驶控制器的风险预测。
2. **逐次选择场景。**分段条件前瞻计算当前五次测试片段中的预期发现数量。每次只执行一个场景，得到实测风险后更新后验，再选择下一个场景。
3. **预测还值得测多少。**剩余发现数量预测网络汇总当前反馈和未测场景的风险预测，估计继续测试的新增发现数及其分布。
4. **用实际发现校正预测。**每完成五次测试，比较预测和实测发现比例，拟合本任务的偏差，并按预测长度调整剩余发现分布。
5. **决定继续、停止和后续学习。**从第 10 次起，每五次检查一次发现保留条件。任务结束后，新增记录可用于更新共享预测头，验证通过后再发布并重新校准停止阈值。

每类场景使用两个神经网络：历史响应关联编码器、剩余发现数量预测网络。任务内更新风险后验和一个收益校正系数，网络权重保持固定。这七个分量组成一个联合后验，每轮共用一次实测反馈。

## 运行

在项目根目录、`metadrive` 环境执行：

```powershell
python -B -m methods.gcrt.build
python -B -m methods.gcrt.evaluation
python -B -m methods.gcrt.report
```

- `build`：按功能场景读取历史记录，构建基础核、历史编码器、联合先验、历史轨迹和剩余发现网络。已有组件直接复用。
- `evaluation`：复用或补齐 36 条完整轨迹，按实际反馈逐步回放停止判断，完成一次预测头更新和成本分析。
- `report`：从已有记录重新生成结果表和发现曲线。

核心接口为 `selection.select(prior, oracle, ..., initial_indices=...)`。`stopper=None` 时执行完整预算；传入 `train.load_stopper('cutin')` 或 `train.load_stopper('brake')` 启用对应场景的停止规则。

## 代码对应

| 文件 | 内容 |
|---|---|
| [config.py](config.py)、[build.py](build.py) | 固定设置与完整构建入口 |
| [posterior.py](posterior.py)、[selection.py](selection.py) | 联合风险后验、虚拟条件更新与场景选择 |
| [model.py](model.py)、[controller.py](controller.py) | 剩余发现网络、片段校正、分布调整与停止判断 |
| [fit.py](fit.py)、[data.py](data.py)、[train.py](train.py) | 先验拟合、历史监督、反馈回放、网络训练与周期更新 |
| [evaluation.py](evaluation.py)、[report.py](report.py) | 隔离风险回放、停止评价与完整成本 |

共享组件见 [common](../common/README.md)，历史编码器与基础核分别复用 [DART](../baselines/dart/README.md)、[HGT](../baselines/hgt/README.md) 的训练实现。

历史验证与正式评价共用 `data.forecast_task` 回放真实反馈，按相同规则完成片段校正。未来结果只用于离线评分，不参与当时的预测或停止判断。

已完成两类模型、36 条完整轨迹和一次周期更新。切入保留原预测头，制动采用 400 步更新后的预测头。当前结果见 [实验报告](../../results/gcrt/comparison.md)，[方法图](../../method-overview.png)位于项目根目录。
