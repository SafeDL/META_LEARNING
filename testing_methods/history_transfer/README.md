# 历史响应迁移与联合反馈测试

本目录实现碰撞—RSS 联合响应、来源后验、目标适配和预算内选例。

| 文件 | 职责 |
|---|---|
| [model.py](model.py)、[shared_private.py](shared_private.py) | 定义原联合核及共享／独立潜过程核 |
| [inference.py](inference.py) | 构建后验、适配目标并分块预测 |
| [training.py](training.py) | 训练基础核和历史编码器，保存模型与构建配置 |
| [selection.py](selection.py) | 查询、记录反馈并更新来源权重 |
| [reliability.py](reliability.py)、[lookahead.py](lookahead.py) | 保留局部权重和联合反馈前瞻算子 |

历史来源保存两个输出的均值及联合协方差块。目标推断先对 RSS 做高斯条件化，再对碰撞标签做 Laplace 推断。来源权重仅按查询前碰撞预测似然更新，下一次选择使用更新后的状态。

在线冻结共享特征，每轮从固定初始化使用全部已付费反馈适配 10 步。目标核与原始参数核各更新六个标量，学习率分别为 0.003 和 0.01；历史组／独立目标的初始比例为 0.75／0.25，原始参数分量权重为 0.1。默认每批预测 128 个候选，按碰撞概率贪心选择。

## 构建与使用

训练先拟合历史联合响应的基础核，再以历史前缀的下一条碰撞标签训练编码器。默认每阶段 1,500 步，学习率 0.001、种子 1301，使用最终模型。输入来自同类场景的历史档案。

论文选定 `SharedPrivateKernel`。下面示例从已有档案构建一套该结构的模型：

~~~python
from pathlib import Path

from experiments.history import load_training_archives
from testing_methods.history_transfer.shared_private import SharedPrivateKernel
from testing_methods.history_transfer.training import Config, train

kind = 'car_following'
history = load_training_archives('artifacts/data/joint_response/manifest.json', kind)
train(
    history,
    Path('artifacts/runs/joint_response/models/shared_private') / kind,
    config=Config(device='cpu'),
    kernel_type=SharedPrivateKernel,
)
~~~

加载该结构时同样传入 `load_model(..., kernel_type=SharedPrivateKernel)`。`Config.device` 和 `select(device=...)` 支持 CPU／CUDA。当前已选模型见[模型索引](../../artifacts/runs/joint_response/method_exploration/e2/models/)。

`python -B -m testing_methods.history_transfer.training` 默认构建 `JointResponseKernel`，输出到各场景的 `all_sources/`。该入口用于原联合核；共享／独立核使用上面的显式类型入口。

[history.build_source](../../experiments/history.py) 根据模型、历史档案和实际候选构建来源。`select` 接收候选、查询接口、共同初始化和预算；无历史联合 GP 通过 `raw_kernel_type` 指定核类型，只使用原始归一化参数和目标反馈。显式参数域通过 `feature_fn=domain.features` 传入。

## 归档与核验

任务归档、来源构建和共享模型发布分别由 [history.py](../../experiments/history.py) 的 `finalize_task`、`build_source`、`update_shared_model` 完成。任务内固定启动时的模型与来源快照。

~~~powershell
python -B -m experiments.check_training
python -B -m pytest tests/history_transfer -q
~~~

`check_training` 使用临时执行与模型核对链路。局部权重和前瞻算子保留各自数值回归；当前主方法采用默认的全局权重与贪心选择。比较记录见[方法比较](../../artifacts/runs/joint_response/method_exploration/)，实验运行见 [experiments/](../../experiments/README.md)。
