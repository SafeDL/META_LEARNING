# 仿真执行与响应档案

本目录保存历史响应和实际执行证据，以及仅供评估端读取的完整目标真值。

| 位置 | 内容 |
|---|---|
| `history/`、`physical/history/` | 原始历史响应及对应执行 |
| `joint_response/batches/` | 原始记录转换与补采批次 |
| `joint_response/history/` | 按功能场景合并的训练档案 |
| `joint_response/physical/` | 补采历史的实际执行与轨迹 |
| `joint_response/targets/` | 各协议的目标执行库及共享执行引用 |

`.npz` 保存参数和响应，`manifest.json` 登记场景、版本与环境种子，执行文件保留轨迹和调用凭据。论文跟车合并库通过引用复用原始执行和补点轨迹。

历史数据由 [prepare_joint.py](../../experiments/prepare_joint.py) 准备，目标库由 [prepare_targets.py](../../experiments/prepare_targets.py)及[论文入口](../../experiments/paper_validation.py)准备。[query.py](../../experiments/query.py)逐次开放已付费反馈；完成任务的归档保存在所属 `runs/` 分支，模型和结果见[产物索引](../README.md)。
