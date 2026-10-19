# LRT 计数停止对照

本目录保存基于固定 LRT 查询顺序的自适应停止实验，作为历史对照。本文当前方法为 [GCRT](../../../gcrt/README.md)。

切入与前车紧急制动分别运行，固定种子 11、最多 200 次测试，前 10 次初始化计费。已完成 72 条历史轨迹、两类停止网络和 36 条正式自适应轨迹，共 3828 次计费回放，无新增物理仿真。

| 功能场景 | 平均查询数 | 发现数：固定 → 停止 | 宏平均覆盖率：固定 → 停止 | 原实验质量验收 |
|---|---:|---:|---:|---|
| 车辆切入 | 153.22 | 120.17 → 110.17 | 88.18% → 81.48% | 通过 |
| 前车紧急制动 | 59.44 | 48.89 → 37.56 | 87.12% → 26.94% | 未通过 |

此项实验的验收要求为保留固定 LRT 至少 90% 的发现数和宏平均覆盖率。制动存在尚未发现危险就停止的任务，其耗时下降作为诊断记录。逐任务结果见 [comparison.md](comparison.md)。

| 位置 | 内容 |
|---|---|
| `cutin/`、`brake/` | 各类停止网络、训练记录和费用 |
| 各类 `historical/`、`trajectories/` | 历史续期轨迹与正式停止查询 |
| 各类 `diagnostics/` | 剩余发现预测及均值停止诊断 |
| `comparison.md`、`comparison.json` | 发现数量、覆盖率、质量保留与成本 |
| `protocol.json`、`frozen_inputs.zip`、`reviewed_interfaces.zip` | 当时的设置、权重、源码与停止接口 |
| `verification.json` | 严格前缀、反馈隔离和计费核验 |

入口位于 `methods/baselines/lrt/evaluation/adaptive.py`、`adaptive_report.py`，实验设置与实现见 [LRT](../../../../methods/baselines/lrt/README.md)。
