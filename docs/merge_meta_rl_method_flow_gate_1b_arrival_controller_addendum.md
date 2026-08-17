# Gate 1B 到达时间控制器补充记录

日期：2026-08-17。该记录是机制门禁证据，不是论文性能结果。

## 配置与协议

- 任务：物理地图、路线和 mechanism Casebook 完全匹配的 `adversary_first` 与 `sut_first`；二者仅冻结的 conflict-entry order 不同。
- 动作：机制诊断专用的一维目标到达时间差控制器，动作范围对应 `[-0.06, 0.06] s`；横向仍由 route tracker 保持。
- 严格指标：`logical_order_spatiotemporal_near_miss_v3`。碰撞与 VCSR 互斥，target collision 的惩罚为 `-1000`。
- 训练：每个任务独立 SAC，10,000 环境步；每 8 环境步更新一次、batch 64。仅用于 Gate 1B，不是 PEARL baseline 或 holdout 性能评估。

证据目录：`results/pearl_learning/merge_method_flow_logical_order_arrival_controller/gate_1b_single_task_sac_10k_bounded_target/`。

## 2x2 迁移矩阵

行是训练策略，列是评估任务；数值是严格、无碰撞 VCSR（8 个 matched train-pool cases）。

| 训练策略 | adversary-first | sut-first |
| --- | ---: | ---: |
| adversary-first SAC | 0.25 | 0.50 |
| sut-first SAC | 0.00 | 0.25 |

相对本任务策略的 VCSR 优势为：

- adversary-first：`0.25 - 0.00 = +0.25`，满足最小 +0.125 的方向性要求，但仍有 0.75 target-collision rate。
- sut-first：`0.25 - 0.50 = -0.25`，与要求方向相反；其自任务策略有 0.75 invalid rate。

因此 Gate 1B 状态为 **fail**，不得继续 Gate 2、20k PEARL 或正式性能比较。

## 可解释的结论

这次不是“严格目标不可达”：两个任务都能出现严格 near-miss，且 adversary-first 已出现正确的自任务优于迁移策略的信号。真正未解决的是控制接口与固定 IDM 交互造成的非对称风险：为抢先进入冲突区的策略容易跨过无碰撞边界而变为 target collision；反向策略则容易转为 non-target invalid contact。将 collision 惩罚从 200 提高到 1000、以及把目标范围从 ±0.08 s 收紧到 ±0.06 s，均没有同时修复两个方向。

故下一步应是预先声明、且可检验的任务重定义，而不是继续扩大 PEARL 或训练预算：例如把安全到达窗口设计为两个顺序方向均有有限宽度的可控走廊，或引入一个明确、对称且不泄漏 Task label 的响应控制接口。重定义后必须从 Gate 1 和 Gate 1B 重新开始。
