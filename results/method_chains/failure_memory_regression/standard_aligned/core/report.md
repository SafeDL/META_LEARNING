# FBRT core 结果

场景为规范启发的 highway-env 研究场景；参考通过指完整时长无 ego 碰撞。
所有选择器仅通过逐次查询接口看到目标结果；方法共用同一实测结果库。

候选数：320；历史局部边界对：67；
本配置使用的实测参考 episode：480；实测目标 episode：960。
本次从已保存的实测结果库重放选例，新增仿真 episode：0。

## 每个种子与目标的 @50 首次回归查询位置

| 种子 | 目标 | 方法 | 检测 | 首次位置 | 碰撞数 |
|---|---|---|---:|---:|---:|
| 4179801 | merge_blind06 | Random | 1 | 31 | 4 |
| 4179801 | merge_blind06 | ART-Maximin | 1 | 1 | 5 |
| 4179801 | merge_blind06 | HistoryMargin | 1 | 1 | 5 |
| 4179801 | merge_blind06 | FailureDistance | 1 | 1 | 7 |
| 4179801 | merge_blind06 | HistoryRank-UCB | 1 | 4 | 6 |
| 4179801 | merge_blind06 | FBRT-Static | 1 | 1 | 7 |
| 4179801 | merge_blind06 | FBRT-Adaptive | 1 | 1 | 7 |
| 4179801 | merge_blind06 | FBRT-RegionBandit | 1 | 3 | 7 |
| 4179801 | merge_brake2 | Random | 1 | 41 | 2 |
| 4179801 | merge_brake2 | ART-Maximin | 1 | 1 | 2 |
| 4179801 | merge_brake2 | HistoryMargin | 1 | 1 | 2 |
| 4179801 | merge_brake2 | FailureDistance | 1 | 1 | 4 |
| 4179801 | merge_brake2 | HistoryRank-UCB | 1 | 4 | 4 |
| 4179801 | merge_brake2 | FBRT-Static | 1 | 6 | 4 |
| 4179801 | merge_brake2 | FBRT-Adaptive | 1 | 4 | 4 |
| 4179801 | merge_brake2 | FBRT-RegionBandit | 1 | 4 | 4 |
| 4179801 | slow_front_brake2 | Random | 1 | 1 | 32 |
| 4179801 | slow_front_brake2 | ART-Maximin | 1 | 2 | 31 |
| 4179801 | slow_front_brake2 | HistoryMargin | 1 | 3 | 47 |
| 4179801 | slow_front_brake2 | FailureDistance | 1 | 2 | 29 |
| 4179801 | slow_front_brake2 | HistoryRank-UCB | 1 | 1 | 47 |
| 4179801 | slow_front_brake2 | FBRT-Static | 1 | 1 | 32 |
| 4179801 | slow_front_brake2 | FBRT-Adaptive | 1 | 2 | 32 |
| 4179801 | slow_front_brake2 | FBRT-RegionBandit | 1 | 1 | 47 |
| 4179802 | merge_blind06 | Random | 1 | 8 | 4 |
| 4179802 | merge_blind06 | ART-Maximin | 1 | 1 | 4 |
| 4179802 | merge_blind06 | HistoryMargin | 1 | 1 | 7 |
| 4179802 | merge_blind06 | FailureDistance | 1 | 2 | 12 |
| 4179802 | merge_blind06 | HistoryRank-UCB | 1 | 4 | 9 |
| 4179802 | merge_blind06 | FBRT-Static | 1 | 5 | 12 |
| 4179802 | merge_blind06 | FBRT-Adaptive | 1 | 4 | 12 |
| 4179802 | merge_blind06 | FBRT-RegionBandit | 1 | 4 | 11 |
| 4179802 | merge_brake2 | Random | 1 | 9 | 2 |
| 4179802 | merge_brake2 | ART-Maximin | 1 | 1 | 1 |
| 4179802 | merge_brake2 | HistoryMargin | 1 | 1 | 3 |
| 4179802 | merge_brake2 | FailureDistance | 1 | 4 | 5 |
| 4179802 | merge_brake2 | HistoryRank-UCB | 1 | 4 | 5 |
| 4179802 | merge_brake2 | FBRT-Static | 1 | 5 | 5 |
| 4179802 | merge_brake2 | FBRT-Adaptive | 1 | 4 | 5 |
| 4179802 | merge_brake2 | FBRT-RegionBandit | 1 | 4 | 5 |
| 4179802 | slow_front_brake2 | Random | 1 | 1 | 34 |
| 4179802 | slow_front_brake2 | ART-Maximin | 1 | 2 | 32 |
| 4179802 | slow_front_brake2 | HistoryMargin | 1 | 4 | 46 |
| 4179802 | slow_front_brake2 | FailureDistance | 1 | 1 | 29 |
| 4179802 | slow_front_brake2 | HistoryRank-UCB | 1 | 1 | 47 |
| 4179802 | slow_front_brake2 | FBRT-Static | 1 | 1 | 31 |
| 4179802 | slow_front_brake2 | FBRT-Adaptive | 1 | 2 | 34 |
| 4179802 | slow_front_brake2 | FBRT-RegionBandit | 1 | 1 | 47 |
| 4179803 | merge_blind06 | Random | 1 | 1 | 7 |
| 4179803 | merge_blind06 | ART-Maximin | 1 | 1 | 7 |
| 4179803 | merge_blind06 | HistoryMargin | 1 | 1 | 8 |
| 4179803 | merge_blind06 | FailureDistance | 1 | 1 | 11 |
| 4179803 | merge_blind06 | HistoryRank-UCB | 1 | 4 | 8 |
| 4179803 | merge_blind06 | FBRT-Static | 1 | 2 | 11 |
| 4179803 | merge_blind06 | FBRT-Adaptive | 1 | 2 | 11 |
| 4179803 | merge_blind06 | FBRT-RegionBandit | 1 | 4 | 10 |
| 4179803 | merge_brake2 | Random | 1 | 1 | 3 |
| 4179803 | merge_brake2 | ART-Maximin | 1 | 1 | 4 |
| 4179803 | merge_brake2 | HistoryMargin | 1 | 1 | 3 |
| 4179803 | merge_brake2 | FailureDistance | 1 | 1 | 4 |
| 4179803 | merge_brake2 | HistoryRank-UCB | 1 | 4 | 4 |
| 4179803 | merge_brake2 | FBRT-Static | 1 | 9 | 4 |
| 4179803 | merge_brake2 | FBRT-Adaptive | 1 | 2 | 4 |
| 4179803 | merge_brake2 | FBRT-RegionBandit | 1 | 4 | 4 |
| 4179803 | slow_front_brake2 | Random | 1 | 5 | 32 |
| 4179803 | slow_front_brake2 | ART-Maximin | 1 | 2 | 32 |
| 4179803 | slow_front_brake2 | HistoryMargin | 1 | 3 | 46 |
| 4179803 | slow_front_brake2 | FailureDistance | 1 | 4 | 31 |
| 4179803 | slow_front_brake2 | HistoryRank-UCB | 1 | 1 | 47 |
| 4179803 | slow_front_brake2 | FBRT-Static | 1 | 1 | 36 |
| 4179803 | slow_front_brake2 | FBRT-Adaptive | 1 | 1 | 36 |
| 4179803 | slow_front_brake2 | FBRT-RegionBandit | 1 | 1 | 47 |

## 原型做通了什么

四类功能场景均完成参考版与三个局部修改版的实际执行；共同候选池有 320 个参考通过场景。历史失败和邻近通过样本形成 67 个局部配对，这些配对进入 FBRT 的选例分数。目标结果只在每次逻辑查询时揭示。

## 跨种子预算结果

每个受测版本和种子构成一项检测任务，共 9 项。随机方法每项重放 20 次，表中给出平均检测任务数；其他方法每项运行一次。

| 方法 | @5 检测/9 | @10 检测/9 | @20 检测/9 | @50 检测/9 | @50 平均回归碰撞数 | @10 平均回归功能数 |
|---|---:|---:|---:|---:|---:|---:|
| Random | 4.95 | 6.15 | 7.85 | 8.95 | 12.60 | 1.29 |
| ART-Maximin | 9.00 | 9.00 | 9.00 | 9.00 | 13.11 | 1.67 |
| HistoryMargin | 9.00 | 9.00 | 9.00 | 9.00 | 18.56 | 1.00 |
| FailureDistance | 9.00 | 9.00 | 9.00 | 9.00 | 14.67 | 1.67 |
| HistoryRank-UCB | 9.00 | 9.00 | 9.00 | 9.00 | 19.67 | 1.78 |
| FBRT-Static | 7.00 | 9.00 | 9.00 | 9.00 | 15.78 | 2.00 |
| FBRT-Adaptive | 9.00 | 9.00 | 9.00 | 9.00 | 16.11 | 2.00 |
| FBRT-RegionBandit | 9.00 | 9.00 | 9.00 | 9.00 | 20.22 | 2.00 |

@5 中，FBRT-Static 检出 7.00/9，FBRT-Adaptive 检出 9.00/9，HistoryMargin 和 FailureDistance 分别为 9.00/9 与 9.00/9。这说明原型可以在少量新版本查询中发现真实回归；简单历史排序在当前场景中同样有效。
@10 的功能数是发现回归碰撞的不同功能场景数，不等于独立软件缺陷数。FBRT-RegionBandit 把历史失效边界的局部排序与新版本反馈的跨功能分配结合；它在本轮开发结果库中的平均碰撞数略高于 HistoryRank-UCB，但该差异来自同一结果库上的方法开发，不能当作独立验证或统计优势。

## 一条可追溯的选例链

参考版实测失败场景 `4179801:fbrt_cutout_static:initial:15` 与附近实测通过场景 `4179801:fbrt_cutout_static:core_boundary:3` 构成局部配对；FBRT-Static 根据这段历史，在第 1 次查询选中 `4179801:fbrt_cutout_static:initial:23`。同一场景参考版通过，`merge_blind06` 实测发生 ego 碰撞。

## 按功能观察

下表把 3 个种子的参考结果与三个目标修改的回归碰撞合计；参考结果只计一次，不因目标版本重复。

| 功能 | 参考通过 | 参考碰撞 | 目标回归碰撞（3 版本合计） | 局部边界对 |
|---|---:|---:|---:|---:|
| fbrt_cutin | 101 | 19 | 32 | 19 |
| fbrt_cutout_static | 69 | 51 | 80 | 24 |
| fbrt_lead_emergency_brake | 30 | 90 | 30 | 24 |
| fbrt_stop_hold_go | 120 | 0 | 102 | 0 |

停车—保持—起步若无参考碰撞，就按已记录的历史裕度回退；其目标回归仍属于真实执行结果。
局部修改按车辆状态触发，报告不把每个场景随机种子解释为独立软件缺陷。

## 物理成本与回放

本配置使用参考 480 次、目标 960 次实测 episode；本次从保存的结果库离线重放，新增仿真 0 次。逐方法逻辑查询见 `compute_ledger.json`。
配对回放：`4179801:fbrt_cutout_static:initial:23`；参考无碰撞，`merge_blind06` 于 9.9 s 与 `static` 碰撞。`replay/paired.gif` 每 0.05 s 一帧，`replay/paired_trace.json` 保留全部状态。
停车—保持—起步的参考版速度曲线在 `replay/stop_hold_go_speed.png`，对应 20 Hz 轨迹在 `replay/stop_hold_go_trace.json`。

当前结论对应本轮受控 IDM 修改和 highway-env 功能场景。
