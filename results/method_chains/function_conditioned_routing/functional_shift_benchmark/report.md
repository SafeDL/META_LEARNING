# 物理功能迁移基准结果

## 为什么需要该基准

原六 SUT 对照的功能 Oracle 与全局 Oracle 在 Recall@50 上完全相同。继续在该设置
调参不能证明功能条件化创新。新基准将“不同功能继承不同历史版本”作为可执行的
软件发布结构，并增加功能专属时序和强度，使实验与方法假设一致。

## 结果

固定开发种子 `20260914` 下：

| 方法 | Recall@50 |
| --- | ---: |
| Random | 20.53% |
| DETOUR | 44.15% |
| Mining | 46.86% |
| Mining-DETOUR | 47.31% |
| AdaTE Global | 57.00% |
| Function Routing（风险支持） | 77.06% |
| **Function-Conditioned Mining** | **77.21%** |
| Function Oracle | 78.92% |

本文方法相对 AdaTE 提高20.20个百分点，相对Mining提高30.35个百分点，达到功能
Oracle召回的97.83%。预测层面，K=10后的未查询严重度MSE由Mining的0.11814
降至0.02754；纯路由MSE为0.02432，说明增益来自可恢复的功能迁移结构。

## 组件解释

无覆盖约束的功能路由已经达到77.06%，说明主要贡献来自功能条件化权重。覆盖约束
保证每种功能至少一个诊断样本，B=50再提高0.14个百分点；代价是B=10从14.39%
降到13.53%。因此它的作用是可辨识性和最坏功能保护，不应宣称其提高所有预算点。

Mining门控在历史失配时提供回退，但本基准的历史模块确实能解释目标，所以门控只带来
很小修正。最终尺度0.30在开发种子上选择，确认种子没有重新调参。

## 稳健性

三个场景种子的Recall@50：

| 种子 | AdaTE | 本文方法 | Function Oracle |
| --- | ---: | ---: | ---: |
| 20260914 | 57.00% | 77.21% | 78.92% |
| 20261011 | 54.85% | 76.34% | 78.46% |
| 20261017 | 57.40% | 77.95% | 80.27% |

平均增益为20.75±0.67个百分点，三个种子全部胜出。当前证据支持受控功能回归场景
中的稳定优势，但不等同于真实道路部署或任意黑盒系统的普适保证。

## 定性回放

五类功能各保留一个由本文方法在 `B=50` 内真实查询到的碰撞回放。回放使用响应库中
相同的目标功能模块、随机种子、初始间距、相对速度、时序和强度，不重新搜索参数。
为避免瞬时碰撞掩盖功能动作，每类选择初始间距最大的已发现碰撞。

- [`fast_intrusion.gif`](gifs/fast_intrusion.gif)：相邻车道快速侵入后碰撞；
- [`cutin_braking.gif`](gifs/cutin_braking.gif)：切入并制动后碰撞；
- [`lead_braking.gif`](gifs/lead_braking.gif)：同车道前车制动后碰撞；
- [`stop_and_go.gif`](gifs/stop_and_go.gif)：长制动走停过程中碰撞；
- [`slow_lead_following.gif`](gifs/slow_lead_following.gif)：慢车跟驰闭合过程中碰撞。

逐案例的索引、物理参数、事件标签和回放时长见 [`gifs/manifest.json`](gifs/manifest.json)。
GIF 用于核验场景机制和失效可解释性，不参与 Recall 计算，也不替代全量查询轨迹。

## 可复验产物

- `response_bank.npz`：开发种子的物理响应库；
- `benchmark_manifest.json`：模块映射、场景范围和冻结配置；
- `oracle_headroom.json`：全局/功能Oracle上限；
- `mining_results.csv`：逐目标、逐方法查询轨迹；
- `routing_diagnostics.csv`：功能权重、门控和预测误差；
- `summary.json`：开发种子汇总；
- `robustness_summary.json`：三种子确认；
- `gifs/`：五类危险场景回放及机器可读清单；
- 三张PNG：预算曲线、逐目标对比和预测误差。
