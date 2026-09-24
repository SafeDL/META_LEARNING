# Mining 功能条件化历史先验路由

本目录独立维护本文提出的 `Function-Conditioned Mining`。它解决的问题是：新目标
系统在切入、制动、走停和跟驰功能上可能分别接近不同历史版本，单一 AdaTE 全局
权重会混淆这些功能迁移。

## 方法

对场景功能 (m(x)) 分别学习历史响应权重：

\[
\hat y_R(x)=\sum_j\alpha_{m(x),j}Y_j(x).
\]

前 `K=10` 次测试仍按 AdaTE 风险顺序执行，但在预算耗尽前强制每种功能至少得到
一个真实目标反馈。随后每执行一个目标场景就更新功能权重；历史响应解释不了目标
时，由残差门控回退到 Mining 后验。所有诊断执行都计入总预算 `B=50`，没有免费目标
标签。

## 两个互补基准

结果按语义目录分开，避免再次混淆实验目的。

### `aligned_benchmark`

与既有 `detour_fusion` 完全共用六个 SUT、240 个场景、五种功能、LOSO 划分、
随机种子和响应库。该基准检验新方法在“不存在功能迁移优势”的普通控制器上是否
负迁移。

结果：AdaTE Global 的 Recall@50 为 74.99%，本文方法为 74.89%，差异为 -0.09
个百分点。Oracle 检查显示分功能权重在该基准没有任何 Recall@50 上限，因此不能
要求方法在这里制造不存在的增益；它的作用是安全性对照。

### `functional_shift_benchmark`

六个模块化发布版本由三个真实 highway-env 控制器模块构成，但每个版本在五种功能
上的模块组合不同。每类场景除间距和相对速度外，还有自己的时序和强度参数。所有
不同的“控制器模块—场景”组合都真实执行，不拼接或手写碰撞标签。

固定种子结果：

| 方法 | B=10 | B=20 | B=30 | B=50 |
| --- | ---: | ---: | ---: | ---: |
| Random | 4.05% | 8.13% | 12.12% | 20.53% |
| DETOUR | 11.18% | 21.09% | 29.19% | 44.15% |
| Mining | 6.87% | 19.55% | 30.96% | 46.86% |
| Mining-DETOUR | 7.07% | 19.41% | 30.98% | 47.31% |
| AdaTE Global | **14.39%** | 28.29% | 38.89% | 57.00% |
| Function Routing（无覆盖约束） | **14.39%** | **30.22%** | **45.59%** | 77.06% |
| **Function-Conditioned Mining** | 13.53% | 29.60% | 45.43% | **77.21%** |
| Function Oracle | 16.06% | 32.13% | 48.19% | 78.92% |

本文方法比 AdaTE Global 高 20.20 个百分点，距离使用全部目标响应拟合的功能 Oracle
仅 1.71 个百分点。其 K=10 后未查询响应 MSE 为 0.02754，显著低于相同支持上的
Mining（0.11814）。功能覆盖约束牺牲了 0.86 个百分点的 Recall@10，但修复了部分
功能零样本问题，并在 B=50 略高于无覆盖约束消融。

## 危险场景回放

每个 GIF 都是 `Function-Conditioned Mining` 在 `B=50` 内实际查询并发现的碰撞，使用
对应目标发布版本在该功能上的真实控制器模块和完全相同的场景参数重新执行。为保证
动作过程可读，每类功能选择初始间距最大的已发现碰撞；若某类没有碰撞，代码才会
回退到近失效。选择依据和逐场景参数保存在
[`gifs/manifest.json`](../../results/method_chains/function_conditioned_routing/functional_shift_benchmark/gifs/manifest.json)。

| 功能场景 | 回放 | 目标发布版本 | 时长 |
| --- | --- | --- | ---: |
| 快速侵入 | [`fast_intrusion.gif`](../../results/method_chains/function_conditioned_routing/functional_shift_benchmark/gifs/fast_intrusion.gif) | Modular-Estuary | 4.8 s |
| 切入后制动 | [`cutin_braking.gif`](../../results/method_chains/function_conditioned_routing/functional_shift_benchmark/gifs/cutin_braking.gif) | Modular-Cascade | 4.0 s |
| 前车制动 | [`lead_braking.gif`](../../results/method_chains/function_conditioned_routing/functional_shift_benchmark/gifs/lead_braking.gif) | Modular-Boreal | 3.4 s |
| 走停 | [`stop_and_go.gif`](../../results/method_chains/function_conditioned_routing/functional_shift_benchmark/gifs/stop_and_go.gif) | Modular-Delta | 3.8 s |
| 慢车跟驰 | [`slow_lead_following.gif`](../../results/method_chains/function_conditioned_routing/functional_shift_benchmark/gifs/slow_lead_following.gif) | Modular-Atlas | 3.2 s |

这些回放是“方法确实找到了哪些失效”的定性证据；方法优于基线的定量证据仍是固定
预算 Recall、跨种子统计和 Oracle 差距，不能用五个展示案例替代。

## 当前证明的 idea

证据支持一个有条件的结论：新目标版本不是整体继承某一个历史 SUT，而可能在不同
驾驶功能上分别继承不同历史行为。先用少量、计入预算的目标反馈识别这种功能级来源，
再按功能迁移历史风险，并在历史失配时回退到 Mining，可以在相同测试预算内更快找到
目标系统的碰撞与近失效。对齐基准没有功能迁移空间时，本方法与 AdaTE 基本持平，
说明收益来自被实验显式验证的功能迁移结构，而不是无条件更强的排序器。

## 跨种子确认

开发种子和两个未参与参数选择的确认种子全部胜过 AdaTE：

- 本文方法：77.17% ± 0.80%
- AdaTE Global：56.42% ± 1.37%
- Function Oracle：79.22% ± 0.94%
- 平均增益：20.75 ± 0.67 个百分点，3/3 个种子胜出

机器可读统计见
[`robustness_summary.json`](../../results/method_chains/function_conditioned_routing/functional_shift_benchmark/robustness_summary.json)。

## 目录与重建

```text
detour_fusion/                         Mining–DETOUR 独立对照链
function_conditioned_routing/          本文功能迁移方法链
  benchmark.py                              物理功能迁移基准
  routing.py                                支持集、权重和门控
  experiment.py                             对齐/功能迁移双基准
  robustness.py                             三场景种子确认
  replay.py                                 五类危险场景可复验回放
results/method_chains/function_conditioned_routing/
  aligned_benchmark/                        普通六 SUT 安全性对照
  functional_shift_benchmark/               正式功能迁移结果与 GIF
```

```powershell
conda run -n metadrive python -m pytest method_chains/function_conditioned_routing/tests -q -p no:cacheprovider

conda run -n metadrive python -m method_chains.function_conditioned_routing.experiment

conda run -n metadrive python -m method_chains.function_conditioned_routing.experiment --reuse-bank

conda run -n metadrive python -m method_chains.function_conditioned_routing.robustness

conda run -n metadrive python -m method_chains.function_conditioned_routing.replay
```

算法定义见 [`method.md`](method.md)，逐项验收见 [`validation.md`](validation.md)。正式解释见
[`functional_shift_benchmark/report.md`](../../results/method_chains/function_conditioned_routing/functional_shift_benchmark/report.md)。
