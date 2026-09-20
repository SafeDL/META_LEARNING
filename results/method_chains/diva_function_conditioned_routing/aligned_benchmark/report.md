# 普通六SUT对齐基准

该目录复用 `diva_detour_fusion` 的同一响应库，用于检查功能条件化方法在核心假设
不成立时是否负迁移。

- AdaTE Global Recall@50：74.99%
- Function-Conditioned DIVA Recall@50：74.89%
- 差异：-0.09个百分点
- 功能Oracle相对全局Oracle增益：0.00个百分点

这说明原六SUT主要体现全局保守程度差异，并未形成足以改变Top-50排序的功能迁移。
本文方法没有在这里制造虚假优势，性能与AdaTE基本持平。该结果只作为安全性和
向后兼容对照，正式创新证据来自相邻的 `functional_shift_benchmark`。
