# 推断失败的诊断

`matched_scenes.py` 在同一组 24 个场景上测量 12 个既有验证 SUT。场景来自已公开的 idm_12/pool_0，按该参考系统风险最接近 0.99、0.65、0.20 的点各取 4 个／场景族；两族共 24 场景。

严格复用场景初始条件、参数、仿真种子、SUT 设置、风险公式和碰撞判据。总预算 288 次物理测量，只作响应差异诊断，不进入方法训练或独立确认，也不替换完整开发测试池。参考系统重跑必须与既有物理记录一致。

入口：`python -B -m research.risk_inference_meta_testing.diagnostics.matched_scenes`。结果位于上级 `results/matched_scene_diagnostic/`。
