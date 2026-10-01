# 历史 IDM 库 A

A 是已冻结的历史库：同一个 IDM 控制器的五组参数分别在相同的 2,048 个 S01 场景上执行。五个历史响应模型只使用 A 中的物理响应训练并冻结。

A 为 [D 上的 FVDM 测试](../s01_uniform_fvdm_speed_23_mps/README.md)提供历史经验；其中没有 D 的 FVDM 标签。

- `history_manifest.jsonl`：2,048 个历史场景。
- `banks/`：五组 IDM 参数在所有历史场景上的物理响应。
- `models/`、`frozen_training.json`：冻结权重与训练设置。
- `protocol.json`：源构建、仿真契约与文件摘要。
- `source_snapshot/fbrt_unified_env.py`：迁移共享模块名称前的仿真器源码字节，用于核对冻结协议中的原始源码哈希；当前执行仍使用 `highway_sim_env/envs/unified_env.py`。
