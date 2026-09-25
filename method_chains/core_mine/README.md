# CoRe-Mine 历史实验

这里保留早期 CoRe-Mine 方法及其负结果，便于追溯旧实验。它不是当前回归测试主链；历史结果位于 `results/method_chains/core_mine/`。

FBRT 仍复用本目录中的 IDM 参考配置和受控故障实现：

- `idm_revision_pilot.py` 提供 `REFERENCE`；
- `local_fault_idm.py` 提供 `FAULTS`、`LocalFault` 和 `LocalFaultIDMVehicle`。

这两份实现被当前主链直接导入，不能作为废弃代码删除。其余 CoRe-Mine 脚本与结果属于独立的历史实验，不参与 FBRT 的正式结果生成。

CoRe-Mine 的旧开发协议已从 `docs/` 清理；当前 FBRT 的入口和结果说明见根目录 `README.md` 与 `docs/FBRT_STANDARD_ALIGNED_CODEX_PLAN.md`。
