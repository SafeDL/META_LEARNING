# MetaDrive 仿真与方法实现

本目录维护本文方法在 MetaDrive 上的完整实现，包括地图与场景语义、
安全判定、Risk Mining、Formal Teacher、实验脚本和测试。它不包含外部论文复现或
Mining–DETOUR 融合代码。

MetaDrive 被测控制器统一位于根目录 `sut_algorithms/metadrive/`。

主要目录：

- `scenario/`、`map/`、`control/`、`safety/`：仿真和安全语义；
- `mining/`：Risk Mining 的先验、后验、采集和挖掘；
- `formal/`：Formal Teacher 的教师与形式事件校准；
- `scripts/`：数据构建与离线评价入口；
- `tests/`：方法和仿真契约。

```powershell
conda run -n metadrive python -m pytest metadrive_sim_env/tests -q -p no:cacheprovider
conda run -n metadrive python -m metadrive_sim_env.scripts.validate_formal_teacher
```

Formal Teacher 当前入口使用 `configs/formal_teacher.yaml`，结果写入语义化的
`results/metadrive/formal_teacher/cutin_g01/formal_teacher/`。已失败停止且被当前协议
取代的早期教师结果、脚本和配置均不再保留。
