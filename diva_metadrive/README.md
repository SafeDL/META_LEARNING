# DIVA MetaDrive implementation

本目录维护本文方法在 MetaDrive 上的完整实现，包括地图与场景语义、控制器、
安全判定、DIVA-Mine、DIVA-Former、实验脚本和测试。它不包含外部论文复现或
DIVA–DETOUR 融合代码。

主要目录：

- `scenario/`、`map/`、`control/`、`safety/`：仿真和安全语义；
- `diva/`：DIVA-Mine 的先验、后验、采集和挖掘；
- `diva_ai/`：DIVA-Former 的教师与形式事件校准；
- `scripts/`：数据构建与离线评价入口；
- `tests/`：方法和仿真契约。

```powershell
conda run -n metadrive python -m pytest diva_metadrive/tests -q -p no:cacheprovider
conda run -n metadrive python -m diva_metadrive.scripts.validate_diva_former
```

DIVA-Former 当前入口使用 `configs/diva_former.yaml`，结果写入语义化的
`results/metadrive/diva_former/cutin_g01/formal_teacher/`。已失败停止且被当前协议
取代的早期教师结果、脚本和配置均不再保留。
