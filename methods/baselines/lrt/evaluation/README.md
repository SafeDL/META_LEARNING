# LRT 实验

本目录保存局部风险迁移（LRT）的实验入口。本文最新方法的入口在 [GCRT](../../../gcrt/README.md)。

| 入口 | 实验范围 |
|---|---|
| `pipeline.py` | 两类场景共同使用 200 次预算；主比较、消融与档案失配实验 |
| `by_family.py` | 切入、制动分别训练和测试；种子 11、各 200 次，共 288 条轨迹 |
| `family_report.py`、`family_verify.py` | 汇总并核验按类实验的效果与成本 |
| `adaptive.py`、`adaptive_report.py` | 使用固定 LRT 查询顺序的计数停止对照 |
| `diagnostics.py` | 在评价端计算分支续期收益和少量反馈诊断 |

在项目根目录、`metadrive` 环境中查看按类结果：

```powershell
python -B -m methods.baselines.lrt.evaluation.family_report
python -B -m methods.baselines.lrt.evaluation.family_verify
```

真实风险保存在评价进程，测试方法只逐次接收已请求的风险。分支审计和危险场景总数用于评价，不返回测试方法。

结果见[联合实验](../../../../results/baselines/lrt/README.md)、[按类实验](../../../../results/baselines/lrt/by_family/README.md)和[计数停止对照](../../../../results/baselines/lrt/adaptive/README.md)。
