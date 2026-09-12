# Highway-env DIVA-Mine MVP 执行记录

本执行将 Highway-env 作为与 MetaDrive 并列的轻量仿真器；它不替换
`mvr/metadrive/diva/`、`mvr/metadrive/scenario/` 或任何 MetaDrive 实验代码。

## 工程位置

- 实现：`mvr/highway/`
- 命令入口：`mvr/highway/scripts/run_diva_highway_mvp.py`、`mvr/highway/scripts/render_diva_highway_mvp.py`
- 测试：`mvr/tests/test_diva_highway.py`、`mvr/tests/test_diva_highway_cutin.py`
- 工件：`results/diva_highway/cutin_mvp/`

`_highway` 后缀只用于 Highway-env 的结果、入口和可视化文件，避免与 MetaDrive 结果混淆。

## 运行

```powershell
conda run -n metadrive python -m pytest mvr/tests/test_diva_highway.py mvr/tests/test_diva_highway_cutin.py -q
conda run -n metadrive python -m mvr.highway.scripts.run_diva_highway_mvp --rebuild-bank
conda run -n metadrive python -m mvr.highway.scripts.render_diva_highway_mvp
```

环境依赖固定于 `environment.yml`：`highway-env==1.9.1`，与项目当前 `gymnasium=0.29.1` 兼容。

## 可视化记录

结果目录保留四张静态图：脆弱性图、SVD 解释方差、K=4 ranking 和 B=20 mining 曲线；并对 SUT-A 至 SUT-F 分别记录一个代表性高风险 Cut-in GIF。`visualization_manifest_highway.json` 记录每段动画使用的场景、帧数和安全结局。

## 当前结论

完整 6×128 response bank 的首轮机制检查未证实 DIVA-Mine 优势：rank-2 EVR 为 95.55%，但 Shared Prior 的 NDCG@10 为 0.987、Score@20 为 17.917，均未被 adapted 方法超越。因此应先改进非嵌套 SUT 失效模式或诊断采集准则，不能据此声称 Highway-env 验证成功。详见 `results/diva_highway/cutin_mvp/experiment_summary_highway.md`。
