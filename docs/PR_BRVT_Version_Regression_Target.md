# PR-BRVT Version Regression MVP

本实现将 PR-BRVT 限定为受控控制器版本的固定场景回归测试：严格更早
的同谱系版本拟合低秩先验，直接前一版本确认 noncritical 的场景构成统一
候选集，新版本仅通过一次次 reveal 提供观测。

正式协议见 [highway+target.md](highway+target.md)。冻结输入、两个 bank、
回放明细和结论写入 `results/diva_highway/version_regression_v1/`。旧 A--F
LOSO、PR-DVM 和 PR-BRVT 结果仍是历史跨系统实验，不构成版本回归证据。

本轮只比较 `Random-Eligible`、`Previous-Boundary`、`Frozen-History` 和
`Sequential-History`。主指标是 `RegressionCount@20`，即上一版本无关键
事件且新版本发生 collision 或 near-miss 的真实场景数；它再分解为
`new_in_archive_count` 和 `reintroduced_count`。历史 rarity 不参与选择。

运行顺序：

```powershell
conda run -n metadrive python -m mvr.highway.scripts.run_version_regression_mvp freeze --config mvr/highway/configs/version_regression_mvp.yaml --manifest mvr/highway/configs/version_lineages_v1.json --output results/diva_highway/version_regression_v1
conda run -n metadrive python -m mvr.highway.scripts.run_version_regression_mvp build --run-dir results/diva_highway/version_regression_v1
conda run -n metadrive python -m mvr.highway.scripts.run_version_regression_mvp evaluate --run-dir results/diva_highway/version_regression_v1
```
