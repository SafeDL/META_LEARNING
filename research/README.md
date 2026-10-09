# 研究方法与实验索引

当前研究统一依据 [Failure_Discovery_Design.md](../docs/Failure_Discovery_Design.md)，实现与新结果位于 [representative_sut_testing](representative_sut_testing/README.md)。六个固定代表 SUT 共享两类场景，预算为 200 次；当前方法仍属开发，尚未建立显著优势。

```powershell
conda activate metadrive
python -B -m research.representative_sut_testing.summarize
```

原始方法仍位于 `methods/history_guided_testing/`，原始结果仍位于 `results/history_guided_testing/`；这些文件与根目录说明属于冻结快照，保持原样。

既有研究链与结果按各自 README 和归档保留，用于基线比较和追溯。清理前的研究目录存于 `history_response_testing/archives/development.zip`。旧设计不再作为现行方案入口；方法设计与实验计划只维护上面的统一文档。

当前结果入口：[完整现有基线](representative_sut_testing/results/baseline_comparison.md)、[指标索引](representative_sut_testing/results/comparison.md)、[阶段报告](representative_sut_testing/results/stage_report.md)。
