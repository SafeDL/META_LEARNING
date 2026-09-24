# Raw historical ranking versus online mode feedback

This B=50 test isolates whether target event labels improve over a static
ranking by mean historical continuous safety response. The protocol was
frozen before running the new comparator. It reuses four already inspected
seeds and two targets, so this is development evidence rather than an
independent confirmation.

Across four seeds and two targets, the static comparator physically executed
400 new target episodes. Each method used 50 distinct charged queries per
seed-target unit. The analysis verified 504 repeated physical scenarios
against stored adaptive and static traces. The primary endpoint is the
number of distinct 4 x 4 cells containing an executed ego collision.

| Method | VI/TTC cells | FVDM-revision cells | Mean cells | Mean ego collisions | Mean new failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| `SourceMeanRaw-Static` | 18.25 | 18.75 | 18.500 | 35.375 | 43.250 |
| `ModeLabelShift-Risk` | 24.00 | 20.25 | **22.125** | 41.500 | 45.625 |
| `ModeShift-Risk` | 24.00 | 20.00 | 22.000 | 41.625 | 45.750 |
| `ModeQuantile-Static` | 20.50 | 19.75 | 20.125 | 37.000 | 45.375 |
| `SourceStatic-Marginal` | 18.50 | 18.75 | 18.625 | 35.125 | 43.375 |

`ModeLabelShift-Risk` exceeds raw static ranking by 5.75 cells on VI/TTC
and 1.50 on the FVDM revision. The pooled paired difference is +3.625 cells
per seed-target unit, with a four-seed cluster-bootstrap interval of
[2.625, 4.500]. It also finds more ego collisions on average. The gain
persists at the reported 3 x 3 and 5 x 5 grid resolutions.

This supports online allocation using revealed mode-level event labels over
the tested no-feedback historical mean. It does not isolate a novel
algorithmic contribution: this selector is closely related to simple
per-mode adaptive allocation, the seeds were already inspected, and both
targets use the same simulator family and scenario grammar. A new-seed,
prior-art-matched confirmation remains necessary before making a broader
claim. The wider method boundary and failed component tests are summarized
in [the revised research position](../../../../../docs/core_mine_revised_research_position.md).

Per-run ledgers and all metrics are in `analysis50.json` and the two
seed-target subdirectories. Reproduce the frozen analysis with:

```powershell
conda run -n metadrive python -m method_chains.core_mine.source_raw_static_development --stage analyze
```
