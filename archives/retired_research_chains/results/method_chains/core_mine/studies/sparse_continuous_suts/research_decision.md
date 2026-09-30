# CoRe-Mine final sparse-risk decision

## Safety-label audit (2026-09-23)

The old safety interpretation is withdrawn. The B=50 validation banks contain
zero ego collisions. All 345 archived `near_miss` labels have minimum recorded
TTC >= 1.5 s; every event was triggered only by the old distance proxy. That
proxy subtracts half the vehicle lengths from the Euclidean distance between
their centres, ignoring lateral vehicle width. It can report zero clearance
when cars pass side by side in adjacent lanes.

Four B=50 selected cases were replayed at the physical simulation rate (20 Hz).
All four completed without collision; their actual vehicle-polygon clearances
were about 2 m at closest approach. The GIFs in `gifs/` are diagnostic replays
of this false-positive mechanism, not examples of discovered danger. See
`label_audit.json` and `gifs/manifest.json`. A complete replay of all 176
archived MeanResidual-Risk B=50 hits across the nine validation target units
found **zero** corrected collisions or near misses; the smallest actual vehicle
outline clearance was 1.992 m (`geometry_audit.json`). The archived CVS, CriticalCount,
SeveritySum and F values therefore cannot support a safety-testing claim.

The requested evaluation budget is **50 target SUT runs**. The old code did run
50-step campaigns but tuned its configuration and declared its primary
endpoint at B=20. B=50 rows can describe the old proxy labels, but they are
neither a valid safety endpoint nor a prospectively selected primary endpoint.
After correcting the event definition, the scenario bank must be rerun and
the B=50 protocol frozen before inspecting its validation outcomes.

## Historical decision under the invalid proxy labels

## Decision

Do **not** claim that the compositional posterior in CoRe-Mine is viable on
the present Highway-env SUT family. The comparisons below use the archived
invalid proxy labels and are retained only as a history of the experiment.

At the frozen budget of 20, over 3 held-out seeds x 3 held-out targets:

| Method | CVS@20 | Severity@20 | CoRe difference (CVS) |
|---|---:|---:|---:|
| MeanResidual-Marginal | 3.000 | 4.000 | reference |
| CoRe-Marginal | 2.667 | 3.222 | -0.333, bootstrap [-0.722, -0.056] |
| FPS-Marginal | 2.722 | 3.556 | -0.056, bootstrap [-0.333, +0.222] |
| TargetOnlyGP-Marginal | 1.056 | 1.056 | +1.611, bootstrap [+0.611, +2.667] |
| Random | 0.694 | 0.717 | n/a |

The composition ablation is also unfavorable: `NoComposition` reaches CVS@20
2.833, above CoRe-Marginal's 2.667.  `NoNull` ties CoRe-Marginal at 2.667.
The evidence therefore supports the minimal, defensible fallback
**MeanResidual-Marginal** (source-average response plus target residual GP),
not the full compositional claim.

## What was tested

* v2 (`core_mine_sparse_suts`) made risks sparse but concentrated almost
  wholly in one common challenge tail.  CoRe tied FPS-Marginal at CVS@20.
* v3 (`core_mine_continuous_suts`) removed the common-tail overlap, but
  `vi_ttc` was critical in almost every candidate and raised the mean rate to
  36.47%; CoRe remained below MeanResidual-Marginal.
* v4 (`core_mine_sparse_continuous_suts`) excluded that empirically
  constant-density policy only for this sparse test family, shifted the
  continuous headways upward, and used a disjoint qualification seed.  The
  qualification gate confirmed two multi-mode branch-disagreement targets,
  no redundant SUT triplet, and dispersed eligible-target risks.  The
  confirmation pool's mean critical rate is 7.67%.

The v4 manifest records 1,500 qualification and 7,500 confirmation physical
episodes, fixed bank hashes, seeds, SUTs, and worker count.  The target
response remains in the cache oracle until after each method's selection.

## Reproduce

```powershell
$env:OPENBLAS_NUM_THREADS='8'
$env:OMP_NUM_THREADS='8'
$env:MKL_NUM_THREADS='8'
conda run -n metadrive python -m method_chains.core_mine.sparse_sut_experiment `
  --proposal v4_sparse_continuous --stage all --workers 3
```

Existing v4 bank files are reused by this command.  Removing or replacing a
bank would create new physical samples and must be treated as a new study.
