# Development test: is online mode calibration better than raw history?

Frozen before running the new comparator on 2026-09-24. This reuses the
four already inspected fresh-confirmation seeds and is **development**, not
independent confirmation. Preserve the earlier method traces unchanged.

The test isolates the value of **online target feedback**. Use the same
two historical source banks, all-source-safe eligibility, five-mode
mixed-lane grammar, two targets, 20 Hz ego control and physics, and
common first-ten-query mode-support rule as the existing B=50 study.
For each of four seeds and two targets, execute 50 distinct target cases
sequentially: 400 new physical target episodes in total. Charge every
execution; do not reveal any target outcome before its query.

New comparator `SourceMeanRaw-Static` ranks every eligible candidate by
the arithmetic mean of the two historical continuous responses
`source_y(x)`, without normalization within modes and without any target
feedback. Its within-mode ordering is exactly that of
`ModeLabelShift-Risk` and `ModeShift-Risk` before their additive mode
offsets. Thus this is the strongest direct no-update control for the
proposed online mode-allocation mechanism, rather than a static
equal-quota surrogate. Compare against the existing charged traces for
those two adaptive selectors, `ModeQuantile-Static`, and
`SourceStatic-Marginal`.

Primary endpoint is distinct actually executed ego-collision-bearing
4 x 4 physical cells within modes at query 50. Secondary outcomes:
ego-collision count, collision-or-near-miss count, 3 x 3 and 5 x 5
cell sensitivities, per-mode query allocation, and exact agreement on
repeated physical scenarios. Cell boundaries remain the frozen
source-blind physical bounds.

The online-allocation explanation passes this **development** check
only if `ModeLabelShift-Risk` exceeds `SourceMeanRaw-Static` in mean
primary endpoint on both targets, has a positive seed-cluster bootstrap
95% lower endpoint for the pooled paired difference, and does not
produce fewer mean ego collisions. A positive result still requires
fresh independent confirmation and a prior-art-matched baseline before
any algorithmic novelty claim. A failed result retires the online
allocation explanation in this grammar.
