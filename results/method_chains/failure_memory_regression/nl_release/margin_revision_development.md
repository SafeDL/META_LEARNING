# Margin and frontier revision (exploratory)

The third confirmation diagnosed a specific failure: parent binary labels
were all pass in the S08 context containing five new regressions, so the
parent edge feature was absent. `min_ttc` is already recorded by the complete
**parent** physical bank. The revised selector uses this continuous parent
margin to probe the lowest-TTC parent-pass scenario once per context before
reusing charged target feedback. After a regression is found, it prioritizes
the four-neighbor grid frontier of the latest discovered context, then falls
back to the context-specific UCB target model. Improvement queries keep the
same directed Laplace model. It has no target-bank access beyond the oracle.

Two ablations isolate the additional ingredients: `static_margin_coverage`
does the same first context sweep but no target adaptation, and
`coordinate_margin_frontier` uses the same sweep and spatial frontier with an
ordinary coordinate residual instead of directed boundary features. All have
the same 20+20 schedule and use one physical target episode per query.

The following replays on already-read banks are **post-hoc diagnostics**, not
independent confirmation for this revision:

| Bank | True R / I | Directed margin frontier R / I @20 | Static risk R / I | Coordinate margin frontier R / I |
|---|---:|---:|---:|---:|
| 7×7 development | 1 / 4 | 1 / 4 | 1 / 4 | 1 / 4 |
| Second confirmation | 2 / 9 | 2 / 9 | 0 / 4 | 2 / 4 |
| Third confirmation | 6 / 21 | 6 / 16 | 1 / 12 | 6 / 14 |

The regression gain on these old banks comes mainly from context coverage and
frontier feedback: the coordinate-frontier ablation finds the same R points.
The directed features improve I discovery relative to that ablation in both
old confirmation banks. These observations motivate an independent frozen
test; they cannot be counted as prospectively validated superiority.
