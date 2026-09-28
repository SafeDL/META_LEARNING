# PPO confirmation of the role-gated revision

This split contains 225 previously unmeasured scenes and 675 physical episodes
from fixed, weight-inherited V0/V1/V2 PPO checkpoints. The selector and eight
comparison methods were frozen before measuring target outcomes. All paired
labels are valid; no unknowns remain. Ten random repeats replay the same bank.

| Update | Direction | True changes | Directed role gated @20 (area) | Static risk | Static role coverage | Coordinate role gated |
|---|---|---:|---:|---:|---:|---:|
| V0→V1 | Regression | 15 | 7 (0.386) | 0 | 14 (0.581) | 7 (0.381) |
| V0→V1 | Improvement | 4 | 3 (0.210) | 0 | 3 (0.119) | 3 (0.210) |
| V1→V2 | Regression | 0 | NA | NA | NA | NA |
| V1→V2 | Improvement | 6 | 2 (0.138) | 0 | 1 (0.071) | 2 (0.138) |

The method improves on the old static-risk ranking for tasks with changes,
but it does **not** establish the paper's claimed overall advantage. The
parent-only static role coverage finds twice as many V0→V1 regressions, and
the ordinary coordinate model exactly ties the directed model on both
improvement tasks. On V1→V2 improvements, ten random replays also average
2/6 discoveries. The unchanged V1→V2 regression direction provides no
evidence of regression discovery. Family-level paired comparisons are not
significant with only three related scenario families. The first update's
regressions occur in S01 and S08, improvements in S02; the second update's
improvements occur in S02.

The complete truth, query order, context and region summaries, family-level
statistics, physical cost ledger, and figures are preserved here. This
finding is a new-scene confirmation of a method developed post-hoc on earlier
NL/PPO banks; those earlier explorations are not additional confirmations.
