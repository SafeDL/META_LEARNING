# Independent UCB revision: negative bidirectional confirmation

The protocol and selector rule in `method_selection.md`, `protocol.json`, and
`selector_config.json` were fixed before the 1323 physical episodes were
measured. `frozen_selector_source.py` is the exact source snapshot matching
the selector SHA-256 in the configuration and cost ledger. The 441 scenarios
use three new contexts per S01, S02, and S08 family, with a 7×7 grid per
context. All build/scene rows were complete and valid; there were no unknown
transition labels. Every method used 20 queries per available direction.

| Transition | Direction | True changes | UCB discoveries / early area | Static risk | Coordinate residual | Earlier Laplace |
|---|---|---:|---:|---:|---:|---:|
| nl_v0→nl2_v1 | Regression | 6 | 1 / 0.095 | 1 / 0.095 | 1 / 0.095 | 1 / 0.095 |
| nl_v0→nl2_v1 | Improvement | 0 | NA | NA | NA | NA |
| nl2_v1→nl2_v2 | Regression | 0 | NA | NA | NA | NA |
| nl2_v1→nl2_v2 | Improvement | 21 | 16 / 0.905 | 12 / 0.648 | 14 / 0.757 | 16 / 0.881 |

The UCB revision improves the improvement-direction discovery area but **does
not improve regression discovery**. It does not establish the proposed
bidirectional superiority. The family-level two-sided sign-flip p values are
1.0 for the regression tie and 0.5 for the improvement-area gains (Holm-adjusted
p=1.0). Three related scenario families cannot establish a population-level
effect, and random selector repeats reuse the same physical bank.

Five of the six regressions lie in the all-parent-pass `S08:confirmation3:c2`
context at low grid indices. The parent binary-risk model gives those points
approximately 0.03 failure probability and has no pass/fail edge in that
context; UCB never queries them. The remaining S01 regression is found by
UCB and static risk. This analysis was made **after** confirmation and is a
diagnostic for any later method revision, not a retroactive change to this
result. The complete truth, query sequence, 4/8-neighbor region coverage,
paired statistics, and figures are in `ucb_evaluation/`.
