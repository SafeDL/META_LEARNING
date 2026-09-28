# Prospective fourth bank: parent-margin / directed-frontier revision

This plan was written after freezing `protocol.json` and the 441-case
`scenario_manifest.jsonl`, **before** reading any target outcome in this bank.
The previous three confirmation results remain available and negative for a
bidirectional superiority claim. The third bank diagnosed regressions in an
all-parent-pass S08 context, where a binary parent edge does not exist. The
new revision uses the already recorded continuous **parent** TTC margin.

The primary method is `directed_margin_frontier`. On regression turns, it
queries the smallest parent `min_ttc` once in each parent-pass context, in
lexicographic frozen context order. Then it ranks the 4-neighbor frontier of
the most recently discovered regression context using the context-specific
directed target-response UCB; if no frontier exists, it uses UCB over the
remaining parent-pass pool. Improvement turns use the same directed Laplace
predictive score as before. Unknown margins sort last; target feedback enters
only after a charged oracle query. All methods share the fixed 20+20
alternating direction schedule, no-repeat rule, target endpoint, and tie rule.

Predeclared primary comparisons are against `static_risk`, `static_boundary`,
`coordinate_residual`, `static_margin_coverage`, and
`coordinate_margin_frontier`, separately for R and I. The last two isolate
the context sweep and ordinary coordinate adaptation from the directed
features. `directed_context_ucb` is the immediately preceding method. The
same nine legacy methods and three margin methods are frozen in the protocol.
The endpoints are discoveries by 20 direction queries and early-discovery
area; zero-change directions are NA. Family-level paired statistics and 4/8
adjacency region coverage are diagnostics, not independent selector repeats.

The `confirmation4` manifest uses three new, nonoverlapping physical contexts
per S01/S02/S08 family, with ego/lead speed and event-time offsets spanning
negative, middle, and positive levels: `(-1.5, -1.25, -0.10)`,
`(0.5, 0.0, 0.10)`, `(1.5, 1.25, 0.0)`. The same 7×7 grid and builds
`nl_v0→nl2_v1→nl2_v2` are used. This is a sequential confirmation of a
revised method, not one preregistered test pooled with earlier banks. The
old-bank replay in `../margin_revision_development.md` is explicitly
post-hoc and cannot supply independent evidence for this revision.

The selector source is snapshotted as `frozen_selector_source.py`, SHA-256
`fbf1ad652b0c4c46571c1bee7fbabfdf65e648e23e748ceca04b53234f813d57`.
