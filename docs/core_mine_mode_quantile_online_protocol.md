# Frozen independent physical B=50 confirmation

The cached 3-seed mode-quantile validation was positive, but the complete
target banks were physically precomputed for benchmarking. This follow-up
is frozen before any target outcome on three new seeds is inspected:
**20300905, 20300919, 20301003**. Keep all three IDM parameter targets:
`idm_delay07`, `idm_brake3`, and `idm_delay07_brake3`.

For each seed, execute the same 640 Sobol proposals on `idm_ref` at 20 Hz
internal ego control and physics. In original proposal order, take the first
64 completed, event-free scenarios in each of the five modes. If any mode
has fewer than 64, stop without replacing the seed. Only source-side
responses and candidate geometry may be used to derive each selection order.

Compare exactly three frozen, non-adaptive selectors:
`ModeQuantile-Static`, `HistoryMargin-Static`, and `RiskDiverse-Static`, with
identical implementation and deterministic ties to the cached validation.
For every seed, target, and selector, sequentially execute exactly 50 unique
target scenarios in that selector's source-only order. Count repeated
scenarios *again* for another selector's 50-query budget, but verify their
physical outcomes agree across repeats. Do not execute the target on any
unselected scenario and do not precompute a target bank. At most
`3 seeds × 3 targets × 3 selectors × 50 = 1350` physical target episodes.

Primary endpoint: newly failing target scenarios@50 (ego collision or
corrected polygon/TTC near miss) on historically safe candidates. Report ego
collisions, failure modes, early discovery, CVS, per-seed/per-target tables,
actual episode counts, query uniqueness, cross-selector repeat agreement, and
source/candidate hashes. This is independent confirmation of the simple
ranking effect, not real ADS-release validation, deployment safety evidence,
or proof of novelty over the complete published literature.
