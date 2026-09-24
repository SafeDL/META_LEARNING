# Frozen development protocol: mode-calibrated historical safety margin

This protocol is frozen after inspecting the negative 3-seed IDM validation
result, but before executing either development target on the seeds below.
The three old validation seeds (20300109, 20300123, 20300206) are **spent**:
they may motivate the design and remain in the audit, but must not choose
hyperparameters or validate a revised selector.

## Hypothesis and fixed scenario construction

The hypothesis is that the source TTC margin nearly preserves target-failure
ordering *within* each functional mode, while cross-mode calibration of
failure likelihood changes under a controller revision. A raw continuous-
response GP is inappropriate for sparse, discontinuous collision labels.
Preserve within-mode source ranking and learn only a mode-level allocation
correction from charged target feedback. This is a falsifiable hypothesis,
not a novelty or efficacy claim.

Use the same four IDM builds, simulation semantics, scenario bounds, 640
source proposals, first-64-source-safe-per-mode gate, target outcome,
candidate pool size 320, and B=50 as the frozen IDM revision validation.
The development seeds are **20291230** (previously source-only qualification)
and **20300315** (new); run all three target mutants on each. Do not adjust
bounds, drop targets, or redefine events after seeing results. For each
seed/target, every selector sees the same source-safe pool and target labels
only on its 50 distinct charged queries; first ten mode-support queries count.

## Candidate selectors, at most two design variants

1. `MarginModeCal`: source response `s(x)` is min-TTC response in [0,.25].
   Normalize to `z(x)=(s(x)-median(s))/std(s)` using source-only candidates.
   For each mode `m`, fit an additive intercept `u_m` after each target
   observation via binomial logistic MAP:
   `P(E|x)=sigmoid(2*z(x)+u_m)`, minimizing observed binomial loss plus
   `sum_m u_m^2/(2*1.5^2)`. All intercepts initialize at zero. Select the
   highest event probability under the matched ten-query support rule.
   The fixed positive source coefficient guarantees unchanged within-mode
   ordering. The response and binary event label are not leaked for
   unqueried target candidates.
2. `MarginModeRate`: source-only percentile rank within each mode sets the
   candidate order. Track `Beta(1,1)` posterior mode failure rates from
   charged outcomes, and score each remaining scenario as
   `0.75 * global_source_percentile + 0.25 * posterior_mode_rate`.
   This is a simple allocation alternative, not an extra model family.

Compare both with `HistoryMargin-Static`, `HistoryMargin-Residual`,
`TargetOnly-Residual`, `ModeQuantile-Static` (source-only percentile score),
and ten-order `RandomSafe`. Do not replace matched baselines with weaker
ones. Primary metric is new target failures@50; secondary: ego collisions,
early discovery, CVS, failure modes, and complete-pool opportunity/recall.
Preserve all per-target and per-seed results. Select at most one proposed
variant for a later *new* frozen validation, only if it improves failures
over the strongest static comparator in both development seeds averaged
across all three targets, without losing more than 10% ego collisions.
Otherwise retain the negative conclusion and do not run confirmatory targets
for this idea.

If a variant qualifies, freeze it without tuning and use new independent
validation seeds **20300411, 20300425, 20300509** with the same source-only
gate. Its publication claim still requires a faithful direct prior-work
baseline and genuinely sequential physical B=50 confirmation; a cache
campaign alone is insufficient.
