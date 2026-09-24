# Frozen cross-family FVDM B=50 validation

The 80-scenario FVDM opportunity pilot passed its predeclared gate; this
study is frozen before any source or target outcome at the new seeds below.
It is an independent family-generalization check, not a replacement for
the IDM results. Retain the `SM-Strong-FVDM` reference and **all three**
fixed FVDM mutants from `docs/core_mine_fvdm_revision_pilot_protocol.md`.

Use the exact Sobol proposal code, five functional modes, mode-specific
parameter bounds, source-only first-64-safe-per-mode selection, simulator
seed mapping, event definition, and 20 Hz internal ego control/physics of
`docs/core_mine_idm_revision_validation_protocol.md`. Qualification seed:
**20301121**; validation seeds: **20301205, 20301219, 20310109**. For each,
run 128 proposals per mode on `SM-Strong-FVDM`. If any mode has fewer than
64 completed event-free source scenarios, stop rather than adjust the
source, bounds, or seed. The selected 320 candidate scenarios are fixed
before any target execution.

Every target executes the same 320 source-safe candidates. The selector
may access target outcomes only through 50 distinct charged queries; first
ten mode-support queries count toward B=50. Compare:
`ModeQuantile-Static`, `HistoryMargin-Static`, `RiskDiverse-Static`,
`HistoryMargin-Residual`, `TargetOnly-Residual`, and ten-order `RandomSafe`.
These implementations and hyperparameters are unchanged from the IDM
mode-quantile validation. Do not describe RiskDiverse as faithful SPECTRE.

Primary endpoint: new target failures@50, counting ego collision or TTC /
polygon near miss. Report collisions, failure modes, CVS, early discovery,
full-pool opportunity/recall, every seed/target (including low-opportunity
ones), and paired seed/target bootstrap intervals. A family-generalization
claim requires positive mean differences over **both** raw historical margin
and risk-diversity, and no target build dropped after inspection. Any result
here remains two synthetic controller families in the same two-vehicle road
simulator, not a claim about real software releases or complete published
baseline superiority.
