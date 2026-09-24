# Developmental hierarchical residual check at B=50

The frozen mixed-lane confirmation has already been inspected. Its
ModeShift-Risk method beat the local-only GP and original CoRe-Marginal on
actual collision-cell coverage. This **post hoc developmental** check
tests one mechanistic explanation: the transfer error has a mode-wide
component that a local Matérn residual incorrectly lets decay to zero
away from observed target cases. It is not independent validation and no
claim of success can rest on these seeds alone.

Freeze one model before execution. For each mode, use the historical mean
response `s(x)` plus a zero-mean residual GP with covariance
`k(x,x') = 0.35^2 + 0.15^2 * Matern52(x,x'; length=0.30)` within that mode
and zero covariance across modes. Retain observation standard deviation
0.05, response encoding, collision/event thresholds, all physical feature
masks, and deterministic tie-breaking from the previous experiments.
The constant-kernel term models a shared mode-level shift; the Matérn term
models local deviations. The acquisition is **risk-only predicted event
probability**, because the matched marginal rule failed in the frozen
mixed-lane study. No hyperparameter fit or threshold tuning is allowed.

Reuse only the already qualified source banks from the four mixed-lane
seeds, and run one new `HierarchicalResidual-Risk` B=50 campaign for each
of the two target controllers on every seed: 8 x 50 = **400 new physical
target episodes**. Each campaign is sequential and sees only its own
charged observations. Compare post-run to the already completed
ModeShift-Risk, MeanGP-Risk, ModeQuantile-Static, and TargetGP-Risk traces.
Primary endpoint remains actual ego-collision cells in the fixed physical
grid; report ego collision count, collision modes, total new failures,
CVS, early AUC, cost, and exact repeat checks. A fresh-seed confirmation
is justified only if the new model has a positive mean collision-cell
gain over ModeShift on **both** targets, positive differences in at least
six of eight seed-target units, and no mean collision-count regression
against ModeShift. Otherwise retain ModeShift as the simpler explanation
and do not tune a second kernel after seeing these results.
