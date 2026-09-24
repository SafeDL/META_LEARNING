# Developmental simple-residual ablation at B=50

This is a **post hoc developmental** experiment on the already inspected
heterogeneous VI/TTC seeds. It cannot be called an independent replication.
The purpose is to test whether the observed gain of historical-mean plus
local Gaussian-process (GP) residual requires the GP, rather than just
online target feedback. No original selector, source bank, or target trace
is modified. Results must be retained even if the new comparators win.

Use the existing 20 Hz source-safe banks from the IDM/MCTS and IDM/FVDM
heterogeneous studies, with their existing three seeds each. Preserve the
same eligible cases, mode-support rule, target VI/TTC implementation, 20 Hz
external decision and physics cadence, and **50 distinct charged target
episodes per method and seed**. No target outcome bank, result sharing, or
free query is permitted. Each new method executes its own target cases
sequentially, and each target observation is revealed only after selection.

Let `s(x)` be the mean continuous response of the two historical sources.
For already queried target cases `i`, let `d_i = y_target(i) - s(i)`. All
corrections use only `i` in the candidate's functional mode:

- `HistoryMargin-ModeShift`: score `s(x) + mean(d_i)`; the shift is zero
  with no queried case in that mode. This tests whether mode-level
  recalibration alone explains the GP gain.
- `HistoryMargin-KernelShift`: score
  `s(x) + sum_i w(x,i)d_i / (1 + sum_i w(x,i))`, where
  `w(x,i) = exp(-||z(x)-z(i)||^2 / (2 * 0.30^2))` on the fixed effective
  normalized physical coordinates. The `1` is a fixed zero-residual
  pseudo-observation. This tests a simple local correction with no fitted
  covariance or uncertainty model.

Both use descending score with the original deterministic `choose`
tie-break and mode-support rule. They get exactly the same target response
encoding as the GP and do not inspect any other campaign's outcome.
Report new ego collision-or-near-miss count at B=50 as primary, plus ego
collisions, early discovery AUC, failure modes, independent-grid CVS,
per-seed paired differences against `HistoryMargin-Residual`, and cross-
method deterministic repeat checks. A GP-specific claim would require a
clear positive advantage over **both** simpler corrections in both source
configurations; this developmental study cannot itself establish that
claim. If either simpler correction matches or beats the GP, simplify the
method narrative before any further validation.
