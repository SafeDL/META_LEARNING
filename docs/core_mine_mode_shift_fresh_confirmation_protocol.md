# Fresh B=50 confirmation of simple historical mode calibration

Frozen on 2026-09-24 **before source or target outcomes** for seeds
20330703, 20330717, 20330731, and 20330814. These seeds are disjoint
from all development and earlier formal runs. Do not replace a seed,
retune scenario bounds, or change methods after source qualification.

The purpose is deliberately narrower than validating original CoRe-Mine:
test whether the best *development-identified* simple method,
`ModeShift-Risk`, reproducibly improves fixed-suite regression test
prioritization at B=50. This fresh test cannot rescue the failed
composition or marginal-coverage components.

Use the unchanged v8 scrambled-Sobol proposal: 64 scenarios per mode,
five modes, 320 candidates per seed. `fast_intrusion` and
`cutin_braking` retain two lanes; `lead_braking`, `stop_and_go`, and
`slow_lead_following` use one lane. Both historical controllers
(`idm_mobil`, `fvdm_ref`) execute all 320 cases per seed before any
target run. Eligible means both historical controllers completed with
neither ego collision nor near miss. The source-only gate is at least
50 eligible overall and at least 24 in every mode in every seed. If it
fails, stop without changing the seeds. Targets are the unchanged
external `vi_ttc` and internal `fvdm_delay05_brake3`. Ego decisions and
physics run at 20 Hz; an internal outer environment call spans four
0.05 s updates. Outcomes are never precomputed.

Each selector sequentially executes exactly 50 **distinct** eligible
target scenarios, with the common first-ten-query mode-support rule
charged inside B=50. Freeze these seven selectors:

1. `ModeShift-Risk`: source mean response plus the average observed
   target-minus-source residual within each mode; rank the corrected
   response. This is the candidate method.
2. `ModeQuantile-Static`: strong within-mode source-margin percentile.
3. `SourceStatic-Marginal`: source predictions plus archive-aware
   marginal selection, without feedback correction.
4. `TargetGP-Marginal`: target-only GP with the identical marginal rule.
5. `MeanGP-Marginal`: historical mean plus local GP and marginal rule.
6. `CoReGP-Marginal`: original composition, local GP, marginal rule.
7. `RandomSafe`: deterministic random eligible order.

Primary endpoint: distinct actually executed ego-collision-bearing
4 x 4 `(initial_gap, relative_speed)` cells within modes at B=50, with
fixed source-blind physical bounds. Retain actual ego-collision count,
collision modes, total new collision-or-near-miss count, CVS, early AUC,
and 3 x 3 / 5 x 5 cell sensitivity. Never call cells distinct software
bugs. Verify deterministic repeated-scenario outcomes across methods.

An effectiveness confirmation requires ModeShift's mean primary endpoint
to exceed **all six** comparators, to be positive against the strongest
static and target-only comparators within **both** targets, and to have
a positive four-seed cluster-bootstrap 95% lower endpoint against those
two. Its mean ego-collision count must be at least 90% of the strongest
comparator. For robustness, its mean 3 x 3 and 5 x 5 collision-cell
counts must not be below both the static and target-only comparators.
These gates are descriptive within this simulator, not population or
road-safety guarantees. Report all methods and targets even if a gate
fails. No post-hoc change to the primary endpoint or exclusion of an
unfavorable target is allowed.

A positive result would establish a conditional empirical advantage
for simple mode-level calibration in this synthetic mixed-lane replay
benchmark. It would *not* establish that the original full CoRe-Mine
method is superior, novel relative to all prior art, or validated for
real software-release pairs. Those claims require separate evidence.
