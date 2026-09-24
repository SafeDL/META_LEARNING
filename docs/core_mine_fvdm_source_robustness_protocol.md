# Frozen clean-source heterogeneous transfer robustness, B=50

This protocol is frozen after the positive three-seed IDM/MCTS→VI/TTC
replication but **before** any target outcome at the seeds below. It tests
whether that result depends on the simplified MCTS-CV historical source's
rollout time-scale mismatch. It is an independent robustness study, not a
replacement for any negative or positive prior result.

Use v8's unchanged 320 Sobol scenarios (five functional modes × 64), at new
seeds **20320405, 20320419, 20320503**. The historical sources are
`idm_mobil` in `ExternalCutInEnv` at 20 Hz external decisions/physics and
the frozen `SM-Strong-FVDM` profile in `CutInEnv` at 20 Hz internal ego
control/physics. These two environments inherit the same cut-in scenario
schedule and physical failure thresholds; source records include ego and
background collision, near miss, minimum TTC, clearance, and completion.
Run both sources on all 320 proposals for every seed before touching the
VI/TTC target. The eligibility gate is at least 50 completed, source-event-
free candidates spanning all five modes in **each** seed. If any source-only
gate fails, stop without seed or parameter changes.

After all gates pass, run exactly the same six frozen B=50 selectors and
hyperparameters as `docs/core_mine_heterogeneous20_replication_protocol.md`:
CoRe-Residual, HistoryMargin-Residual, HistoryMargin-Static,
ModeQuantile-Static, TargetOnly-Residual, and one seeded RandomSafe order.
The target is unchanged `vi_ttc` at 20 Hz high-level decision/physics.
Every selector physically executes its own 50 distinct eligible scenarios
sequentially; no precomputed target bank or sharing of target outcomes across
selectors. The first ten mode-support queries count within B=50.

Primary endpoint and comparison: new VI/TTC ego collision or near miss
found by B=50. Report CoRe minus the stronger frozen static selector per
seed, CoRe minus TargetOnly-Residual, and CoRe minus HistoryMargin-
Residual, each with per-seed differences and descriptive seed bootstrap
intervals. Also report ego collisions, functional failure modes, early AUC,
independent-grid CVS, every seed/method trace, cross-method replay
consistency, and execution/decision cost. A source-robustness claim requires
positive mean differences for all three primary comparisons and no material
collision decrease; failing any of these leaves the original result
conditional on its historical source pair. Do not retune after inspection.

`SM-Strong-FVDM` is a synthetic profile, not a real software release, and
the target is a distinct decision algorithm. Even a positive result remains
one two-vehicle simulator and one target family, not general ADS safety or
publication-grade superiority over published methods.
