# v7 corrected-geometry, balanced-speed protocol

Frozen before v7 qualification. Pilot seed 20290317, all v5 validation seeds,
and the failed v6 qualification seed are development history, not v7 evidence.

The question is whether CoRe-Mine's function-specific historical composition
and local residual improve discovery of genuinely dangerous, nonredundant
scenarios in 50 target-controller tests. The event rule is ego collision or,
without ego collision, TTC < 1.5 s or rotated vehicle-polygon clearance < 1 m.
Only executed target tests count. Source responses are historical information;
target labels remain hidden until queried.

There are five modes with 64 scrambled Sobol candidates each. Per-mode
initial-gap (m) and relative-speed (m/s) bounds are:

| Mode | Gap | Relative speed |
|---|---:|---:|
| fast_intrusion | 6–25 | −8–8 |
| cutin_braking | 6–25 | −8–8 |
| lead_braking | 8–30 | −7–8 |
| stop_and_go | 5–24 | −8–8 |
| slow_lead_following | 5–22 | −8–7 |

Timing and intensity are continuous in [0.05,0.95]. SUTs are idm_mobil,
mcts_cv, ppo_ece and vi_ttc; each is target once per seed, with the other
three as sources. Qualification seed 20290325 is disjoint from development
seeds 20290408 and 20290422 and validation seeds 20290506, 20290520 and
20290603. Random repeats ten times per unit. All methods receive 50 target
queries; the first ten cover five modes and are charged.

The independent qualification checks source-discordant events in multiple
modes for at least two targets, no redundant SUT triplet, and no eligible
target with over 75% of events in one gap quartile. Failure stops the formal
validation. Development tunes CoRe-Marginal among the six prespecified
configurations in `sparse_sut_experiment.py` using CVS@50 while retaining at
least 90% of the best development severity sum. The winner is frozen before
the three validation seeds are analyzed.

Primary matched comparison: CoRe-Marginal versus the stronger of
MeanResidual-Marginal and FPS-Marginal at CVS@50, with actual critical and
collision counts reported alongside it. CoRe-Risk versus MeanResidual-Risk
isolates historical composition under a common risk choice. TargetOnlyGP and
Random test the value of sources. Report paired differences by seed and
target, hierarchical bootstrap intervals, event-mode coverage, early event
discovery, costs and ablations. A small CVS gain alone is insufficient for a
claim of improved danger discovery.
