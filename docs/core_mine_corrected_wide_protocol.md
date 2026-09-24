# Corrected-geometry sparse v6 protocol

Frozen before v6 qualification. v5 validation is previous-study design
information and is excluded from all v6 development and validation averages.
The wide-range pilot uses only seed 20281201 and 16 scenarios per mode; it is
excluded from the new independent study.

## Question and outcome

At 50 target-controller tests, does function-specific historical composition
plus local residual updating discover more physically verified, nonredundant
critical scenarios than source-mean residual updating or the closed historical
posterior? A critical event is an ego collision, or (without ego collision)
TTC < 1.5 s or rotated vehicle-polygon clearance < 1.0 m. Background-only
collisions never count. The physical simulator runs at 20 Hz for up to 7 s.

The primary comparison is CoRe-Marginal versus the strongest of
MeanResidual-Marginal and FPS-Marginal on CVS@50, accompanied by true critical
count, ego collision count, and number of affected functional modes. An
increase in CVS without a credible, small loss in true events and collisions
supports only a coverage trade-off, not better safety-event discovery. CoRe-Risk
versus MeanResidual-Risk isolates the composition component with the same
risk-first choice rule. TargetOnlyGP and Random assess the contribution of
historical data. Development may choose among six prelisted configurations;
the frozen configuration is then used once for validation.

## Fixed candidate pool

There are 64 scrambled Sobol candidates in each of five modes, 320 per seed.
Inputs are initial gap, relative speed, timing and intensity. Physical ranges
(metres; metres per second) are:

| Mode | Gap | Relative speed |
|---|---:|---:|
| fast_intrusion | 8–40 | −8–1 |
| cutin_braking | 8–42 | −8–1 |
| lead_braking | 10–42 | −7–1 |
| stop_and_go | 8–42 | −8–1 |
| slow_lead_following | 7–36 | −8–0 |

All four retained controllers are included: idm_mobil, mcts_cv, ppo_ece and
vi_ttc. Each controller is a hidden target in turn and the other three provide
historical responses. Qualification seed: 20281209. Development seeds:
20281217, 20290103. Validation seeds: 20290117, 20290131, 20290214. Random has
ten repetitions per target unit. Each method selects one 50-step trajectory;
the first ten queries used for mode support are charged to that budget.

Qualification checks at least two controllers with source-discordant events
in two modes, no near-identical three-controller group, and no eligible target
with over 75% of events in one initial-gap quartile. Failure stops the frozen
validation campaign; it does not authorize editing v6 bounds after looking at
held-out results. Report per-target paired differences and hierarchical
bootstrap intervals, along with costs, ablations and reproducible bank hashes.
