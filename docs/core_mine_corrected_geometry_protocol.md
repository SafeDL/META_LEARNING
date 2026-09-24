# Corrected-geometry B=50 protocol (v5)

Status: frozen before the v5 qualification and validation banks are inspected.
The development-only physical pilot used seed 20280901, 16 Sobol scenarios per
mode and three retained controllers. It established that the corrected event
definition yields real ego collisions in multiple modes. Pilot episodes are
excluded from the v5 qualification, tuning and validation statistics.

## Testing question

With a budget of 50 target-controller simulations, can response histories from
other controllers and sequential target feedback discover more genuine safety
events, especially across different functional modes, than matched alternatives?

The claim requires gains against the strongest matched method, not merely
against random choice. We keep CoRe-Mine only if its composition or marginal
coverage component adds independently measured value over the simpler
source-mean residual model or risk-only selection.

## Physical outcome

An ego collision counts if the simulator marks the ego as crashed. Otherwise
near miss means either minimum TTC < 1.5 s or minimum Euclidean distance
between the two rotated vehicle polygons < 1.0 m. Background-only collisions
do not count. The environment checks these conditions at 20 Hz over 7 s.
The true polygon metric replaces the invalid centre-distance proxy of v2-v4.

## Candidate generation and split

Each seed has 64 scrambled four-dimensional Sobol points per mode, 320 total.
The four dimensions are initial gap, relative speed, timing and intensity;
the latter two are omitted from model distance calculations when a mode does
not use them. Ranges (gap in m, relative speed in m/s):

| Mode | Gap | Relative speed |
|---|---:|---:|
| fast_intrusion | 6–25 | −8–0 |
| cutin_braking | 6–25 | −8–0 |
| lead_braking | 8–30 | −7–0 |
| stop_and_go | 5–24 | −8–−1 |
| slow_lead_following | 5–22 | −8–−1 |

The independent qualification seed is 20280909. Development seeds are
20280917 and 20281001. Validation seeds are 20281015, 20281029 and 20281112.
All three SUTs (idm_mobil, mcts_cv, ppo_ece) are evaluated; each serves once
as target, with the other two used as historical sources. No target outcome is
visible to the selector before its scenario is queried. Every method pays for
all 50 target queries, including the first ten used to cover all five modes.
Random is repeated ten times per target unit.

## Method comparison

CoRe-Risk and CoRe-Marginal test whether marginal value helps at fixed
posterior. MeanResidual-Risk and MeanResidual-Marginal test whether historical
composition helps at fixed objective. FPS-Risk, FPS-Marginal,
TargetOnlyGP-Marginal and Random supply additional baselines. The matched
comparisons and ablations share candidate sets, seeds and query budgets.

Development selects among the first six prelisted CoRe configurations in
`sparse_sut_experiment.py` using CVS@50, subject to retaining at least 90% of
the best development SeveritySum@50. Validation uses the frozen winner without
retuning. Earlier B=20 results are supplementary prefixes, not the primary
budget.

## Decision evidence

Report genuine `CriticalCount@50`, `CollisionCount@50`, `SeveritySum@50`,
CVS@50, continuous coverage F@50, and number of modes with a verified event.
The first two describe actual safety outcomes; CVS and F describe redundancy.
Include per-seed and per-target paired comparisons and bootstrap intervals.
Any claimed method increment must hold against the strongest matched
alternative on the relevant outcome without hiding losses in collision count
or another target. A positive proxy score alone is insufficient.

The qualification bank is used only to check that the corrected proposal has
nontrivial events, cross-controller disagreement and more than one affected
mode. A failed gate stops validation generation; it does not authorize
changing the held-out validation bounds after inspecting validation outcomes.
