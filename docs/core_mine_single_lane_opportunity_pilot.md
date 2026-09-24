# Developmental feasibility pilot: lane-constrained longitudinal scenes

The six-seed collision-primary study found ego collisions only in
`cutin_braking` and `fast_intrusion`, despite tested longitudinal cases.
This pilot tests a specific, physically interpretable scenario-design
hypothesis: the two-lane road lets the VI/TTC ego escape a longitudinal
conflict by an immediate lane change. It is **not** an efficacy validation
and its seed must not be included in subsequent confirmation.

Use seed 20330105 and the already defined v8 outcome-blind 5 x 64 Sobol
physical bounds. Keep the two cut-in modes on the original two-lane road.
Put `lead_braking`, `stop_and_go`, and `slow_lead_following` on a straight
one-lane road, with otherwise unchanged lead schedules, collision logic,
TTC/clearance thresholds, and 7 s horizon. This is implemented in separate
environment subclasses so the prior two-lane results are untouched.
Execute IDM/MOBIL and SM-Strong-FVDM historical controllers on all 320
proposals. Source eligibility requires both runs to complete with no ego
collision or near miss. The source-only feasibility gate is at least 50
eligible cases across all five modes.

Only if the source gate passes, take the first 24 eligible proposal indices
in each mode, fixed without target information, and run two target systems
on these 120 cases each: external VI/TTC and internally controlled
`fvdm_delay05_brake3`. External ego decisions, internal profile control,
and physics all update at 20 Hz. Report actual ego collisions and near
misses by mode; never infer target outcomes for the other eligible cases.
This pilot asks whether at least one formerly collision-empty longitudinal
mode becomes collision-bearing for either target while enough historical
safe tests remain. It cannot establish that any test selector is effective.
If promising, freeze **new seeds and a B=50 comparison protocol** before
more target runs. If not, do not tune the single-lane setup repeatedly to
manufacture a positive result; move to a genuinely richer interaction
benchmark.
