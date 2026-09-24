# Prospective collision-primary confirmation, B=50, 20 Hz

This protocol is frozen **after** the developmental simple-residual
ablation, and **before** any source or target outcome for these new seeds.
The earlier studies used collision-or-near-miss count as primary. Their
result that a mode-average shift matches the GP on that endpoint is
retained. Here the predeclared question is narrower and more safety-
interpretable: does the historical-mean plus mode-local GP residual find
more **ego collisions** among previously safe replay cases within 50 target
tests than simpler correction, static history, and target-only feedback?

The six fresh seeds are **20320901, 20320915, 20321006, 20321020,
20321103, 20321117**. On each, generate the unchanged five-mode x 64
scrambled-Sobol scenario proposal. Execute historical IDM/MOBIL and
SM-Strong-FVDM controllers on all 320 proposals before querying VI/TTC.
Both historical ego controllers and physics loops operate at 20 Hz. The
target's high-level decision and physics cadence are also 20 Hz. Eligibility
requires both historical episodes to complete with no ego collision or
near miss. The source-only gate is >=50 eligible cases spanning all five
modes in **every** seed. If any gate fails, stop; do not replace seeds or
modify bounds.

Compare exactly five frozen methods, each with 50 distinct sequentially
executed eligible target queries: `HistoryMargin-Residual` (mode-local
Matern-5/2 GP, unchanged frozen hyperparameters),
`HistoryMargin-ModeShift` (the exact constant within-mode correction in
`docs/core_mine_simple_residual_ablation_protocol.md`),
`HistoryMargin-Static`, `ModeQuantile-Static`, and
`TargetOnly-Residual`. The same first-ten-query mode-support rule applies
to all five and counts within B=50. No target bank, shared target outcomes,
free calibration calls, or method-dependent safety gates are allowed.

The **primary endpoint** is target ego collisions found at B=50 on cases
safe under both sources. The primary paired comparisons are GP minus
ModeShift, GP minus the stronger frozen static ranking in each seed, and
GP minus TargetOnly. Report each seed, the mean differences, and a
10,000-draw paired seed-bootstrap percentile interval. The component
confirmation gate requires GP minus ModeShift collision mean >0, the
bootstrap 95% lower endpoint >0, GP minus each of the other two baselines
collision mean >0, and no mean regression of more than two in the
collision-or-near-miss count versus ModeShift. Secondary endpoints are
new failure count, independent-grid CVS, collision-grid count, functional
failure modes, early AUC, target execution cost, and selector decision
cost. Report all results regardless of gate outcome. Repeated scenarios
between methods must agree exactly on event labels and numerical safety
responses. Do not change method, seed, threshold, or endpoint after target
inspection.

This is still one synthetic two-vehicle road model and one VI/TTC target
algorithm. A positive gate would validate only a limited collision-focused
method-component effect, not broad ADS release generality, GP novelty by
itself, or safety certification. A negative gate rejects the GP-specific
claim and favors the simpler mode-shift explanation.
