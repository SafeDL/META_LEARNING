# Development-only cell-aware mode calibration at B=50

This protocol is fixed before executing the new `CellAware-ModeShift`
campaigns. The four mixed-lane seed/target source banks and the original
eleven-method B=50 traces have already been inspected. Therefore this is
**development**, not independent confirmation. Keep the original
`ModeShift-Risk` traces intact and compare paired seed/target units. No
target outcome bank may be consulted by the new selector. Every proposed
target outcome is a newly executed 20 Hz physical episode and counts
inside the same B=50 budget. The first-ten-query mode-support rule and
source-safe eligibility remain identical.

Motivation: the existing ModeShift-Risk selector finds on average 39.5
ego collisions but only 21.125 distinct collision-bearing cells per 50
queries. Repeated collisions within one physical cell have no value for
the predeclared collision-cell endpoint. A mild occupancy penalty might
trade some redundant collisions for new vulnerable regions without
discarding the historical margin and online mode correction.

For each eligible candidate x, start with the *unchanged* ModeShift score
`s_t(x) = historical_mean_response(x) + mean_observed_target_residual(mode(x))`.
Use the already frozen 4 x 4 `(initial_gap, relative_speed)` cells within
each mode, with boundaries determined by physical scenario bounds. Let
`v_t(c)` be the maximum **observed** severity in cell c among previously
charged target queries: 1 for ego collision, 0.5 for near miss, 0
otherwise. The only new selector is

`a_t(x) = s_t(x) - 0.25 * v_t(cell(x))`.

The fixed 0.25 penalty equals the full range of the TTC-only component
of the common response encoding. It is not tuned on the target results.
Use the same deterministic tie-breaking and 50 unique eligible queries.

Primary developmental pass gate: the new method must improve mean
`CollisionCells@50` over ModeShift-Risk on **each** target, improve by at
least 1.0 cell per seed-target unit overall, and retain at least 90% of
ModeShift's mean ego-collision count. A descriptive four-seed cluster
bootstrap interval is reported, not used as a significance claim. Also
report collisions/near misses, mode coverage, 3 x 3 and 5 x 5 cell-grid
sensitivity, and exact repeated-scenario consistency against the original
traces. Do not change the penalty, grid, or endpoints after looking at
these outcomes. If the gate fails, stop this cell-penalty line. If it
passes, freeze *fresh seeds and a new source-only gate* before any new
confirmation target run; compare at minimum ModeShift-Risk, static
source ranking, target-only learning, random selection, and this method.

Even a confirmed gain would support only fixed-suite collision-region
discovery in this synthetic Highway-env grammar. Cells are not software
defects, and this test cannot establish road safety or original CoRe
composition benefit.
