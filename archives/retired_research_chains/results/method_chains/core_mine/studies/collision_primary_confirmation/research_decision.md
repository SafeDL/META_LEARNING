# Prospective six-seed collision-primary decision

The protocol was frozen in
`docs/core_mine_collision_primary_confirmation_protocol.md` before any
outcome at seeds 20320901, 20320915, 20321006, 20321020, 20321103, and
20321117. Both historical controllers executed 320 proposed scenarios per
seed at 20 Hz ego control/physics. All six source-only gates passed with
274, 269, 270, 270, 267, and 269 eligible previously safe candidates;
every seed retained all five functional modes. Five methods then
physically and sequentially executed 50 distinct eligible VI/TTC cases
per seed, for **1,500 charged target episodes** with no precomputed target
bank. The audit found valid 50-query ledgers and deterministic agreement
on 422 scenarios repeated between methods. The numeric details and every unit
are in `analysis50.json`.

| Method | Ego collisions@50 | New collision-or-near-miss cases@50 | Collision grids@50 | Failure modes@50 | Early AUC | CVS@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical mean + local GP residual | 15.17 | 35.17 | 9.50 | 4.33 | 0.693 | 14.00 |
| Historical mean + mode-average shift | 12.83 | 35.67 | 8.00 | 4.33 | 0.755 | 12.83 |
| Historical mean, static | 8.67 | 24.17 | 6.67 | 4.33 | 0.559 | 10.75 |
| Within-mode historical quantile, static | 8.83 | 24.67 | 6.67 | 4.33 | 0.562 | 11.00 |
| Target-only local GP residual | **16.83** | 33.17 | **10.67** | 2.17 | 0.628 | 13.67 |

The local GP improves on the simple mode-average shift by a mean **+2.33
ego collisions** per 50, with per-seed differences `[+2,+4,+4,+4,-2,+2]`
and descriptive paired-seed bootstrap interval `[+0.33,+3.67]`. It also
gains +6.33 collisions over the stronger of the two frozen static rankings
per seed. New failure count is not better than ModeShift (mean difference
-0.50). These findings support the limited observation that spatially
local correction changes **severity mix** compared with a constant
per-mode shift in this simulator; they do not establish more total
collision-or-near-miss discoveries.

The predeclared **full component gate fails**: TargetOnly-Residual finds
1.67 *more* ego collisions per 50 than the historical-mean GP on average;
the GP-minus-TargetOnly per-seed differences are `[-6,-1,-2,+5,-3,-3]`,
bootstrap interval `[-4.00,+1.33]`. Target-only also finds more distinct
collision grid cells (10.67 versus 9.50). Thus historical transfer cannot
be claimed to outperform target-only learning for collision finding or
collision-region diversity. The GP does find two more total failures and
spans more functional failure modes (4.33 versus 2.17), but ModeShift
matches it on failure modes and slightly exceeds it on total failures.
No single tested method dominates all scientifically relevant outcomes.

The structural limitation is sharper than the table's five-mode labels
suggest. Across the 1,500 target executions, **all observed ego collisions
occurred in only `cutin_braking` or `fast_intrusion`**, regardless of
method. Static history tested 50 `lead_braking`, 85 `slow_lead_following`,
and 51 `stop_and_go` cases; its within-mode quantile comparator tested
66, 60, and 56 respectively, with zero collisions in those three modes.
Those modes did produce some TTC-defined near misses. Consequently, the
current scenario suite cannot substantiate a claim to discover different
*collision mechanisms* across five modes. The target algorithm's behavior
and the two-vehicle scenario design may both contribute; this experiment
does not causally separate them.

Research decision: retain the source-safe B=50 regression-testing problem
and the honest outcome trade-off, but do **not** claim that the complete
CoRe-Mine composition is validated, that the historical-mean GP is the
best collision finder, or that five functional collision categories are
covered. More seeds with the same VI/TTC target and same scenario grammar
would mostly refine uncertainty around this limited task, not repair its
scope. The next meaningful validation needs a different target controller
and collision-bearing interaction families, with the candidate design
frozen from source-only evidence before any new target testing. A broader
simulation or real release pair is necessary for a publication-grade ADS
regression-testing contribution.
