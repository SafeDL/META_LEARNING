# Frozen B=50 validation of source-safe mode-quantile prioritization

Frozen after the two-seed `ModeQuantile-Static` development result and before
any source or target outcome is inspected for the validation seeds below.
This is a candidate *simple-method* validation, not a pre-established novelty
claim. All previous IDM and local-fault seeds are spent for method selection.

## Testing problem and fixed physical benchmark

Given 320 scenarios completed safely by `idm_ref`, select 50 to replay first
on each of three seeded IDM parameter revisions. Retain all three targets:
0.7 s reaction delay, 3 m/s² max braking, and their combination. Internal
ego longitudinal control and physics are 20 Hz; ego never initiates a lane
change. The simulator, five functional modes, proposal bounds, source-only
candidate construction, scenario seeds, event semantics, and all 50-query
accounting are exactly those in
`docs/core_mine_idm_revision_validation_protocol.md`. Each seed has 128
Sobol proposals per mode; the first 64 completed event-free reference tests
per mode become the 320 candidate pool. Fail the validation if a seed has
fewer than 64 eligible reference cases in any mode; do not replace it.

Independent validation seeds: **20300701, 20300715, 20300729**. Do not inspect
their target banks before selector and analysis code is frozen. All target
builds execute on the same source-only pool, but target results remain behind
the query oracle when ranking each B=50 campaign.

## Selectors and endpoints

`ModeQuantile-Static`: within each mode, rank the historical continuous TTC
response across its 64 candidate scenarios, normalize ranks to [0,1], and
select the highest remaining rank. No target feedback enters the score;
the common first-ten mode-support rule counts inside B=50. Deterministic
ties use candidate index. Compare to `HistoryMargin-Static` (raw source TTC
response), `RiskDiverse-Static` (the frozen 50/50 source-risk/diversity
heuristic), `HistoryMargin-Residual`, `TargetOnly-Residual`, and ten-order
`RandomSafe`. None is described as a faithful SPECTRE implementation.

Primary endpoint: newly failing target scenarios@50, counting ego collision
or corrected physical near miss. Secondary endpoints: ego collisions,
failure modes, CVS, early discovery, full-pool opportunity and recall,
per-target/per-seed results, and paired hierarchical bootstrap intervals.
The candidate claim requires positive mean difference versus **both** raw
history and risk-diversity, with each of the three target builds represented
and no seed/target suppression. A modest positive result only establishes
simulator-specific efficacy; it does not establish publication-level novelty
without direct prior-work adaptation, other SUT families, and physical
sequential confirmation. Negative results remain reported unchanged.
