# Frozen source-safe mode-quantile validation at B=50

The pre-outcome protocol is
`docs/core_mine_mode_quantile_validation_protocol.md`. Three new independent
seeds (20300701, 20300715, 20300729) passed the reference-only eligibility
gate with at least 121 completed event-free proposals in each mode before the
first 64 per mode were fixed. All three predeclared same-family IDM parameter
targets were kept. The simulator ran physics and ego longitudinal control at
20 Hz, with no ego-initiated lane changes. Full target outcome banks were
physically built for matched cached evaluation, but hidden from selectors
until each of their 50 distinct logical queries.

| Selector | New target failures@50 | Ego collisions@50 | Failure modes@50 | CVS@50 |
|---|---:|---:|---:|---:|
| ModeQuantile-Static | 33.00 | 26.56 | 5.00 | 15.11 |
| HistoryMargin-Static | 30.33 | 24.56 | 5.00 | 14.11 |
| RiskDiverse-Static | 26.89 | 21.00 | 5.00 | 14.33 |
| HistoryMargin-Residual | 27.33 | 20.89 | 5.00 | 13.33 |
| TargetOnly-Residual | 22.00 | 16.33 | 3.11 | 10.44 |
| RandomSafe (ten orders) | 7.32 | 5.14 | 3.62 | 5.41 |

The proposed simplest selector improves new failures over raw historical
margin by **+2.67 per 50** on paired seed/target units (hierarchical
bootstrap 95% interval [0.78, 4.44]) and ego collisions by **+2.00**
([0.56, 3.78]). It improves new failures over the frozen risk-diversity
heuristic by **+6.11** ([2.67, 9.67]) and ego collisions by **+5.56**
([2.22, 8.89]). Its CVS difference versus raw margin is +1.00, but that
interval [-0.11, 1.83] includes zero; do not claim proven coverage gain.
Eight of nine seed/target units gain new failures over raw margin; one ties.
All three target-build means gain; no version was excluded.

The test problem is narrow but concrete: when all old-version candidate
scenarios passed, replay the most concerning **relative** safety margins in
each functional class first, not the globally smallest TTC alone. The
method uses no new-controller feedback; past attempts to learn continuous
residuals and mode failure rates did not improve on this simple ranking.
This is a *simulator-specific efficacy result*, not a general ADS safety or
mathematical novelty claim. A direct published-method comparison is still
incomplete because SPECTRE's four execution attributes cannot be copied
faithfully from this longitudinal two-vehicle bank. The source/target
interface mismatch is recorded in
`results/method_chains/core_mine/studies/prior_art/scope.md`. A separate frozen
on-the-fly B=50 physical confirmation is being run; its results must be
reported regardless of outcome.

Raw evidence: `analysis50.json`, `records.csv`, and each seed's
`source_bank.npz`, `candidate_pool.npz`, `target_bank.npz`, and `gate.json`
under `results/method_chains/core_mine/studies/idm_revision_validation/`.
