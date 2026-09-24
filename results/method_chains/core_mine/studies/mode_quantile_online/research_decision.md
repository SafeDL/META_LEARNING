# Independent physical confirmation at B=50

The pre-outcome protocol is
`docs/core_mine_mode_quantile_online_protocol.md`. All three independent
source-only gates passed at seeds 20300905, 20300919, and 20301003. The
candidate pool in each seed contains 64 completed, event-free reference
scenarios per functional mode. No target outcome bank exists for these
seeds. Each static selector's order was computed from source responses and
scenario features alone; its implementation reproduced the exact cached
selection order on an earlier seed before the new target episodes were run.

Three fixed parameter targets and three fixed selectors each spent 50
distinct physical target executions per seed: **1,350 physical target
episodes**. The 699 repeated `(seed, target, scenario)` executions shared
across selectors agreed in event labels and numerical safety responses.
Internal ego control and physics were both 20 Hz; ego did not initiate lane
changes.

| Selector | New failures@50 | Ego collisions@50 | Failure modes@50 | CVS@50 |
|---|---:|---:|---:|---:|
| ModeQuantile-Static | 33.33 | 27.44 | 5.00 | 15.50 |
| HistoryMargin-Static | 29.89 | 25.11 | 5.00 | 14.50 |
| RiskDiverse-Static | 25.33 | 20.78 | 5.00 | 14.28 |

Across nine paired seed/target units, mode quantiles gain **+3.44 new
failures** over raw historical margin (hierarchical bootstrap 95% interval
[2.11, 4.78]), with a positive difference in *every* unit. They also gain
**+2.33 ego collisions** ([1.00, 3.89]) and **+1.00 CVS** ([0.17, 2.22]).
Against the frozen risk-diversity heuristic the gains are +8.00 failures
([6.22, 9.89]) and +6.67 collisions ([4.33, 9.33]). Bootstrap intervals
summarize these three seeds and three related controller mutants, not an
unrestricted population of ADS releases.

This independent on-the-fly run corroborates the cached validation's
efficacy result and resolves its precomputed-target-bank concern. The core
mechanism is source-only ranking: among *previously passing* scenes, compare
TTC safety margins **within** each functional mode and prioritize high
relative risk across modes. It is not a feedback learner, and earlier
feedback/residual variants failed to beat strong static history.

Do not yet present this as publication-ready novelty or general ADS safety.
All versions are seeded IDM parameter changes in one two-vehicle simulator;
the five modes share one road/traffic model. SPECTRE, STRaP, and other prior
regression-prioritization methods operate on richer execution/scenario
attributes; the exact published implementations do not map onto this bank
without redefining inputs. A fair direct prior-work comparison or a broader
cross-family/real-version benchmark is still required to substantiate a
novel-method claim. The prior-art scope audit is
`results/method_chains/core_mine/studies/prior_art/scope.md`.

Raw evidence: `physical_queries.csv`, `summary.json`, `analysis50.json`, and
source bank/candidate/gate files under
`results/method_chains/core_mine/studies/idm_revision_validation/{seed}/`.
