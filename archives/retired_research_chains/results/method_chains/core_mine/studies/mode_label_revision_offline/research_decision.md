# Retrospective mode-label replay across controller revisions

This is a development-only replay over already completed IDM and FVDM
parameter-revision banks. All target outcomes were available to the analysis
before this replay; the selector was nevertheless replayed sequentially and
used only labels revealed by its earlier queries. It is not fresh validation,
and it required no additional simulator episodes. Each unit has a fixed budget
of 50 distinct queries; initial mode-support queries count toward that budget.

The replay has 18 paired seed-by-build units (three seeds and three parameter
mutations in each of two controller families). Its question is narrow: does
updating a source historical-risk ranking with the average observed binary
hazard label within each mode help beyond source-only ranking, within-mode
quantiles, or UCB1 mode allocation?

| Family | Source mean | Mode quantile | Label shift | UCB1 |
|---|---:|---:|---:|---:|
| IDM: new failures@50 | 29.33 | 31.11 | 29.78 | 31.00 |
| FVDM: new failures@50 | 26.33 | 27.00 | 25.22 | 27.11 |
| Both families: new failures@50 | 27.83 | 29.06 | 27.50 | 29.06 |

The label shift is slightly ahead of raw source ranking on IDM (+0.44 failures
per unit on average), but behind it on FVDM (-1.11). It trails both
within-mode quantiles and UCB1 in each family. Paired, seed-cluster bootstrap
95% intervals for label shift minus the baselines include zero in both
families: raw source ranking [-1.33, 1.67] on IDM and [-2.89, 1.00] on FVDM;
mode quantiles [-3.22, 0.22] and [-3.44, 0.44]; UCB1 [-3.00, 0.11] and
[-3.33, 0.11], respectively. These intervals are descriptive of six fixed
seed/build groups, not population guarantees.

Decision: the current mode-label-shift idea is not a robust improvement over
simple alternatives across controller families. The prior fresh-seed result
also failed to show an advantage over UCB1. Do not elevate the present rule to
the paper's core contribution. The credible testing problem remains
budget-limited, source-safe regression testing, but the method claim needs a
different mechanism and a new, frozen validation set before it can be
supported. These simulated parameter mutations do not stand in for deployed
software releases.

Reproduce the retrospective replay with
`conda run -n metadrive python -m method_chains.core_mine.replay_mode_label_revision_offline`.
Full per-unit metrics, queried indices, and paired bootstrap samples are in
`analysis50.json`.
