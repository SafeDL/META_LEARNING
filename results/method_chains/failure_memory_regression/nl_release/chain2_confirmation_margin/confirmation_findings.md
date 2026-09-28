# Fourth frozen NL-IDM bank: finite-bank bidirectional gain

This bank was frozen before target outcomes were read. The 441-case manifest
contains three new physical contexts per S01/S02/S08 family and a 7×7 grid
per context. All 1323 build/scene physical episodes were measured; the
evaluator verified scenario, build, simulator seed, and execution-contract
fingerprints. There are no unknown transitions. `frozen_selector_source.py`
matches the SHA-256 in the pre-outcome `selector_config.json` and the replay
cost ledger. The 12 methods share the same no-repeat, 20+20 query schedule.

| Transition / direction | True changes | Directed margin frontier | Static risk | Ordinary coordinate residual | Static margin coverage | Coordinate margin frontier |
|---|---:|---:|---:|---:|---:|---:|
| nl_v0→nl2_v1 / regression | 3 | **3 / 0.195** | 0 / 0.000 | 0 / 0.000 | 1 / 0.071 | 3 / 0.190 |
| nl2_v1→nl2_v2 / improvement | 17 | **13 / 0.771** | 6 / 0.443 | 7 / 0.410 | 6 / 0.443 | 7 / 0.429 |

Cells give discoveries by 20 direction queries / early-discovery area. The
opposite direction in each transition has zero true changes and is NA. The
earlier directed Laplace method ties 13/17 improvements with area 0.800,
slightly above the revised method's 0.771, but finds 0/3 regressions. The
immediate UCB predecessor finds 1/3 regressions and 13/17 improvements
(areas 0.043 and 0.676). Four-neighbor coverage is 1/1 regression region and
2/2 improvement regions for the revised method; static risk covers 0/1 and
2/2 respectively.

This is the first **independent finite-bank descriptive gain in both
directions** against static risk and ordinary coordinate residual. The
regression discovery gain comes mainly from parent-margin context coverage
and observed-regression frontier search: the coordinate-frontier ablation
also finds 3/3, slightly later. With the same coverage/frontier, directed
features find 13/17 improvements against the coordinate ablation's 7/17.
The result therefore supports the combined acquisition rule and identifies
which component contributes to each direction; it does not show that the
directed feature alone causes the regression gain.

All 3 regressions and all 17 improvements occur in the **S08 family**. The
other two families have no changes. The conservative three-family paired
sign-flip p values and Holm-adjusted values are 1.0, so this bank does **not**
establish statistical or cross-family superiority. The two version updates
also contribute one nonempty direction each; they are not independent
replicates of both directions. Prior failed confirmations and post-hoc
development replays remain in the study record. No PPO weight-inheritance
experiment has yet been run.

See `margin_evaluation/` for the evaluator-only transition truth, exact query
order and selection phase, direction/context summaries, 4/8-neighbor region
coverage, paired statistics, cost ledger, and figures.
