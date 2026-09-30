# Context-bootstrap calibrated confirmation

## Frozen hypothesis

The completed stage-two and stage-three banks used score ranking from the
first query in each direction. Stage three found a task-specific gain from
local offset calibration but still missed NL regressions and varied across
PPO transitions. This stage tests whether online calibration improves when
each eligible target context gets an initial parent-only sample before the
posterior ranks the remaining queries.

The primary is `directed_bootstrap_offset_ucb`; its coordinate ablation is
`coordinate_bootstrap_offset_ucb`. Both reuse the stage-three local slope
correction prior (mean 0, variance 0.5) and +1 posterior-standard-deviation
regression score. Before score ranking, each eligible context receives one
query per direction: regression probes use minimum parent TTC; improvement
probes use the latest valid parent collision. Remaining queries use online
posterior ranking without a neighborhood-frontier override. Predeclared
controls include stage-three calibrated UCB, static/role coverage, coordinate
residual, and target-only selectors.

## Physical protocol

Each NL and PPO chain has three interaction families, three new fixed contexts
per family, and complete 11×11 parameter grids (1,089 scenes and 3,267
physical executions per chain). A pre-measurement audit checks exact
family/context tuples against earlier frozen manifests. Candidate, comparison
methods, source, runner, release profiles, registry, PPO checkpoint hashes,
manifest, selector parameters, and budgets are frozen before target outcomes
are measured.

The NL and PPO physical banks and frozen replays are complete. Context
bootstrap improved regression discovery on several transitions but lowered
improvement discovery on two large transitions; it did not yield consistent
bidirectional gains. See [`findings.md`](findings.md). Three related families
support descriptive comparisons only.
