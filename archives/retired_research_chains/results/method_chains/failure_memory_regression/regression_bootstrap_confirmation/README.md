# Regression-only context-bootstrap confirmation

## Frozen hypothesis

Stage four showed that probing one parent-only sample in each eligible
context improved regression discovery on several transitions, while consuming
too much budget for improvement ranking on two large improvement tasks. This
stage reserves those context probes only for regression queries. Improvement
queries use the calibrated posterior from rank one.

The primary is `directed_regression_bootstrap_ucb`; the coordinate ablation is
`coordinate_regression_bootstrap_ucb`. Both use the stage-three context-local
parent-offset correction prior (mean 0, variance 0.5) and the same deterministic
acquisition rule. Regression probes select minimum parent TTC in each eligible
context, then rank the remaining regression budget by +1 posterior standard
deviation. Improvement selection begins immediately with posterior-mean
complement ranking. No frontier override is used. Controls include full
two-direction bootstrap, stage-three no-bootstrap calibration, role coverage,
static risk, coordinate residual, and target-only.

## Physical protocol

The frozen NL and PPO chains each use three interaction families, three new
fixed contexts per family, and complete 11×11 parameter grids: 1,089 scenes
and 3,267 paired physical executions per chain. Context tuples are audited
against all earlier frozen manifests before measurement. Source snapshots,
PPO checkpoints, candidate parameters, baselines, and budgets are frozen
before target outcomes are measured.

This was a predeclared hypothesis motivated by stage-four direction-specific
results. Both physical banks are complete and audited. The outcome did not
show consistent bidirectional superiority: regression bootstrap helped the
NL regression pool over unbootstrapped calibration, but role-gated and
directed calibrated controls won other transitions. See
[`findings.md`](findings.md) for the full counts, early-area results, and
family-clustered inference.
