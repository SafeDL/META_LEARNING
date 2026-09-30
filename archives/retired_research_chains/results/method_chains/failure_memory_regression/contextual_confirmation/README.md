# Context-local posterior UCB confirmation

This is a separate stage-two confirmation after the completed
`core_confirmation` found no consistent superiority for the paper's original
`directed_residual` selector. The primary hypothesis is that the parent-risk
offset needs a context-local residual intercept with weaker shrinkage when
the target version differs from parent predictions.

The predeclared primary is `directed_context_ucb_loose`: one shared
target-failure posterior with fixed parent-risk offset, context-local
coordinate and directed-edge features, context-intercept prior variance 4.0,
and deterministic one-standard-deviation upper-confidence ranking for
regression queries. Improvement queries use the posterior-mean complement.
There are no forced context probes, coverage slots, or neighborhood-frontier
overrides. `coordinate_context_ucb_pure` tests whether directed edges add
value; `directed_context_ucb_pure` tests the prior-variance change. Static
risk, coordinate residual, original directed residual, and target-only are
predeclared controls.

Each chain uses the same 1,089 scenes: three families, three new physical
contexts per family, and complete 11×11 grids. Each chain therefore requires
3,267 physical executions. The context audit found no overlap with earlier
scenario manifests. The selector, runner, loader, registry, NL release
profiles, protocol freezer, source hashes, PPO checkpoint hashes, primary
method, baselines, and budget were frozen before target measurement.

The complete paired NL and PPO physical banks are now evaluated separately by
direction and release transition. The primary tied the coordinate-context UCB
ablation on both nonempty improvement tasks, missed all NL regressions, and
did not beat target-only on nonempty regression tasks. The findings therefore
support context-local adaptation on selected tasks but do not establish broad
superiority or an incremental gain from directed edges; see
[`findings.md`](findings.md). The finite-bank replay and three family clusters
do not justify a population-level superiority claim.
