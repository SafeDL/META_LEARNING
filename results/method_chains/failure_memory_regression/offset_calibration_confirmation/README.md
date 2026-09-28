# Context-local parent-offset calibration confirmation

## Frozen hypothesis

Stage two found that context-local UCB can help selected improvement tasks,
but its fixed parent-risk offset did not beat target-only on the nonempty
regression tasks. This confirmation tests whether queried target labels can
correct that transfer error by estimating a context-local slope adjustment to
the parent-risk logit. The correction is centered at zero, with prior variance
0.5, and is learned online from queried labels only.

The primary method is `directed_offset_calibrated_ucb`. It uses the same shared
target posterior, context-local coordinate and directed-edge features,
deterministic +1 posterior-standard-deviation regression score, and
posterior-mean improvement score as the coordinate ablation
`coordinate_offset_calibrated_ucb`. There are no forced context probes or
frontier overrides. The frozen comparison also includes both stage-two
contextual UCB variants, original directed residual, coordinate residual,
static risk, and target-only.

## Physical protocol

Each NL and PPO release chain uses three interaction families, three new fixed
contexts per family, and complete 11×11 parameter grids: 1,089 scenes and
3,267 physical executions per chain. A pre-measurement audit checks exact
family/context tuples against every earlier frozen scenario manifest. Source,
runner, release profiles, registry, PPO checkpoint, manifest, selector
parameters, baselines, and query budget are snapshotted before target outcomes
are measured.

The NL and PPO physical measurements and frozen replays are complete. The
primary did not beat its coordinate calibration ablation or the simple
baselines consistently across tasks, and all adjusted family-comparison
p-values are 1.0. See [`findings.md`](findings.md) for results and the next
development hypothesis. With three related families, the paired summaries
remain descriptive and cannot establish population-level superiority.
