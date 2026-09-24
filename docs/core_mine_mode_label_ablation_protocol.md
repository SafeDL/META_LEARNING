# Development-only ablation: continuous target response versus event labels

Frozen before executing the new ablation campaign on 2026-09-24. This is
**development on already inspected fresh-confirmation seeds**, not another
independent confirmation. Preserve every existing B=50 trace and report
unfavorable outcomes.

## Question

Does the continuous target TTC term in `ModeShift-Risk` materially improve
collision-region discovery, or is its benefit explained by coarse
per-mode event-label feedback and continuous *historical* ranking?

## Only changed component

Reuse the four seeds and both targets of
`docs/core_mine_mode_shift_fresh_confirmation_protocol.md`, including their
frozen source banks, source-safe eligibility, 20 Hz dynamics/control,
five-mode grammar, and first-ten-query mode-support rule. Execute 50 new,
distinct physical target episodes for each seed-target unit (400 total).
No target outcome is looked up before its charged query.

The new `ModeLabelShift-Risk` starts with the same mean historical response
`source_y(x)` as `ModeShift-Risk`. Within each mode, update its additive
offset with the mean of

```
label_response(executed target) - source_y(executed scenario),
label_response = 0.5 * event + 0.5 * ego_collision.
```

Thus the sole removed observation component is the target's
`0.25 * exp(-TTC/3)` term. Collision and near-miss labels, historical
continuous TTC margins, matching selected-case source subtraction,
candidate set, and selection policy remain the same. This is an ablation,
not a distinct proposed algorithm. `ModeShift-Risk`,
`ModeQuantile-Static`, and `SourceStatic-Marginal` are fixed comparators
from the earlier campaign; their target executions are not rerun.

Primary endpoint is the same 4 x 4 collision-bearing physical cells@50;
also report ego collisions, new collision-or-near-miss events, 3 x 3 and
5 x 5 cells, and per-mode query allocation. Verify repeated-scenario
outcomes against the stored charged comparators. An explanation that
target continuous feedback is necessary requires `ModeShift-Risk` to
exceed `ModeLabelShift-Risk` on mean primary endpoint in both targets,
without fewer mean ego collisions. Otherwise reject that explanation.
Because seeds are reused, even a positive contrast is developmental
only and requires new held-out confirmation.
