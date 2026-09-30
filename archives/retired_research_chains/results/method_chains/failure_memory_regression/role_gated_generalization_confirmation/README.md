# Role-gated contextual adaptation: fresh-family confirmation

## Frozen question

The stage-five primary did not retain an advantage on stage-six contexts.
Exploratory replays of both opened banks ranked `coordinate_role_gated` first
in summed early discovery area, narrowly ahead of the previous primary when
pooled descriptively. That result selected a new candidate; it is not
confirmation evidence. This protocol freezes the candidate before measuring
any stage-seven target outcomes.

The primary is `coordinate_role_gated`. It uses parent-only TTC coverage to
seed regression contexts, parent-failure/lead-partner role coverage to seed
improvement contexts, then updates a coordinate posterior and uses target
feedback for UCB/frontier ranking. `directed_role_gated_edges` is included as
an ablation to test whether the directed boundary features add value.

## Frozen design

Both NL and PPO release chains use eight executable scenario families
(S01–S06, S08, S09), three fresh physical contexts per family, and full 11×11
grids. Each chain has 2,904 scenes and requires 8,712 paired physical
episodes across three builds. The overlap audit compares full fixed-context
tuples against earlier manifests. Queries use a 40-call total budget, split
into 20 per direction when both pools are available; all methods see the same
complete response bank and random receives ten replay repeats.

The frozen comparisons include static risk and role coverage, coordinate
residual/UCB controls, the stage-five directed regression bootstrap primary,
and the directed edge-feature role-gated ablation. Family-clustered paired
statistics use scenario family as the unit. The protocol does not promise a
positive result; task-level direction counts, empty pools, costs, and adjusted
statistics will be reported.

## Status

Both manifests contain 2,904 scenes across 8 families and 24 contexts. The
frozen source, manifest, protocol, and checkpoint hash audits passed.
Measurement is paused before evaluation. NL has 4,153 valid cached rows
(2,904 V0, 1,249 V1, 0 V2); PPO has 3,951 (2,904 V0, 1,047 V1, 0 V2).
Neither bank has malformed lines or duplicate build-scene pairs. Target
outcomes have not been compared. Resume each chain with the existing
`bidirectional measure --root ...` command; it will reuse cached episodes.
