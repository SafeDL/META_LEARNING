# Independent context replication: findings

## Protocol and integrity

The stage-six replication tested the stage-five primary selector,
`directed_regression_bootstrap_ucb`, and its frozen controls on fresh contexts.
Each NL and PPO chain used three new contexts from each supported family
(S01, S02, S08), an 11×11 grid per context, and complete paired execution of
all three releases. Each chain contains 1,089 scenes and 3,267 physical
episodes. The frozen audit passed for both manifests, protocols, selector
snapshots, physical runner, build registry, and PPO checkpoints.

The completed response banks contain all expected build-scene pairs. A
post-measure audit returned `cached_episodes=3267` and
`new_physical_episodes=0` for both NL and PPO. Full-bank truth contains 26
regressions and 58 improvements for NL, and 7 regressions and 62 improvements
for PPO. There were no UNKNOWN transitions. The frozen comparison consumed
2,160 logical queries per chain; the extra selector replays described below
reuse these same banks and add no physical episodes.

## Frozen comparison

The primary selector did not show consistent task-level superiority. The
table reports discoveries by the 20-query direction budget (`D@20`) and the
normalized early discovery area. Controls are reported to show where the
primary wins and loses.

| Chain transition | Direction | True changes | Primary D@20 / area | Relevant control D@20 / area |
|---|---:|---:|---:|---|
| NL V0→V1 | Regression | 26 | 12 / 0.481 | Coordinate bootstrap 14 / 0.576; coordinate role-gated 16 / 0.667 |
| NL V0→V1 | Improvement | 2 | 2 / 0.167 | All evaluated methods found both; this pool is saturated |
| NL V1→V2 | Improvement | 56 | 19 / 0.976 | Coordinate role-gated 20 / 1.000; static risk 19 / 0.971 |
| PPO V0→V1 | Improvement | 29 | 3 / 0.143 | Coordinate role-gated 5 / 0.152; directed offset-calibrated UCB 3 / 0.243 |
| PPO V1→V2 | Regression | 7 | 5 / 0.100 | Directed offset-calibrated UCB 6 / 0.181; coordinate role-gated 3 / 0.029 |
| PPO V1→V2 | Improvement | 33 | 4 / 0.276 | Coordinate role-gated 8 / 0.457; directed offset-calibrated UCB 7 / 0.390 |

Across the six nonempty task directions, the primary found 45 changes by
`D@20` and summed early area was 2.143. The coordinate role-gated control
found 54 with area 2.467; coordinate regression bootstrap found 44 with area
2.162; directed offset-calibrated UCB found 38 with area 1.919. These
aggregates are descriptive only: each task receives the same query budget,
but tasks share three scenario families and the NL/PPO release chains.

Every paired family comparison is labeled
`DESCRIPTIVE_ONLY_SMALL_CLUSTER_COUNT`. There are only three family clusters
per SUT, so the exact family sign-flip tests are coarse; all Holm-adjusted
p-values are 1.0. The replication therefore does not establish a
population-level superiority claim. PPO regression favors directed offset
calibration while the primary beats it on the two NL pools; the coordinate
role-gated control has a higher aggregate but loses on PPO V1→V2 regression.

## Post-confirmation exploratory replay

After the frozen comparison was evaluated, all implemented exploratory
selectors were replayed against the same banks into each chain's `exploratory/`
directory. This is candidate-screening evidence, not independent confirmation.
Across the 11 valid stage-five and stage-six task directions, coordinate role
gating had the highest summed area (4.291) and 94 discoveries at `D@20`;
`directed_role_gated_edges` tied its `D@20` total (94) but had lower summed
area (4.200). The stage-five primary plus stage-six replication had 93
discoveries and summed area 4.095. These pooled descriptive totals do not
support a clear advantage for the primary or for the directed edge feature.

The results point to context seeding and role coverage as useful search
controls, while the directed edge features do not add a demonstrated gain
over coordinate role gating. The next method change must be specified from
development evidence and frozen before another fresh confirmation bank. The
stage-five and stage-six banks are now opened and cannot serve as confirmation
for that change.

## Artifacts

- Frozen task tables and repeat-level summaries: `nl/summary_by_direction.csv`,
  `ppo/summary_by_direction.csv`.
- Family-clustered paired comparisons: `nl/paired_family_statistics.json`,
  `ppo/paired_family_statistics.json`.
- Post-confirmation exploratory replays: `nl/exploratory/`, `ppo/exploratory/`.
- Cost and source fingerprints: `nl/cost_ledger.json`, `ppo/cost_ledger.json`,
  and `freeze_audit.json`.
- Task reports and figures: `nl/report.md`, `ppo/report.md`, and each chain's
  `figures/` directory.
