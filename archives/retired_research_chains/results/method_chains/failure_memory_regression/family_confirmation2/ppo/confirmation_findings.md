# Directed-edge feature confirmation: weight-inherited PPO

The frozen 588-scene manifest completed for all three checkpoints (1764
physical episodes, no unknown outcomes). The tested revision retains directed
edge features inside majority-failure contexts.

| Update | Direction | True changes | Directed edges @20 (area) | Static risk | Coordinate role gated |
|---|---|---:|---:|---:|---:|
| V0→V1 | Regression | 7 | 7 (0.162) | 0 | 7 (0.176) |
| V0→V1 | Improvement | 7 | 2 (0.076) | 2 (0.162) | 2 (0.114) |
| V1→V2 | Regression | 4 | 0 (NA) | 0 (NA) | 0 (NA) |
| V1→V2 | Improvement | 2 | 2 (0.176) | 0 (NA) | 2 (0.176) |

The true flips split across S08 regressions, S01 regressions, and S02
improvements. The directed method ties the identical-acquisition coordinate
ablation on every nonempty task; its early area is lower on the first
regression and improvement tasks. It does not establish directed-feature
superiority. The missing second regression and the first-update directions
with zero discoveries are fully reported, not converted into positive wins.
Full prefix metrics, truth, query traces, family statistics, and figures are
preserved beside this report.
