# Directed-edge feature confirmation: NL-IDM

The frozen 588-scene manifest completed for all three builds (1764 physical
episodes, no unknown outcomes). This tests the role-gated model that retains
directed edge features inside majority-failure contexts.

| Update | Direction | True changes | Directed edges @20 (area) | Static risk | Coordinate role gated |
|---|---|---:|---:|---:|---:|
| V0→V1 | Regression | 110 | 11 (0.548) | 18 (0.976) | 11 (0.548) |
| V0→V1 | Improvement | 0 | NA | NA | NA |
| V1→V2 | Regression | 0 | NA | NA | NA |
| V1→V2 | Improvement | 12 | 12 (0.486) | 8 (0.610) | 12 (0.486) |

All regressions are in S04 (98) and S08 (12); all improvements are in S08.
The directed feature version exactly ties its coordinate ablation on the two
directions with changes. It misses seven more regressions than static risk,
while finding four more improvements. The evidence does not establish a
general or directed-feature advantage. Full prefix metrics, truth, query
traces, family statistics, and figures accompany this report.
