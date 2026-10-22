# Collision discovery results

Completed paired runs: 10880.

Main condition: new coordinates. Outer statistical unit: implementation or training lineage.

| Method | Unique collisions | Discovery AUC | Behavior coverage | Version-new | History-specific | RMST queries | Censored runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| fsbo | 36.056 | 20.741 | 6.412 | 6.319 | 1.163 | 43.237 | 45 |
| gp_collision | 29.637 | 16.477 | 6.487 | 4.944 | 1.281 | 43.725 | 45 |
| history_ranking | 21.575 | 11.433 | 5.350 | 4.081 | 0.100 | 47.825 | 40 |
| history_response | 33.044 | 19.536 | 6.756 | 4.894 | 0.913 | 42.288 | 45 |
| nearest_residual | 31.113 | 18.423 | 6.319 | 4.588 | 0.706 | 42.969 | 45 |
| pooled_ft | 33.275 | 19.872 | 6.200 | 5.606 | 0.587 | 39.875 | 40 |
| random | 11.238 | 5.656 | 4.419 | 2.812 | 0.731 | 48.763 | 48 |
| space_filling | 13.869 | 7.473 | 5.100 | 2.819 | 0.694 | 44.288 | 47 |

First-discovery RMST uses the common 128-query window; lower values mean earlier discovery. Runs without a collision remain censored.

Paired lineage bootstrap intervals, censored first-discovery curves, equal-count behavior coverage, strict family holdouts, source replacements, mechanisms and separately charged costs are in [report.json](report.json).

The registered HighwayEnv scope and its three candidate conditions are reported separately.
