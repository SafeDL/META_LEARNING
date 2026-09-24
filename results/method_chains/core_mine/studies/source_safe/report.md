# Historical blind-spot testing at B=50

Primary targets were frozen as MCTS and VI/TTC; all four targets are in `analysis50.json`.

| Method | New failures@50 | Ego collisions@50 | Failure modes | Early AUC | CVS@50 |
|---|---:|---:|---:|---:|---:|
| HistoryMargin-Residual | 22.33 | 9.33 | 2.50 | 0.491 | 8.50 |
| HistoryMargin-Static | 13.33 | 4.83 | 2.50 | 0.313 | 5.83 |
| TargetOnly-Residual | 18.17 | 7.83 | 2.00 | 0.342 | 7.25 |
| CoRe-Residual | 21.83 | 9.33 | 2.50 | 0.497 | 8.33 |
| RandomSafe | 5.32 | 2.47 | 1.82 | 0.110 | 3.23 |

Paired hierarchical bootstrap intervals and every target unit are in `analysis50.json`.
