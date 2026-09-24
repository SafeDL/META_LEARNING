# Corrected geometry validation at B=50

All events below use ego collision or physical near miss.

| Method | Critical count | Collision count | Critical modes | CVS | Early AUC |
|---|---:|---:|---:|---:|---:|
| MeanResidual-Marginal | 38.44 | 25.78 | 4.78 | 16.44 | 0.792 |
| CoRe-Marginal | 37.11 | 26.22 | 4.67 | 17.28 | 0.759 |
| FPS-Marginal | 33.11 | 24.00 | 4.67 | 16.11 | 0.679 |
| MeanResidual-Risk | 44.11 | 28.56 | 4.67 | 16.11 | 0.916 |
| CoRe-Risk | 43.67 | 27.00 | 4.56 | 16.06 | 0.906 |
| TargetOnlyGP-Marginal | 21.67 | 12.78 | 3.44 | 10.56 | 0.394 |
| Random | 11.92 | 6.32 | 4.04 | 7.53 | 0.238 |

Paired differences and hierarchical bootstrap intervals are in `analysis50.json`.
