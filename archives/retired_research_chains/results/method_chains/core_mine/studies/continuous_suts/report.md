# Sparse-risk, multi-SUT CoRe-Mine validation

Four retained policies (idm_mobil, vi_ttc, mcts_cv, ppo_ece) are evaluated as leave-one-SUT-out targets. Each seed has 500 physically executed candidate scenarios across five functions, with an outcome-independent continuous-borderline design.

| Method | CVS@20 | Collision@20 | Severity@20 | CVS@50 |
|---|---:|---:|---:|---:|
| FPS-Risk | 4.292 | 0.083 | 7.042 | 8.708 |
| FPS-Severity | 4.292 | 0.083 | 7.083 | 8.750 |
| FPS-Balanced | 4.125 | 0.083 | 6.167 | 7.625 |
| FPS-Marginal | 5.208 | 0.000 | 5.750 | 10.208 |
| MeanResidual-Marginal | 5.625 | 0.000 | 6.042 | 10.167 |
| CoRe-Marginal | 5.417 | 0.000 | 5.875 | 11.042 |
| TargetOnlyGP-Marginal | 3.708 | 0.000 | 4.000 | 8.375 |
| Random | 3.462 | 0.008 | 3.758 | 7.508 |

Mean target critical-event rate is 36.47%. CoRe-Marginal minus MeanResidual-Marginal at CVS@20 is -0.208, paired bootstrap [-0.750, +0.375].

The source/target split is leave-one-SUT-out; all target response vectors remain inside the cache oracle and are scored only after selection.
