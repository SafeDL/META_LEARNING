# Sparse-risk, multi-SUT CoRe-Mine validation

Four retained policies (idm_mobil, vi_ttc, mcts_cv, ppo_ece) are evaluated as leave-one-SUT-out targets. Each seed has 500 physically executed candidate scenarios across five functions, with an outcome-independent 90:10 benign:challenge design.

| Method | CVS@20 | Collision@20 | Severity@20 | CVS@50 |
|---|---:|---:|---:|---:|
| FPS-Risk | 3.750 | 8.083 | 14.000 | 5.000 |
| FPS-Severity | 3.792 | 8.667 | 14.167 | 5.000 |
| FPS-Balanced | 3.750 | 7.417 | 13.292 | 5.083 |
| FPS-Marginal | 4.292 | 5.250 | 10.250 | 5.000 |
| MeanResidual-Marginal | 4.208 | 5.583 | 10.708 | 5.167 |
| CoRe-Marginal | 4.292 | 5.167 | 9.875 | 5.083 |
| TargetOnlyGP-Marginal | 3.917 | 3.250 | 8.708 | 4.792 |
| Random | 1.150 | 0.408 | 1.300 | 2.337 |

Mean target critical-event rate is 10.57%. CoRe-Marginal minus FPS-Marginal at CVS@20 is +0.000, paired bootstrap [-0.167, +0.167].

The source/target split is leave-one-SUT-out; all target response vectors remain inside the cache oracle and are scored only after selection.
