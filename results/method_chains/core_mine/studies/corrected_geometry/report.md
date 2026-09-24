# Sparse-risk, multi-SUT CoRe-Mine validation

Eligible policies (idm_mobil, mcts_cv, ppo_ece) are evaluated as leave-one-SUT-out targets. Each seed has 320 physically executed candidate scenarios across five functions, with an outcome-independent continuous proposal. Primary budget B=50.

| Method | CVS@50 | Collision@50 | Severity@50 | Critical@50 |
|---|---:|---:|---:|---:|
| FPS-Risk | 17.111 | 28.778 | 35.611 | 42.444 |
| FPS-Severity | 16.611 | 28.000 | 35.389 | 42.778 |
| FPS-Balanced | 15.944 | 25.667 | 31.833 | 38.000 |
| FPS-Marginal | 16.111 | 24.000 | 28.556 | 33.111 |
| MeanResidual-Marginal | 16.444 | 25.778 | 32.111 | 38.444 |
| CoRe-Marginal | 17.278 | 26.222 | 31.667 | 37.111 |
| TargetOnlyGP-Marginal | 10.556 | 12.778 | 17.222 | 21.667 |
| Random | 7.528 | 6.322 | 9.122 | 11.922 |

Mean target critical-event rate is 23.58%. CoRe-Marginal minus MeanResidual-Marginal at CVS@50 is +0.833, paired bootstrap [-0.556, +2.389].

The source/target split is leave-one-SUT-out; all target response vectors remain inside the cache oracle and are scored only after selection.
