# Sparse-risk, multi-SUT CoRe-Mine validation

> **Archived proxy-label result; withdrawn as safety evidence (2026-09-23).**
> The validation bank contains zero ego collisions and zero near misses with
> TTC below 1.5 s. All 345 archived near-miss flags came from a geometrically
> incorrect distance proxy. A B=50 replay of all 176 MeanResidual-Risk hits
> found zero corrected events. See `label_audit.json`, `geometry_audit.json`,
> and `research_decision.md` before interpreting the table below.

Eligible policies (idm_mobil, mcts_cv, ppo_ece) are evaluated as leave-one-SUT-out targets. Each seed has 500 physically executed candidate scenarios across five functions, with an outcome-independent continuous proposal.

| Method | CVS@20 | Collision@20 | Severity@20 | CVS@50 |
|---|---:|---:|---:|---:|
| FPS-Risk | 2.889 | 0.000 | 3.944 | 4.389 |
| FPS-Severity | 2.889 | 0.000 | 3.944 | 4.389 |
| FPS-Balanced | 2.222 | 0.000 | 2.611 | 3.444 |
| FPS-Marginal | 2.722 | 0.000 | 3.556 | 4.333 |
| MeanResidual-Marginal | 3.000 | 0.000 | 4.000 | 5.500 |
| CoRe-Marginal | 2.667 | 0.000 | 3.222 | 4.000 |
| TargetOnlyGP-Marginal | 1.056 | 0.000 | 1.056 | 2.111 |
| Random | 0.694 | 0.000 | 0.717 | 1.633 |

Mean target critical-event rate is 7.67%. CoRe-Marginal minus MeanResidual-Marginal at CVS@20 is -0.333, paired bootstrap [-0.722, -0.056].

The source/target split is leave-one-SUT-out; all target response vectors remain inside the cache oracle and are scored only after selection.
