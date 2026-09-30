# Additional-family confirmation protocol

The NL-IDM and PPO child manifests, selectors, source snapshots, and PPO
checkpoint hashes were frozen before target outcomes in this split were
measured. Each chain has 200 scenes: four families (S01, S02, S04, S08), two
new physical contexts per family, and a 5×5 rule grid. All physical tuples
are disjoint from the prior development, validation, and confirmation banks.
Each complete chain requires 600 physical episodes. The 40-query budget is
alternated between 20 regression and 20 improvement turns, with unused quota
transferred when a parent pool is exhausted.

The primary method is `multi_frontier_role`: parent-only TTC context probes,
role-gated parent-failure probes, online target-response updating, and a
frontier over *all* observed-change contexts. `coordinate_multi_frontier`
uses the identical acquisition rule with coordinate features instead of
directed parent-boundary features. Static risk, static boundary, ordinary
coordinate residual, static role coverage, prior directed margin/frontier,
and the previous role-gated variants are additional comparators; random has
ten replay seeds. All share the same bank and query budget. The primary
questions are directional discoveries and early area for both updates,
including zero-change directions, plus the directed-feature ablation.

Method development used earlier three-family banks after their outcomes were
known. The five-family survey was development-only and showed S04 NL
regressions but no PPO flips. These inputs motivated this confirmatory design
and cannot be counted as independent successes. No method, build, checkpoint,
or scenario will be selected from this confirmation's target labels.

Both complete confirmations are now reported. On NL, the primary finds
17/35 first-update regressions but ties static-role and same-acquisition
coordinate comparators; it finds the two later improvements with lower early
area than static risk. On PPO, it finds 5/5 first-update regressions and 4/4
later improvements but misses both first-update improvements, which static
role coverage finds. The coordinate ablation ties it on every PPO task.
The frozen result therefore fails the full bidirectional/directed-feature
superiority hypothesis. See [`NL findings`](nl/confirmation_findings.md) and
[`PPO findings`](ppo/confirmation_findings.md).
