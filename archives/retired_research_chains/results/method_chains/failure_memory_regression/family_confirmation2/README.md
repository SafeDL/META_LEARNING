# Directed-boundary feature confirmation

This new split tests `directed_role_gated_edges`, which keeps directed
parent-boundary features active inside role-gated contexts. The prior
role-gated implementation suppressed those features whenever it used local
coordinates, making its reported coordinate ablation identical in the very
contexts where target adaptation concentrated.

The NL-IDM and PPO protocols were frozen before target outcomes were measured.
Each contains four scenario families (S01, S02, S04, S08), three new physical
contexts per family, and a 7×7 rule grid: 588 scenes and 1764 physical episodes
per three-build release chain. The physical tuples are disjoint from every
previous frozen bank. Both child directories contain the complete scenario
manifest, method protocol, build lineage, source snapshots, and (for PPO)
checkpoint digests.

The primary is `directed_role_gated_edges`. `coordinate_role_gated` preserves
the same acquisition rule and local-coordinate terms while removing directed
features. `directed_role_gated` is the older feature-gated version. Static
risk, static boundary, coordinate residual, static role coverage, prior
directed margin/frontier, and multi-frontier comparisons are also included;
random uses ten replay seeds. Regression and improvement are separate primary
directions, and both weight updates are reported. This was developed after
all earlier outcomes were known, so this split is a fresh physical-context
confirmation of that revision, not an independent algorithm-development
lineage.

Both banks completed. In NL, directed edges tie the coordinate ablation on
both tasks with changes (11/110 regressions and 12/12 improvements), while
static risk finds 18/110 regressions and 8/12 improvements. In PPO the
directed method again ties the coordinate ablation in recall on every
nonempty task, with lower early area on the first update's regression and
improvement tasks. The edge-preserving revision therefore does not confirm
the directed-feature superiority hypothesis. See the
[`NL report`](nl/confirmation_findings.md) and
[`PPO report`](ppo/confirmation_findings.md).
