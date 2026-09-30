# Prospective cross-SUT confirmation

The two child protocols and physical manifests were frozen before measuring any
target outcomes in this split. The same `directed_role_gated` selector and seven
predeclared comparators are applied to the NL-IDM release chain
(`nl_v0 -> nl2_v1 -> nl2_v2`, 441 scenarios) and the independently trained
weight-inheritance PPO chain (`ppo_release_v0 -> ppo_release_v1 ->
ppo_release_v2`, 225 scenarios). Each chain has a complete three-build
response bank. The fixed selection budget is 20 regression and 20 improvement
queries per update, with transferable unused quota. Random has ten replay
seeds; all other methods are deterministic and use the same bank.

The primary comparison is `directed_role_gated` against `static_risk`,
`static_boundary`, and `coordinate_residual`. `static_role_coverage2` removes
the target-response model while retaining the new parent-only probe rule;
`coordinate_role_gated` removes directed edge features while retaining that
rule. `directed_margin_frontier` is the previous frozen revision. We will
report recall and early discovery area separately for both directions and both
updates, including zero-change/negative results, along with family/context
breakdowns. Scene-level replays do not create independent physical samples.

The role-gated rule was developed **post-hoc** on earlier NL and PPO banks.
This new split tests its transfer to unseen physical contexts of the same
three scenario families and fixed builds. It is not a new policy-training
sample, an independent family sample, or evidence of population-level driving
safety. The tuple of family, fixed physical context, and grid rule parameters
has no overlap with any earlier study split.

No method or checkpoint will be changed based on this split. The selector
source, manifests, protocol, build fingerprints, PPO checkpoint digests, and
execution-source digests are frozen in the child directories.

Both banks are complete. The frozen role-gated method found the two NL
regressions and two NL improvements, but the same-acquisition coordinate
ablation tied it and all NL changes remained in S08. On PPO it found 7/15
and 3/4 changes in the first update, and 2/6 improvements in the second;
static role coverage found 14/15 first-update regressions. Thus the
prospective result does not establish overall or directed-feature superiority.
See [`NL findings`](nl/confirmation_findings.md) and
[`PPO findings`](ppo/confirmation_findings.md). Later hybrid and multi-frontier
replays in `posthoc_*` directories are explicitly exploratory and do not
replace these frozen results.
