# Cut-in 6D path-conditioned Inner SAC validation

This directory is the reproducible Cut-in-only artifact for the repaired Inner
pipeline.  The model is trained on two train geometries, two validation SUT
references, and three 6D training Logical domains.  The checkpoint manifest
contains `interaction_prior` followed by `context_meta`; Outer is not trained.

The fixed validation query is the legal fast-small-gap corner
`[-0.65, 0.25, 0.75, 0.25, 0.65, -0.25]` in the order
`(gap, SUT speed, relative speed, start progress, start time, path length)`.
The validation report uses validation SUTs, train geometries, validation
Logical domains, seeds 11/22/33, and K=0/1/2/4 paired queries. For K>0, it
infers posterior latent `z` from deterministic, task-local low-discrepancy
support scenes that differ from the fixed query; policy weights remain frozen.
Formal failure and semantic criteria are unchanged.

`gif/` contains 36 stratified K=0/K=4 animations (low/medium/high × left/right
for each validation task), with the amber dashed quintic reference path shown
in both chase and top-down views.  `visualization/inner_sac_training_curve.png`
is the per-stage SAC return curve.
