# Rebuilt Cut-in Inner experiment

This artifact replaces the invalid Direct/v2 Cut-in outputs.  It trains only
`interaction_prior → context_meta`; Outer is disabled and no test task is
executed.

## Reproducible Logical scenario

For a task, candidate, normalized vector, and episode seed, Cut-in is uniquely
defined by:

1. `initial_gap_m`: adversary longitudinal position minus SUT position.
2. `sut_initial_speed_mps`.
3. `relative_speed_mps`: adversary speed minus SUT speed.
4. `cutin_onset_time_s`: the time at which SAC lateral control becomes legal.

The adversary spawn is `merge_window_start_m - adversary_speed * onset_time`;
the SUT spawn is then `adversary_spawn - initial_gap`.  The corridor is a legal
road constraint, not a fixed Logical conflict point.  `T_lc` is intentionally
absent: SAC retains direct longitudinal and lateral control after onset.

Both vehicles use the same MetaDrive/Bullet vehicle-force configuration and a
bounded action envelope: acceleration <= 3 m/s², braking <= 6 m/s², command
jerk <= 2 m/s³, lateral acceleration <= 3 m/s², and steering-rate <= 1.5/s.

## Training and validation

- Training: Cut-in g01/g02, IDM normal/assertive, and three Logical domains.
- Validation: validation SUT × validation Logical domain, fixed x0,
  K = 0/1/2/4, three fixed simulator seeds.
- `inner_sac_return.png` is the recorded episodic return visualization.
- Each `gif/cutin-g0*/` folder contains the historical dual-view animation for
  shared prior K=0 and adapted K=4.

The fixed-x0 validation found nonzero latent changes for K=1/2/4, but no
post-onset first-action change and zero paired risk delta versus the shared
prior.  This run is therefore a negative adaptation result, not evidence of
an adapted-policy safety-risk improvement.
