# MVR Framework Pilot Assessment — seed 11

Status: `FRAMEWORK_PILOT_PASS`.

This is a fixed-budget framework integration pilot, not a formal Stage1
performance or transfer claim.  All evidence below was regenerated after the
current control-contract changes; no previous Stage1 checkpoint or report was
used.

## Provenance and scope

- Source: commit `8b666ef769f7c631b447bbb7b02cc05a8b062042`, source-tree hash
  `568116798549c290b43c6e22a54168007dcfd5f95a1a6510f682ed46dd5b5d51`.
  The worktree was intentionally dirty and this fact is embedded in each new
  checkpoint.
- Control contract: the current three-dimensional interaction residual,
  lane-stable native IDM, and scenario contract.
- Training budget: 80 simulator episodes only: 36 Inner, 4 posterior,
  4 Inner latent-calibration, and 36 Outer.  No formal Stage1 or G3 run was
  executed.

## Preflight

`sut_only_diagnostic.json` passed for Merge, Cut-in, and Roundabout.  Every
SUT completed its route, had zero out-of-road events and zero routing-target
lane mismatches, and showed no sustained steering-sign oscillation.  Lateral
RMS errors were 0.000 m, 0.000 m, and 0.161 m respectively (Roundabout
maximum: 0.435 m).

The regenerated Base zero-residual GIFs are in `base_visualization`.
They use the current lane-stable IDM and the same control contract as the
pilot.  The manifests record no adversary traffic violation and complete the
prescribed SUT route.

## Mini pipeline

`manifest.json` records four reloadable checkpoints.  Inner pretraining
collected 7,166 transitions from all 36 train tasks, with 36 requested and
completed optimizer updates.  It covered every train family (12 episodes
each), all nine train geometries, all four IDM profiles (9 episodes each),
both Cut-in candidates, both Merge candidates, all three Roundabout
candidates, and all three traffic-intent options (12 episodes each).  Actor,
critic, and entropy losses were finite; mean absolute interaction residual
was 0.3194 and saturation rate was 0.1347.

## Validation-only evidence

`stage1_validation.json` evaluates g04/fast_small_gap only.  Base,
random-residual, and trained-Inner each ran three deterministic episodes with
valid rate 1.0 and no traffic violations.  The trained residual produced a
different valid interaction outcome from Base; this pilot does not establish
superiority.

`e2e_validation.json` runs three validation tasks × K={0,1} × four episodes
= 24 exactly accounted simulator episodes.  All runs were finite.  The six
framework checks passed: K=0 stays at the prior, K=1 updates the posterior,
the posterior changes the Inner action and Outer proposal, and the MoE router
is active.  Per-episode records retain geometry hash, candidate, initial
state, outer decision, 3-D Inner action, semantic/traffic validity, failure
signature, and seed.

## Next boundary

The framework control and information-flow integration is now verified.  A
formal Stage1 claim still requires the separate fixed-budget training and
acceptance protocol, multiple seeds, and the designated transfer gates.
