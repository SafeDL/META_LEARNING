# Frozen cross-family FVDM revision-opportunity pilot

The positive mode-quantile results to date use only same-family IDM
parameter mutants. Before any FVDM target outcome is seen, freeze this
separate 20 Hz opportunity pilot. It is **not** an efficacy validation, and
favorable pilot results cannot be merged into the IDM confirmation average.

Use the existing `SM-Strong-FVDM` profile from
`sut_algorithms/highway_env/idm_profiles.py` as the reference: max braking
6 m/s², desired gap 8 m, comfort acceleration 2.5 m/s², FVDM sensitivity
0.45, velocity gain 1.0, and transition gap 8 m. Freeze three same-family
parameter mutants with all unmentioned settings unchanged:

1. `fvdm_delay05`: reaction delay 0.5 s, max braking 6 m/s².
2. `fvdm_brake3`: reaction delay 0 s, max braking 3 m/s².
3. `fvdm_delay05_brake3`: both changes.

Execute all four builds on the **same fixed 80 scenarios** from seed
20291202 and the same scenario-index simulator seeds as the prior IDM pilot.
No new scenario proposal, target-dependent filtering, or hyperparameter
adjustment is allowed. Source-safe means completed without ego collision or
physical near miss (TTC < 1.5 s or polygon clearance < 1 m). Report the
source-safe count in every mode, plus all three mutants' new events,
collisions, and failure modes inside that exact source-safe subset. Physics
and ego longitudinal control run at 20 Hz; ego never initiates a lane change.

Progression gate: at least 8 source-safe scenarios in *each* of five modes,
and at least two target mutants each with at least 2 new failures spanning
at least 2 modes. If this fails, do not tune the FVDM profiles, search new
pilot seeds, or claim cross-family generality. If it passes, freeze a new
candidate-bank study before seeing any additional target result, compare
`ModeQuantile-Static` with raw source margin and risk-diversity at B=50, and
report every mutant including low-opportunity builds.
