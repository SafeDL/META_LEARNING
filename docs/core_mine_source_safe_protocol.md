# v8 historical blind-spot testing protocol

Frozen before the independent qualification seed is inspected. Earlier v5
validation and v6/v7 qualification results are development history; none is
included in v8 effect estimates. The corrected physical event definition is
ego collision or, without collision, TTC < 1.5 s or rotated vehicle-polygon
clearance < 1 m. Background-only collisions do not count.

## Practical testing problem

A newly evaluated controller may fail where every historical controller was
safe. With only 50 executions of that controller, which historically safe
scenarios should be tested first? For each hidden target, the eligible set is
fixed *before* target testing: scenarios with no collision and no near miss
for any of the other three controllers. All compared methods query only this
same eligible set. They may use the historical continuous TTC margin even
though the historical event labels are all safe. Target outcomes are revealed
one at a time; the first ten mode-support queries count toward 50.

## Fixed pool and split

The four controllers are idm_mobil, mcts_cv, ppo_ece and vi_ttc. Each seed has
five modes × 64 scrambled Sobol scenarios. Gap (m) and relative-speed (m/s)
bounds are fast_intrusion 6–25/−8–8, cutin_braking 6–25/−8–8,
lead_braking 8–30/−7–8, stop_and_go 5–24/−8–8, and
slow_lead_following 5–22/−8–7. Timing and intensity vary continuously over
[0.05,0.95]. The independent qualification seed is 20290617; validation
seeds are 20290729, 20290812 and 20290826. No v8 hyperparameter tuning is
performed. The residual GP uses development-fixed length 0.30, amplitude
0.15 and observation noise 0.05.

Qualification requires ≥50 historically safe eligible candidates spanning
all five modes for each target. It also requires at least five source-safe
target failures in MCTS and ten in VI/TTC, each across at least two functional
modes. IDM and PPO remain in the full results even if they have zero such
failures. Qualification failure stops validation; it does not authorize
changing the held-out seeds or eligibility rule.

## Method and matched controls

The proposed simplest method, **HistoryMargin-Residual**, uses the source
mean continuous response (including finite subcritical TTC) as its initial
map, then updates a mode-specific local GP residual after each target test.
The next eligible scenario has the highest predicted target event probability.
The following controls use the identical 50-query budget and eligibility:

- **HistoryMargin-Static:** the same source mean without target residual.
- **TargetOnly-Residual:** the same GP and feedback, with no source map.
- **CoRe-Residual:** separate historical source hypotheses plus residual,
  testing whether composition improves over the simpler source mean.
- **RandomSafe:** ten seeded repetitions, averaged within target unit.

The primary endpoint is the number of newly revealed target failures at B=50
among MCTS and VI/TTC, preselected as opportunity-bearing targets from the
prior development study. Report all four targets as a full-cohort secondary
analysis. Also report ego collisions among the new failures, number of modes
with a new failure, CVS, early discovery, complete-pool headroom, per-seed
paired differences and hierarchical bootstrap intervals. The residual
component is retained only if it improves over the static historical map;
the composition component is retained only if it adds a reproducible gain
over the source-mean residual. Results are simulation evidence, not a claim
about on-road safety.
