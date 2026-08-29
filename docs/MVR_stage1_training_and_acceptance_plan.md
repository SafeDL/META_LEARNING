# Few-shot Inner training and acceptance

## Objective

The first acceptance target is not a universal SAC accident rate. It is whether
few support trajectories let the Inner adversarial controller adapt to an
unseen SUT and Logical Scenario domain on retained map topology.

\[
\tau=(U,F,M,\Theta_F),\qquad
h_\tau=E(F,M,\Theta_F,\mathcal C_F),\qquad
z_\tau=q_\phi(D_\tau^{support};h_\tau).
\]

`z` denotes residual response/vulnerability characteristics not explained by
observable structure; it is not claimed to be an intrinsic SUT label. A task
contains concrete episodes \(c_j=(candidate_j,x_{0,j})\):

\[
\pi_{Inner}(a_t\mid s_t,h_\tau,c_j,z_\tau).
\]

The model never receives SUT identity, controller identity, or profile ID.

## Controller and semantic contract

The only Inner action is `[delta steering, delta acceleration]` added to a
native nominal controller and projected by `TrafficActionShield`. Functional
and Logical Scenario contracts determine route, candidate, onset and legal
traffic behavior. There are no learned maneuver options or nominal behavior
profiles.

Cut-in uses vehicle-footprint lane intrusion, Merge uses the branch/mainline
conflict corridor, and Roundabout uses route-conflict windows. Collision takes
precedence over near-miss; an event bonus is captured only once.

## Training

`interaction_prior` samples the training task distribution and learns a shared
residual interaction skill with prior latent context. `context_meta` samples
task-local support/query groups. Support and query use different concrete
episodes. The group-level context replay stores support episodes once; query
transitions refer to them by group ID.

The critic and posterior/outcome objectives update the context encoder. The
actor receives a stop-gradient latent, preventing actor optimization from
arbitrarily distorting the posterior.

Training uses semantic-gated dense TTC/distance/closing-speed shaping, valid
event reward, residual energy cost, shield cost and invalidity cost. This is a
learning signal only.

## Evidence protocol

Reachability uses its own residual probe and is never reused as training or
evaluation data. The headroom casebook accepts only Base episodes that are
legal, challenge-active and not already valid-critical.

The formal score is fixed before evaluation: valid target collision is `1`,
valid critical near-miss is `0.5`, otherwise `0`.

Two independent experiments are required:

- Adaptation quality fixes eight paired query cases for every `K=0,1,2,4` and
  reports the additional support cost `K+8`.
- Budget efficiency fixes 20 total simulator episodes, charges support to that
  budget, and reports cumulative valid-critical score.

Both compare Base, random residual, shared prior and adapted `h+z` Inner. The
ablation matrix is state-only, state+`h`, state+`z`, and state+`h+z`; all four
receive the same concrete scenario input.

Formal claims require paired query provenance, lower invalidity than the
allowed baseline bound, and a pre-registered positive adaptation gain on
unseen SUT × unseen Logical domain tasks. This stage does not claim topology
OOD or Outer-policy benefit.

## Commands

```powershell
conda run -n metadrive python -m mvr.scripts.build_taskbook --output mvr/configs/taskbook.json
conda run -n metadrive python -m mvr.scripts.build_headroom_casebook --config mvr/configs/mvr.yaml --output results/mvr/headroom_casebook.json
conda run -n metadrive python -m mvr.scripts.evaluate_inner_fewshot --config mvr/configs/mvr.yaml --checkpoint results/mvr/run/context_meta.pt --casebook results/mvr/headroom_casebook.json --protocol adaptation_quality --output results/mvr/adaptation_quality.json
```
