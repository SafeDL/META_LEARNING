# Transferable Scenario Mining

`mvr` implements semantics-constrained, few-shot adversarial scenario mining.
The active method is a two-stage Inner controller; `archives/` contains frozen
baselines only.

## Task and policy

A task is `SUT × Functional Scenario × retained topology × Logical domain`.
The SUT identifier is never a model input. Observable structure is encoded as
`h = E(map, interaction candidates, Logical-domain bounds, active-parameter mask)`. A support group
produces `z`, the residual response/vulnerability characteristics not explained
by `h`. A task then contains many concrete scenarios
`c = (candidate, x0)`.

```text
support trajectories -> z
h + c + physical interaction state + z -> Inner SAC direct vehicle action
physical traffic shield -> semantic monitor
```

The Inner action is always `[steering, throttle/brake]`. Candidate
and Logical parameters define the concrete scenario; no artificial behavior
option or profile is part of the policy interface.

## Training

```text
interaction_prior -> context_meta -> outer
```

`interaction_prior` learns a shared, two-dimensional interaction skill over
the training task distribution. `context_meta` uses disjoint, task-local
support/query groups. The critic and posterior losses update the context
encoder; the actor consumes a stop-gradient latent. `outer` is deliberately
outside the first Inner few-shot claim.

## Evaluation

Use a calibration-SUT casebook, distinct from any residual-reachability probe.
Its validation-SUT screening provenance is retained for every query; it is not
claimed to make an unseen test SUT Base-safe. Test results use paired policy
deltas on the same concrete queries and three simulator seeds.
The formal score is collision `1`, near-miss `0.5`, otherwise `0`; dense TTC /
distance / closing-speed shaping is training-only.

```powershell
conda run -n metadrive python -m mvr.scripts.build_taskbook --output mvr/configs/taskbook.json
conda run -n metadrive python -m mvr.scripts.check_residual_reachability --config mvr/configs/mvr.yaml --output results/mvr/reachability.json
conda run -n metadrive python -m mvr.scripts.train_mvr --config mvr/configs/mvr.yaml --output results/mvr/run
conda run -n metadrive python -m mvr.scripts.build_calibration_casebook --config mvr/configs/mvr.yaml --output results/mvr/calibration_casebook.json
conda run -n metadrive python -m mvr.scripts.evaluate_inner_fewshot --config mvr/configs/mvr.yaml --checkpoint results/mvr/run/context_meta.pt --casebook results/mvr/calibration_casebook.json --protocol adaptation_quality --output results/mvr/adaptation_quality.json
```

`adaptation_quality` fixes eight paired query scenarios and reports the
additional `K` support cost. `budget_efficiency` fixes the total 20-episode
budget and charges support episodes against it. Neither protocol claims
topology OOD in this implementation.

## Verification

```powershell
conda run -n metadrive python -m pytest mvr/tests -q
conda run -n metadrive python -m compileall -q mvr
```
