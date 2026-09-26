# FBRT interaction redesign: implementation and measured evidence

This directory is a new contract. The frozen `memory_v2` inputs were not
relabelled or overwritten. The source is
`method_chains/failure_memory_regression/configs/interaction_holdout_catalogue.yaml`; `scenario_cases.jsonl` contains
64 Sobol/anchor candidates for each of IA and IB, with six active physical
parameters per candidate. `measured/*.jsonl` contains one real 20 Hz episode
per case and build. No target outcome was used to filter candidates.

## Implemented behavior

- IA combines a moving/braking lead, an IDM rear vehicle with a timed speed
  target increase, and an adjacent front vehicle. IB combines measured lead
  merge completion, delayed braking to a 20 m/s floor, and an IDM rear
  constraint. Ego is controlled only by the selected SUT.
- `mobil_rear_state_age` uses a 0.30 s observed rear snapshot solely for the
  candidate-lane predicted rear braking term in MOBIL. Its physical vehicles,
  current front observation, and other logic remain live. The original guard
  off and reference builds remain registered.
- The selector accepts actual parameter dimensions, nominal initial geometry
  and event relations, historical margin predictions with missing/uncertainty
  markers, and outcome-only phase/partner information in pattern cards. A
  pass-only compatible history uses the target-only branch; a history with
  failures uses a source/target mixture updated with pre-query likelihoods.

## Measured outcome

The `metadrive` environment detected an NVIDIA GeForce RTX 4090 D. The
highway-env physics and existing deterministic PPO loader execute on CPU;
there was no driving-policy training. The measured bank has 640 unique
case/build episodes, all completed with zero inconclusive outcomes.

| Build | Valid ego collisions / 128 | Notes |
|---|---:|---|
| `mobil_rear_guard_off_v2` | 5 | Four IA rear contacts in the saved bank; one IB contact has an unresolved partner in the saved row. `diagnostic_ib49.json` records a separate rerun identifying the rear contact. |
| `mobil_ref_v2` | 0 | Full parent pass for the MOBIL regression task. |
| `mobil_rear_state_age` | 0 | One observed decision differed from reference, but no new ego collision. |
| `ppo_ref_v2` | 1 | IB lead collision at first intrusion, before measured merge/braking. |
| `ppo_obs_age020_v2` | 1 | The same IB collision; no new PPO regression. |

At budget 20, the age-vs-reference and PPO age-vs-reference regression tasks
have **zero target failures**, so no method can demonstrate regression
discovery there. In the guard-off-vs-reference diagnostic task, FBRT-Memory,
FBRT-NoMemory, and FailureDistance each find 2 of the 5 failures;
HistoryRank-UCB finds 0 and Random averages 0.9 over ten fixed repeats.
For cross-agent PPO reference, one failure exists; Memory and NoMemory find
zero at budget 20, while Random finds it in one of ten repeats. These
evaluations use the same cases and valid-ego-collision reward for all methods.
The guard-off build is used as an explicit historical source only when it is
not the hidden target.

The corrected frozen-bank replay in `legacy_replay_target_only_corrected/` covers 21 tasks, 182
method runs and 3,640 logical queries, with unchanged input fingerprints.
Compared with an earlier replay now superseded by the full-bank result,
FBRT-Memory's @20 discoveries across its
13 task runs rise from 119 to 122. Its two pass-only tasks improve from
10 to 12 and 13 to 17, while two other tasks lose 2 and 1 discoveries.
Memory ties NoMemory in those two pass-only tasks. This is a **mixed**
result, not evidence of significant overall superiority.

An exact paired sign-flip audit of the ten nonempty, directly comparable
frozen tasks gives Memory vs NoMemory totals of 4:1, 26:21, 55:43, and
105:95 at budgets 1, 5, 10, and 20. The corresponding two-sided task-level
sign-flip p-values are 0.25, 0.50, 0.141, and 0.133. Grouping the nine legacy
tasks by their three shared simulator seeds (plus the compact task) gives
p-values 0.50, 0.625, 0.375, and 0.25. These are exploratory, unadjusted
tests with only four clusters; none establishes a significant advantage.
The frozen `legacy_replay_target_only_corrected/summary_by_task.csv` retains the
task-level values used for this calculation. The branch-specific
cross-agent task is excluded from the paired test because its predecessor
history differs by method.

An additional common-seed offline audit with ten selector repeats per
regression task is in `legacy_paired_seed_audit/README.md`. At budget 20,
Memory finds 1056 failures against NoMemory's 900 across the ten comparable
task pools and ten repeats each. The descriptive gain is consistent across
three legacy simulator seeds, but the four-cluster two-sided p-value is 0.25;
it is not proof of a significant advantage on the redesigned interaction
tasks.

## Reproduction and checks

```powershell
conda run -n metadrive python -m pytest method_chains/failure_memory_regression/tests -q
conda run -n metadrive python -m method_chains.failure_memory_regression.interaction --root results/method_chains/failure_memory_regression/interaction_v1 --catalogue method_chains/failure_memory_regression/configs/interaction_holdout_catalogue.yaml --builds mobil_rear_guard_off_v2 mobil_ref_v2 mobil_rear_state_age ppo_ref_v2 ppo_obs_age020_v2
conda run -n metadrive python -m method_chains.failure_memory_regression.interaction --root results/method_chains/failure_memory_regression/interaction_v1 --catalogue method_chains/failure_memory_regression/configs/interaction_holdout_catalogue.yaml --evaluate --target mobil_rear_state_age --parent mobil_ref_v2 --budget 20
conda run -n metadrive python -m method_chains.failure_memory_regression.interaction --root results/method_chains/failure_memory_regression/interaction_v1 --catalogue method_chains/failure_memory_regression/configs/interaction_holdout_catalogue.yaml --evaluate --target ppo_ref_v2 --parent none --budget 20
```

The first command passes 45 tests. The compiler was also compared against
all 80 frozen legacy cases, with exact row-by-row equality. The measurement
command resumes the
separate bank and refuses a cached scenario if its input contract changed.
`legacy_replay_target_only_corrected/repair_report.md` contains the latest frozen-bank
comparison. The observed data do not satisfy a claim that the interaction
redesign has significantly improved failure discovery. A later test design
should be fixed before observing target outcomes and evaluated on all its
candidates; changing seeds or removing successful baseline cases to create
an advantage would invalidate that claim.

The latest replay makes a pass-only Memory branch use the same target-only
feature dictionary as NoMemory. This corrects a source-feature influence that
remained even with zero source weight. All 21 frozen tasks and 3,640 logical
queries replayed without new episodes; every method's discovery count at all
four checkpoints was unchanged. A ten-query same-seed test checks equality of
the complete pass-only selection sequence, including after observed failures.

A separate, fixed development revision using seed `4179902` is documented in
`../interaction_v2/README.md`; all 640 v2 bank episodes had zero valid ego
collisions despite the scheduled events executing.
