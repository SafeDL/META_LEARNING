# FBRT-Memory v2 execution report

## Executive summary

- Selected recipe: `highway_policy_compatible_5` from static policy-action capability, before method comparison.
- Legacy measured data imported: 480 reference and 960 target episodes; cache replay uses 0 new simulations.
- New physical episodes: **400 / 400**; compact-bank records: **320**.
- Delivery level: **IMPLEMENTED_WITH_GAIN**; observed method effect: `local_gain_observed`.
- PPO checkpoint: `available`; pinned asset SHA-256: `386ef97bead6fab69f863fca9f1106a2c8ca2400a0bc896cae3e195b5e094b28`.
- No policy was trained. The Bayesian logistic models ran on CPU; the `metadrive` conda environment reported CUDA availability in `protocol.json`.

The report separates engineering acceptance from test effectiveness. Local gains in legacy replay are not claims of superiority on the compact bank.

## Scope and execution decisions

The catalogue retains 14 candidates. The selected shared pool has 5 templates × 16 cases. Selected IDs: `S01, S02, S05, S06, S08`.
Installed highway-env `1.9.1` exposes PPO actions `LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER` and speed levels `[20.0, 25.0, 30.0] m/s`. Full stop: `False`; lane change: `True`.
The V2 bank uses S01, S02, S05, S06, and S08. S03/S04 are not used because the shared PPO action contract has no zero-speed action. S07 and S09–S14 remain catalogue candidates and were not run.

## Task results

`summary_by_task.csv` retains every repeat and @1/@5/@10/@20 outcome. Random is aggregated across its ten repeats below; other methods show their single run. Missing values remain explicit.

| Task | Target | Method | Queries (mean) | Failure pool | Failures found (mean) | Detected | First failure (median) | Recall (mean) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| compact_regression_mobil_ref_to_rear_guard_off | mobil_rear_guard_off_v2 | FBRT-Memory | 20.0 | 12 | 10.00 | 1/1 (100%) | 10.0 | 0.833 |
| compact_regression_mobil_ref_to_rear_guard_off | mobil_rear_guard_off_v2 | FBRT-NoMemory | 20.0 | 12 | 12.00 | 1/1 (100%) | 6.0 | 1.000 |
| compact_regression_mobil_ref_to_rear_guard_off | mobil_rear_guard_off_v2 | FailureDistance-v2 | 20.0 | 12 | 1.00 | 1/1 (100%) | 20.0 | 0.083 |
| compact_regression_mobil_ref_to_rear_guard_off | mobil_rear_guard_off_v2 | HistoryRank-UCB-v2 | 20.0 | 12 | 8.00 | 1/1 (100%) | 2.0 | 0.667 |
| compact_regression_mobil_ref_to_rear_guard_off | mobil_rear_guard_off_v2 | Random (n=10) | 20.0 | 12 | 2.70 | 9/10 (90%) | 7.0 | 0.225 |
| compact_regression_ppo_ref_to_obs_age020 | ppo_obs_age020_v2 | FBRT-Memory | 20.0 | 0 | 0.00 | 0/1 (0%) | >B | NA |
| compact_regression_ppo_ref_to_obs_age020 | ppo_obs_age020_v2 | FBRT-NoMemory | 20.0 | 0 | 0.00 | 0/1 (0%) | >B | NA |
| compact_regression_ppo_ref_to_obs_age020 | ppo_obs_age020_v2 | FailureDistance-v2 | 20.0 | 0 | 0.00 | 0/1 (0%) | >B | NA |
| compact_regression_ppo_ref_to_obs_age020 | ppo_obs_age020_v2 | HistoryRank-UCB-v2 | 20.0 | 0 | 0.00 | 0/1 (0%) | >B | NA |
| compact_regression_ppo_ref_to_obs_age020 | ppo_obs_age020_v2 | Random (n=10) | 20.0 | 0 | 0.00 | 0/10 (0%) | >B | NA |
| cross_agent_FBRT-Memory_mobil_to_next | mobil_ref_v2 | FBRT-Memory | 20.0 | 0 | 0.00 | 0/1 (0%) | >B | NA |
| cross_agent_FBRT-Memory_ppo_ref_after_mobil | ppo_ref_v2 | FBRT-Memory | 20.0 | 19 | 13.00 | 1/1 (100%) | 6.0 | 0.684 |
| cross_agent_FBRT-NoMemory_mobil_to_next | mobil_ref_v2 | FBRT-NoMemory | 20.0 | 0 | 0.00 | 0/1 (0%) | >B | NA |
| cross_agent_FBRT-NoMemory_ppo_ref_after_mobil | ppo_ref_v2 | FBRT-NoMemory | 20.0 | 19 | 17.00 | 1/1 (100%) | 2.0 | 0.895 |
| cross_agent_FailureDistance-v2_mobil_to_next | mobil_ref_v2 | FailureDistance-v2 | 20.0 | 0 | 0.00 | 0/1 (0%) | >B | NA |
| cross_agent_FailureDistance-v2_ppo_ref_after_mobil | ppo_ref_v2 | FailureDistance-v2 | 20.0 | 19 | 15.00 | 1/1 (100%) | 5.0 | 0.789 |
| cross_agent_HistoryRank-UCB-v2_mobil_to_next | mobil_ref_v2 | HistoryRank-UCB-v2 | 20.0 | 0 | 0.00 | 0/1 (0%) | >B | NA |
| cross_agent_HistoryRank-UCB-v2_ppo_ref_after_mobil | ppo_ref_v2 | HistoryRank-UCB-v2 | 20.0 | 19 | 6.00 | 1/1 (100%) | 2.0 | 0.316 |
| cross_agent_Random_mobil_to_next | mobil_ref_v2 | Random (n=10) | 20.0 | 0 | 0.00 | 0/10 (0%) | >B | NA |
| cross_agent_Random_ppo_ref_after_mobil | ppo_ref_v2 | Random (n=10) | 20.0 | 19 | 4.30 | 10/10 (100%) | 2.5 | 0.226 |
| legacy_regression_seed4179801_merge_blind06 | merge_blind06 | FBRT-Memory | 20.0 | 7 | 5.00 | 1/1 (100%) | 4.0 | 0.714 |
| legacy_regression_seed4179801_merge_blind06 | merge_blind06 | FBRT-NoMemory | 20.0 | 7 | 4.00 | 1/1 (100%) | 9.0 | 0.571 |
| legacy_regression_seed4179801_merge_blind06 | merge_blind06 | FailureDistance-v2 | 20.0 | 7 | 3.00 | 1/1 (100%) | 5.0 | 0.429 |
| legacy_regression_seed4179801_merge_blind06 | merge_blind06 | HistoryRank-UCB-v2 | 20.0 | 7 | 7.00 | 1/1 (100%) | 1.0 | 1.000 |
| legacy_regression_seed4179801_merge_blind06 | merge_blind06 | Random (n=10) | 20.0 | 7 | 1.80 | 10/10 (100%) | 9.5 | 0.257 |
| legacy_regression_seed4179801_merge_brake2 | merge_brake2 | FBRT-Memory | 20.0 | 4 | 1.00 | 1/1 (100%) | 12.0 | 0.250 |
| legacy_regression_seed4179801_merge_brake2 | merge_brake2 | FBRT-NoMemory | 20.0 | 4 | 2.00 | 1/1 (100%) | 3.0 | 0.500 |
| legacy_regression_seed4179801_merge_brake2 | merge_brake2 | FailureDistance-v2 | 20.0 | 4 | 3.00 | 1/1 (100%) | 5.0 | 0.750 |
| legacy_regression_seed4179801_merge_brake2 | merge_brake2 | HistoryRank-UCB-v2 | 20.0 | 4 | 4.00 | 1/1 (100%) | 3.0 | 1.000 |
| legacy_regression_seed4179801_merge_brake2 | merge_brake2 | Random (n=10) | 20.0 | 4 | 0.60 | 4/10 (40%) | 7.0 | 0.150 |
| legacy_regression_seed4179801_slow_front_brake2 | slow_front_brake2 | FBRT-Memory | 20.0 | 65 | 19.00 | 1/1 (100%) | 1.0 | 0.292 |
| legacy_regression_seed4179801_slow_front_brake2 | slow_front_brake2 | FBRT-NoMemory | 20.0 | 65 | 19.00 | 1/1 (100%) | 2.0 | 0.292 |
| legacy_regression_seed4179801_slow_front_brake2 | slow_front_brake2 | FailureDistance-v2 | 20.0 | 65 | 18.00 | 1/1 (100%) | 3.0 | 0.277 |
| legacy_regression_seed4179801_slow_front_brake2 | slow_front_brake2 | HistoryRank-UCB-v2 | 20.0 | 65 | 18.00 | 1/1 (100%) | 1.0 | 0.277 |
| legacy_regression_seed4179801_slow_front_brake2 | slow_front_brake2 | Random (n=10) | 20.0 | 65 | 12.30 | 10/10 (100%) | 1.0 | 0.189 |
| legacy_regression_seed4179802_merge_blind06 | merge_blind06 | FBRT-Memory | 20.0 | 12 | 3.00 | 1/1 (100%) | 11.0 | 0.250 |
| legacy_regression_seed4179802_merge_blind06 | merge_blind06 | FBRT-NoMemory | 20.0 | 12 | 8.00 | 1/1 (100%) | 2.0 | 0.667 |
| legacy_regression_seed4179802_merge_blind06 | merge_blind06 | FailureDistance-v2 | 20.0 | 12 | 3.00 | 1/1 (100%) | 7.0 | 0.250 |
| legacy_regression_seed4179802_merge_blind06 | merge_blind06 | HistoryRank-UCB-v2 | 20.0 | 12 | 11.00 | 1/1 (100%) | 1.0 | 0.917 |
| legacy_regression_seed4179802_merge_blind06 | merge_blind06 | Random (n=10) | 20.0 | 12 | 2.00 | 10/10 (100%) | 6.5 | 0.167 |
| legacy_regression_seed4179802_merge_brake2 | merge_brake2 | FBRT-Memory | 20.0 | 5 | 5.00 | 1/1 (100%) | 2.0 | 1.000 |
| legacy_regression_seed4179802_merge_brake2 | merge_brake2 | FBRT-NoMemory | 20.0 | 5 | 5.00 | 1/1 (100%) | 3.0 | 1.000 |
| legacy_regression_seed4179802_merge_brake2 | merge_brake2 | FailureDistance-v2 | 20.0 | 5 | 4.00 | 1/1 (100%) | 4.0 | 0.800 |
| legacy_regression_seed4179802_merge_brake2 | merge_brake2 | HistoryRank-UCB-v2 | 20.0 | 5 | 5.00 | 1/1 (100%) | 3.0 | 1.000 |
| legacy_regression_seed4179802_merge_brake2 | merge_brake2 | Random (n=10) | 20.0 | 5 | 0.60 | 5/10 (50%) | 9.0 | 0.120 |
| legacy_regression_seed4179802_slow_front_brake2 | slow_front_brake2 | FBRT-Memory | 20.0 | 68 | 19.00 | 1/1 (100%) | 1.0 | 0.279 |
| legacy_regression_seed4179802_slow_front_brake2 | slow_front_brake2 | FBRT-NoMemory | 20.0 | 68 | 19.00 | 1/1 (100%) | 1.0 | 0.279 |
| legacy_regression_seed4179802_slow_front_brake2 | slow_front_brake2 | FailureDistance-v2 | 20.0 | 68 | 15.00 | 1/1 (100%) | 1.0 | 0.221 |
| legacy_regression_seed4179802_slow_front_brake2 | slow_front_brake2 | HistoryRank-UCB-v2 | 20.0 | 68 | 18.00 | 1/1 (100%) | 1.0 | 0.265 |
| legacy_regression_seed4179802_slow_front_brake2 | slow_front_brake2 | Random (n=10) | 20.0 | 68 | 11.60 | 10/10 (100%) | 1.0 | 0.171 |
| legacy_regression_seed4179803_merge_blind06 | merge_blind06 | FBRT-Memory | 20.0 | 11 | 7.00 | 1/1 (100%) | 4.0 | 0.636 |
| legacy_regression_seed4179803_merge_blind06 | merge_blind06 | FBRT-NoMemory | 20.0 | 11 | 4.00 | 1/1 (100%) | 6.0 | 0.364 |
| legacy_regression_seed4179803_merge_blind06 | merge_blind06 | FailureDistance-v2 | 20.0 | 11 | 4.00 | 1/1 (100%) | 5.0 | 0.364 |
| legacy_regression_seed4179803_merge_blind06 | merge_blind06 | HistoryRank-UCB-v2 | 20.0 | 11 | 10.00 | 1/1 (100%) | 1.0 | 0.909 |
| legacy_regression_seed4179803_merge_blind06 | merge_blind06 | Random (n=10) | 20.0 | 11 | 2.40 | 10/10 (100%) | 4.0 | 0.218 |
| legacy_regression_seed4179803_merge_brake2 | merge_brake2 | FBRT-Memory | 20.0 | 4 | 2.00 | 1/1 (100%) | 5.0 | 0.500 |
| legacy_regression_seed4179803_merge_brake2 | merge_brake2 | FBRT-NoMemory | 20.0 | 4 | 3.00 | 1/1 (100%) | 5.0 | 0.750 |
| legacy_regression_seed4179803_merge_brake2 | merge_brake2 | FailureDistance-v2 | 20.0 | 4 | 4.00 | 1/1 (100%) | 8.0 | 1.000 |
| legacy_regression_seed4179803_merge_brake2 | merge_brake2 | HistoryRank-UCB-v2 | 20.0 | 4 | 4.00 | 1/1 (100%) | 1.0 | 1.000 |
| legacy_regression_seed4179803_merge_brake2 | merge_brake2 | Random (n=10) | 20.0 | 4 | 0.90 | 7/10 (70%) | 13.0 | 0.225 |
| legacy_regression_seed4179803_slow_front_brake2 | slow_front_brake2 | FBRT-Memory | 20.0 | 68 | 18.00 | 1/1 (100%) | 1.0 | 0.265 |
| legacy_regression_seed4179803_slow_front_brake2 | slow_front_brake2 | FBRT-NoMemory | 20.0 | 68 | 19.00 | 1/1 (100%) | 2.0 | 0.279 |
| legacy_regression_seed4179803_slow_front_brake2 | slow_front_brake2 | FailureDistance-v2 | 20.0 | 68 | 18.00 | 1/1 (100%) | 1.0 | 0.265 |
| legacy_regression_seed4179803_slow_front_brake2 | slow_front_brake2 | HistoryRank-UCB-v2 | 20.0 | 68 | 18.00 | 1/1 (100%) | 1.0 | 0.265 |
| legacy_regression_seed4179803_slow_front_brake2 | slow_front_brake2 | Random (n=10) | 20.0 | 68 | 12.40 | 10/10 (100%) | 1.0 | 0.182 |

## Compact regression reading

The MOBIL compact holdout contains 80 candidates and 12 verified target-build failures. At B=20, results were:
- FBRT-Memory: 10.00 failures found on average (100% of runs detected at least one).
- FBRT-NoMemory: 12.00 failures found on average (100% of runs detected at least one).
- HistoryRank-UCB-v2: 8.00 failures found on average (100% of runs detected at least one).
- FailureDistance-v2: 1.00 failures found on average (100% of runs detected at least one).
- Random: 2.70 failures found on average (90% of runs detected at least one) across 10 repeats.
This development holdout did not show a benefit from FBRT-Memory at B=20; the `local_gain_observed` acceptance flag refers to observations elsewhere in the legacy replay and is not a general superiority claim.

## PPO and cross-agent reading

The PPO observation-delay regression had 61 parent-pass candidates and 0 new target failures. Its failure recall is undefined because this bank contained no PPO regressions.
The independent PPO-reference session had 19 collisions among 80 candidates. At B=20, the selectors found:
- FBRT-Memory: 13 collisions.
- FBRT-NoMemory: 17 collisions.
- HistoryRank-UCB-v2: 6 collisions.
- FailureDistance-v2: 15 collisions.
- Random: 4.30 collisions on average across 10 repeats.
The preceding MOBIL-reference session observed no collisions, so this sequence did not transfer a prior failure pattern to PPO.
The PPO ZIP was saved with Stable-Baselines3 2.7.0 and inferred under 2.3.0; the loader warned about training-schedule deserialization, while deterministic inference and the smoke runs completed.

## Physical bank

| Build | Episodes | Ego collisions | Inconclusive |
|---|---:|---:|---:|
| mobil_rear_guard_off_v2 | 80 | 12 | 0 |
| mobil_ref_v2 | 80 | 0 | 0 |
| ppo_obs_age020_v2 | 80 | 18 | 0 |
| ppo_ref_v2 | 80 | 19 | 0 |

| Template | Build | Episodes | Ego collisions | Inconclusive |
|---|---|---:|---:|---:|
| fbrt_cutin | mobil_rear_guard_off_v2 | 16 | 0 | 0 |
| fbrt_cutin | mobil_ref_v2 | 16 | 0 | 0 |
| fbrt_cutin | ppo_obs_age020_v2 | 16 | 0 | 0 |
| fbrt_cutin | ppo_ref_v2 | 16 | 0 | 0 |
| fbrt_cutin_then_brake | mobil_rear_guard_off_v2 | 16 | 0 | 0 |
| fbrt_cutin_then_brake | mobil_ref_v2 | 16 | 0 | 0 |
| fbrt_cutin_then_brake | ppo_obs_age020_v2 | 16 | 0 | 0 |
| fbrt_cutin_then_brake | ppo_ref_v2 | 16 | 0 | 0 |
| fbrt_cutout_static | mobil_rear_guard_off_v2 | 16 | 0 | 0 |
| fbrt_cutout_static | mobil_ref_v2 | 16 | 0 | 0 |
| fbrt_cutout_static | ppo_obs_age020_v2 | 16 | 15 | 0 |
| fbrt_cutout_static | ppo_ref_v2 | 16 | 16 | 0 |
| fbrt_lane_change_rear | mobil_rear_guard_off_v2 | 16 | 12 | 0 |
| fbrt_lane_change_rear | mobil_ref_v2 | 16 | 0 | 0 |
| fbrt_lane_change_rear | ppo_obs_age020_v2 | 16 | 3 | 0 |
| fbrt_lane_change_rear | ppo_ref_v2 | 16 | 3 | 0 |
| fbrt_moving_lead | mobil_rear_guard_off_v2 | 16 | 0 | 0 |
| fbrt_moving_lead | mobil_ref_v2 | 16 | 0 | 0 |
| fbrt_moving_lead | ppo_obs_age020_v2 | 16 | 0 | 0 |
| fbrt_moving_lead | ppo_ref_v2 | 16 | 0 | 0 |

### Event interpretation limit

`first_exit_s` was incorrectly recorded as 0.0 in 64/64 frozen v2 S02 episodes by the destination-lane event detector; do not use that timestamp. 31 PPO S02 collisions were with the moving lead vehicle, so they do not establish a collision with the revealed static target. Collision labels and partner roles are retained as measured. The separate v4 audit corrects the detector without changing this bank.
## Pattern memory and persistence

The memory contains 52 pattern cards with execution references. Pattern centers use same-context normalized scenario inputs; system response labels remain separated. The compact cross-agent branch stores only queried valid outcomes. Snapshot metadata and hashes are in `history_snapshots/`.

The selector logs contributing pattern IDs per query. New failure centers are saved only when a real queried collision is observed; unqueried candidates remain unlabeled.

## Costs and artifacts

- New physical calls: 400 / 400.
- Physical call categories: `{"compact_bank": 300, "paired_replay": 0, "repair_or_retry": 0, "smoke": 20, "validation_audit": 80}`.
- Cached legacy replay cost: 0; accumulated logical query count: 3640.
- Wall time recorded by stages: 602.808 s.
- The v3 and v4 validation audits used 80 separate physical episodes. Their GIFs and findings are in `validation_audit_v4/README.md`; they are excluded from the frozen v2 method rankings.
- Figures: cumulative failure discovery, measured labels with observed pattern support, and pattern lifecycle.
- Source provenance and limits: `scenario_sources.md`; full 14-card catalogue: `scenario_catalogue_v2.yaml`.
- Per-session query logs, updates and snapshot hashes are under `sessions/` and `history_snapshots/`.

## Acceptance

Overall: **IMPLEMENTED_WITH_GAIN**. Each E1–E10 item, evidence, physical cap, and resource status is recorded in `acceptance.json`.

The empirical result is reported as observed. This is a development replay and a compact bank; it does not establish general superiority, statistical significance, complete policy validation, or standards certification.
