# FBRT Memory repair replay report

- `engineering_status`: **fixed_and_replayed**
- `empirical_effect`: **mixed**
- Additional physical episodes: **0**; policy training runs: **0**.
- Offline task runs recorded: **182**; result tasks: **21**.
- Unit test result: `39 passed`.
- The frozen input fingerprints matched before and after replay: `True`.

## Code changes

- Historical modeling now keeps valid collision terminated outcomes and excludes evaluator only or incomplete rows; regression candidates still require a complete parent pass.
- RBF schemas include bias, named coordinates, failure centers, coverage centers, template/context/parameterization identity, and feature semantics. Source priors are reindexed by feature ID; new features receive N(0, 4).
- UCB uses regression rewards for regression tasks and valid collision rewards for cross agent tasks. Inconclusive outcomes count as queries without positive reward.
- Observed outcomes and pattern cards retain collision partner, event, contract, signature, and trajectory fields when present. Missing partner labels remain unknown.
- The offline replay reads the frozen banks and writes only under this repair output directory.

## Task effects at B=20

| Task | Method | Queries | Failures found | Failure pool | Recall | History state |
|---|---|---:|---:|---:|---:|---|
| compact_regression_mobil_ref_to_rear_guard_off | FBRT-Memory | 20.0 | 12.00 | 12 | 1.000 | pass_only |
| compact_regression_mobil_ref_to_rear_guard_off | FBRT-NoMemory | 20.0 | 12.00 | 12 | 1.000 | pass_only |
| compact_regression_mobil_ref_to_rear_guard_off | FailureDistance-v2 | 20.0 | 6.00 | 12 | 0.500 | pass_only |
| compact_regression_mobil_ref_to_rear_guard_off | HistoryRank-UCB-v2 | 20.0 | 8.00 | 12 | 0.667 | pass_only |
| compact_regression_mobil_ref_to_rear_guard_off | Random (mean, n=10) | 20.0 | 2.80 | 12 | 0.233 | pass_only |
| compact_regression_ppo_ref_to_obs_age020 | FBRT-Memory | 20.0 | 0.00 | 0 | NA | failure_and_pass |
| compact_regression_ppo_ref_to_obs_age020 | FBRT-NoMemory | 20.0 | 0.00 | 0 | NA | failure_and_pass |
| compact_regression_ppo_ref_to_obs_age020 | FailureDistance-v2 | 20.0 | 0.00 | 0 | NA | failure_and_pass |
| compact_regression_ppo_ref_to_obs_age020 | HistoryRank-UCB-v2 | 20.0 | 0.00 | 0 | NA | failure_and_pass |
| compact_regression_ppo_ref_to_obs_age020 | Random (mean, n=10) | 20.0 | 0.00 | 0 | NA | failure_and_pass |
| cross_agent_mobil_to_next | FBRT-Memory | 20.0 | 0.00 | 0 | NA | empty |
| cross_agent_mobil_to_next | FBRT-NoMemory | 20.0 | 0.00 | 0 | NA | empty |
| cross_agent_mobil_to_next | FailureDistance-v2 | 20.0 | 0.00 | 0 | NA | empty |
| cross_agent_mobil_to_next | HistoryRank-UCB-v2 | 20.0 | 0.00 | 0 | NA | empty |
| cross_agent_mobil_to_next | Random (mean, n=10) | 20.0 | 0.00 | 0 | NA | empty |
| cross_agent_ppo_ref_after_mobil | FBRT-Memory | 20.0 | 17.00 | 19 | 0.895 | pass_only |
| cross_agent_ppo_ref_after_mobil | FBRT-NoMemory | 20.0 | 17.00 | 19 | 0.895 | pass_only |
| cross_agent_ppo_ref_after_mobil | FailureDistance-v2 | 20.0 | 15.00 | 19 | 0.789 | pass_only |
| cross_agent_ppo_ref_after_mobil | HistoryRank-UCB-v2 | 20.0 | 12.00 | 19 | 0.632 | pass_only |
| cross_agent_ppo_ref_after_mobil | Random (mean, n=10) | 20.0 | 4.30 | 19 | 0.226 | pass_only |
| legacy_regression_seed4179801_merge_blind06 | FBRT-Memory | 20.0 | 7.00 | 7 | 1.000 | failure_and_pass |
| legacy_regression_seed4179801_merge_blind06 | FBRT-NoMemory | 20.0 | 4.00 | 7 | 0.571 | failure_and_pass |
| legacy_regression_seed4179801_merge_blind06 | FailureDistance-v2 | 20.0 | 7.00 | 7 | 1.000 | failure_and_pass |
| legacy_regression_seed4179801_merge_blind06 | HistoryRank-UCB-v2 | 20.0 | 7.00 | 7 | 1.000 | failure_and_pass |
| legacy_regression_seed4179801_merge_blind06 | Random (mean, n=10) | 20.0 | 1.80 | 7 | 0.257 | failure_and_pass |
| legacy_regression_seed4179801_merge_brake2 | FBRT-Memory | 20.0 | 4.00 | 4 | 1.000 | failure_and_pass |
| legacy_regression_seed4179801_merge_brake2 | FBRT-NoMemory | 20.0 | 2.00 | 4 | 0.500 | failure_and_pass |
| legacy_regression_seed4179801_merge_brake2 | FailureDistance-v2 | 20.0 | 4.00 | 4 | 1.000 | failure_and_pass |
| legacy_regression_seed4179801_merge_brake2 | HistoryRank-UCB-v2 | 20.0 | 4.00 | 4 | 1.000 | failure_and_pass |
| legacy_regression_seed4179801_merge_brake2 | Random (mean, n=10) | 20.0 | 0.60 | 4 | 0.150 | failure_and_pass |
| legacy_regression_seed4179801_slow_front_brake2 | FBRT-Memory | 20.0 | 19.00 | 65 | 0.292 | failure_and_pass |
| legacy_regression_seed4179801_slow_front_brake2 | FBRT-NoMemory | 20.0 | 19.00 | 65 | 0.292 | failure_and_pass |
| legacy_regression_seed4179801_slow_front_brake2 | FailureDistance-v2 | 20.0 | 15.00 | 65 | 0.231 | failure_and_pass |
| legacy_regression_seed4179801_slow_front_brake2 | HistoryRank-UCB-v2 | 20.0 | 18.00 | 65 | 0.277 | failure_and_pass |
| legacy_regression_seed4179801_slow_front_brake2 | Random (mean, n=10) | 20.0 | 12.30 | 65 | 0.189 | failure_and_pass |
| legacy_regression_seed4179802_merge_blind06 | FBRT-Memory | 20.0 | 9.00 | 12 | 0.750 | failure_and_pass |
| legacy_regression_seed4179802_merge_blind06 | FBRT-NoMemory | 20.0 | 8.00 | 12 | 0.667 | failure_and_pass |
| legacy_regression_seed4179802_merge_blind06 | FailureDistance-v2 | 20.0 | 9.00 | 12 | 0.750 | failure_and_pass |
| legacy_regression_seed4179802_merge_blind06 | HistoryRank-UCB-v2 | 20.0 | 11.00 | 12 | 0.917 | failure_and_pass |
| legacy_regression_seed4179802_merge_blind06 | Random (mean, n=10) | 20.0 | 2.00 | 12 | 0.167 | failure_and_pass |
| legacy_regression_seed4179802_merge_brake2 | FBRT-Memory | 20.0 | 4.00 | 5 | 0.800 | failure_and_pass |
| legacy_regression_seed4179802_merge_brake2 | FBRT-NoMemory | 20.0 | 5.00 | 5 | 1.000 | failure_and_pass |
| legacy_regression_seed4179802_merge_brake2 | FailureDistance-v2 | 20.0 | 4.00 | 5 | 0.800 | failure_and_pass |
| legacy_regression_seed4179802_merge_brake2 | HistoryRank-UCB-v2 | 20.0 | 5.00 | 5 | 1.000 | failure_and_pass |
| legacy_regression_seed4179802_merge_brake2 | Random (mean, n=10) | 20.0 | 0.60 | 5 | 0.120 | failure_and_pass |
| legacy_regression_seed4179802_slow_front_brake2 | FBRT-Memory | 20.0 | 20.00 | 68 | 0.294 | failure_and_pass |
| legacy_regression_seed4179802_slow_front_brake2 | FBRT-NoMemory | 20.0 | 19.00 | 68 | 0.279 | failure_and_pass |
| legacy_regression_seed4179802_slow_front_brake2 | FailureDistance-v2 | 20.0 | 17.00 | 68 | 0.250 | failure_and_pass |
| legacy_regression_seed4179802_slow_front_brake2 | HistoryRank-UCB-v2 | 20.0 | 18.00 | 68 | 0.265 | failure_and_pass |
| legacy_regression_seed4179802_slow_front_brake2 | Random (mean, n=10) | 20.0 | 11.60 | 68 | 0.171 | failure_and_pass |
| legacy_regression_seed4179803_merge_blind06 | FBRT-Memory | 20.0 | 8.00 | 11 | 0.727 | failure_and_pass |
| legacy_regression_seed4179803_merge_blind06 | FBRT-NoMemory | 20.0 | 4.00 | 11 | 0.364 | failure_and_pass |
| legacy_regression_seed4179803_merge_blind06 | FailureDistance-v2 | 20.0 | 10.00 | 11 | 0.909 | failure_and_pass |
| legacy_regression_seed4179803_merge_blind06 | HistoryRank-UCB-v2 | 20.0 | 10.00 | 11 | 0.909 | failure_and_pass |
| legacy_regression_seed4179803_merge_blind06 | Random (mean, n=10) | 20.0 | 2.40 | 11 | 0.218 | failure_and_pass |
| legacy_regression_seed4179803_merge_brake2 | FBRT-Memory | 20.0 | 4.00 | 4 | 1.000 | failure_and_pass |
| legacy_regression_seed4179803_merge_brake2 | FBRT-NoMemory | 20.0 | 3.00 | 4 | 0.750 | failure_and_pass |
| legacy_regression_seed4179803_merge_brake2 | FailureDistance-v2 | 20.0 | 4.00 | 4 | 1.000 | failure_and_pass |
| legacy_regression_seed4179803_merge_brake2 | HistoryRank-UCB-v2 | 20.0 | 4.00 | 4 | 1.000 | failure_and_pass |
| legacy_regression_seed4179803_merge_brake2 | Random (mean, n=10) | 20.0 | 0.90 | 4 | 0.225 | failure_and_pass |
| legacy_regression_seed4179803_slow_front_brake2 | FBRT-Memory | 20.0 | 18.00 | 68 | 0.265 | failure_and_pass |
| legacy_regression_seed4179803_slow_front_brake2 | FBRT-NoMemory | 20.0 | 19.00 | 68 | 0.279 | failure_and_pass |
| legacy_regression_seed4179803_slow_front_brake2 | FailureDistance-v2 | 20.0 | 14.00 | 68 | 0.206 | failure_and_pass |
| legacy_regression_seed4179803_slow_front_brake2 | HistoryRank-UCB-v2 | 20.0 | 18.00 | 68 | 0.265 | failure_and_pass |
| legacy_regression_seed4179803_slow_front_brake2 | Random (mean, n=10) | 20.0 | 12.40 | 68 | 0.182 | failure_and_pass |

## Applicability and interpretation

`HAS_FAILURE_MEMORY`, `PASS_ONLY_HISTORY`, and `NO_COMPATIBLE_HISTORY` describe the source view actually supplied to each run. `NO_TARGET_FAILURE_IN_POOL` means recall is not applicable. `BASELINE_AT_ORACLE_CEILING` marks runs whose B=20 discoveries equal the complete target failure pool. Missing files are listed as `MISSING_CACHE`; no missing episode was generated.

At B=20, Memory versus NoMemory and the existing baselines are:

| Task | Failure pool | Memory | NoMemory | Δ vs NoMemory | HistoryRank-UCB | FailureDistance | Random mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact_regression_mobil_ref_to_rear_guard_off | 12 | 12.00 | 12.00 | 0.00 | 8.00 | 6.00 | 2.80 |
| compact_regression_ppo_ref_to_obs_age020 | 0 | NA | NA | NA | NA | NA | NA |
| cross_agent_mobil_to_next | 0 | NA | NA | NA | NA | NA | NA |
| cross_agent_ppo_ref_after_mobil | 19 | 17.00 | 17.00 | 0.00 | 12.00 | 15.00 | 4.30 |
| legacy_regression_seed4179801_merge_blind06 | 7 | 7.00 | 4.00 | 3.00 | 7.00 | 7.00 | 1.80 |
| legacy_regression_seed4179801_merge_brake2 | 4 | 4.00 | 2.00 | 2.00 | 4.00 | 4.00 | 0.60 |
| legacy_regression_seed4179801_slow_front_brake2 | 65 | 19.00 | 19.00 | 0.00 | 18.00 | 15.00 | 12.30 |
| legacy_regression_seed4179802_merge_blind06 | 12 | 9.00 | 8.00 | 1.00 | 11.00 | 9.00 | 2.00 |
| legacy_regression_seed4179802_merge_brake2 | 5 | 4.00 | 5.00 | -1.00 | 5.00 | 4.00 | 0.60 |
| legacy_regression_seed4179802_slow_front_brake2 | 68 | 20.00 | 19.00 | 1.00 | 18.00 | 17.00 | 11.60 |
| legacy_regression_seed4179803_merge_blind06 | 11 | 8.00 | 4.00 | 4.00 | 10.00 | 10.00 | 2.40 |
| legacy_regression_seed4179803_merge_brake2 | 4 | 4.00 | 3.00 | 1.00 | 4.00 | 4.00 | 0.90 |
| legacy_regression_seed4179803_slow_front_brake2 | 68 | 18.00 | 19.00 | -1.00 | 18.00 | 14.00 | 12.40 |

Random entries are means across the original ten repeats; other methods use their single original run.
Compact S02 observed collision partner counts (lead/static/rear/unknown): `{"lead": 31, "rear": 0, "static": 0, "unknown": 0}`.
The original S02 `first_exit_s=0` values are retained as unreliable event data (64 bank rows); no v4 trajectory or inferred object label changed the v2 ranking labels.

## Answers

1. Historical failures reach pattern construction and source fitting when the task context admits them; the task input ledger records their exact counts and IDs.
2. Feature coefficients and variances follow stable IDs after each dynamic center insertion; source prior arrays are refit against all queried valid observations exactly once per posterior fit.
3. UCB rewards follow the task mode, and query/update fields record the valid collision and reward used.
4. The measured Memory effect is `mixed` under the existing tasks; task level deltas are in `comparison_before_after.csv` and `memory_vs_baselines.json`.
5. Tasks without source failures, compatible history, target regressions, or complete cache coverage are labeled separately; they do not support the corresponding migration claim.

## Provenance

- Source HEAD: `004c8ade6cc69eff88b47f8532c9500e70687866`
- Replay fingerprint: `15cf77fdf283f8ba0f49731b2de51f61b1016aec4c6d4ac58457e308aa63291a`
- Logical queries: 3640
- Model fits: 3126
- Input files fingerprinted: 691
- Missing inputs: `[]`
- See `manifest.json` and `task_inputs.jsonl` for detailed provenance.
