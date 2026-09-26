# FBRT Memory repair replay report

- `engineering_status`: **fixed_and_replayed**
- `empirical_effect`: **mixed**
- Additional physical episodes: **0**; policy training runs: **0**.
- Offline task runs recorded: **578**; result tasks: **21**.
- Unit test result: `41 passed`.
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
| compact_regression_mobil_ref_to_rear_guard_off | FBRT-Memory | 20.0 | 11.30 | 12 | 0.942 | pass_only |
| compact_regression_mobil_ref_to_rear_guard_off | FBRT-NoMemory | 20.0 | 11.30 | 12 | 0.942 | pass_only |
| compact_regression_mobil_ref_to_rear_guard_off | FailureDistance-v2 | 20.0 | 10.00 | 12 | 0.833 | pass_only |
| compact_regression_mobil_ref_to_rear_guard_off | HistoryRank-UCB-v2 | 20.0 | 7.20 | 12 | 0.600 | pass_only |
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
| legacy_regression_seed4179801_merge_blind06 | Random (mean, n=10) | 20.0 | 1.30 | 7 | 0.186 | failure_and_pass |
| legacy_regression_seed4179801_merge_brake2 | FBRT-Memory | 20.0 | 4.00 | 4 | 1.000 | failure_and_pass |
| legacy_regression_seed4179801_merge_brake2 | FBRT-NoMemory | 20.0 | 1.80 | 4 | 0.450 | failure_and_pass |
| legacy_regression_seed4179801_merge_brake2 | FailureDistance-v2 | 20.0 | 4.00 | 4 | 1.000 | failure_and_pass |
| legacy_regression_seed4179801_merge_brake2 | HistoryRank-UCB-v2 | 20.0 | 4.00 | 4 | 1.000 | failure_and_pass |
| legacy_regression_seed4179801_merge_brake2 | Random (mean, n=10) | 20.0 | 0.70 | 4 | 0.175 | failure_and_pass |
| legacy_regression_seed4179801_slow_front_brake2 | FBRT-Memory | 20.0 | 18.80 | 65 | 0.289 | failure_and_pass |
| legacy_regression_seed4179801_slow_front_brake2 | FBRT-NoMemory | 20.0 | 18.60 | 65 | 0.286 | failure_and_pass |
| legacy_regression_seed4179801_slow_front_brake2 | FailureDistance-v2 | 20.0 | 14.90 | 65 | 0.229 | failure_and_pass |
| legacy_regression_seed4179801_slow_front_brake2 | HistoryRank-UCB-v2 | 20.0 | 18.00 | 65 | 0.277 | failure_and_pass |
| legacy_regression_seed4179801_slow_front_brake2 | Random (mean, n=10) | 20.0 | 13.20 | 65 | 0.203 | failure_and_pass |
| legacy_regression_seed4179802_merge_blind06 | FBRT-Memory | 20.0 | 8.30 | 12 | 0.692 | failure_and_pass |
| legacy_regression_seed4179802_merge_blind06 | FBRT-NoMemory | 20.0 | 5.80 | 12 | 0.483 | failure_and_pass |
| legacy_regression_seed4179802_merge_blind06 | FailureDistance-v2 | 20.0 | 9.00 | 12 | 0.750 | failure_and_pass |
| legacy_regression_seed4179802_merge_blind06 | HistoryRank-UCB-v2 | 20.0 | 11.00 | 12 | 0.917 | failure_and_pass |
| legacy_regression_seed4179802_merge_blind06 | Random (mean, n=10) | 20.0 | 2.20 | 12 | 0.183 | failure_and_pass |
| legacy_regression_seed4179802_merge_brake2 | FBRT-Memory | 20.0 | 4.80 | 5 | 0.960 | failure_and_pass |
| legacy_regression_seed4179802_merge_brake2 | FBRT-NoMemory | 20.0 | 4.00 | 5 | 0.800 | failure_and_pass |
| legacy_regression_seed4179802_merge_brake2 | FailureDistance-v2 | 20.0 | 4.00 | 5 | 0.800 | failure_and_pass |
| legacy_regression_seed4179802_merge_brake2 | HistoryRank-UCB-v2 | 20.0 | 5.00 | 5 | 1.000 | failure_and_pass |
| legacy_regression_seed4179802_merge_brake2 | Random (mean, n=10) | 20.0 | 1.50 | 5 | 0.300 | failure_and_pass |
| legacy_regression_seed4179802_slow_front_brake2 | FBRT-Memory | 20.0 | 18.80 | 68 | 0.276 | failure_and_pass |
| legacy_regression_seed4179802_slow_front_brake2 | FBRT-NoMemory | 20.0 | 18.80 | 68 | 0.276 | failure_and_pass |
| legacy_regression_seed4179802_slow_front_brake2 | FailureDistance-v2 | 20.0 | 17.00 | 68 | 0.250 | failure_and_pass |
| legacy_regression_seed4179802_slow_front_brake2 | HistoryRank-UCB-v2 | 20.0 | 18.00 | 68 | 0.265 | failure_and_pass |
| legacy_regression_seed4179802_slow_front_brake2 | Random (mean, n=10) | 20.0 | 12.10 | 68 | 0.178 | failure_and_pass |
| legacy_regression_seed4179803_merge_blind06 | FBRT-Memory | 20.0 | 10.10 | 11 | 0.918 | failure_and_pass |
| legacy_regression_seed4179803_merge_blind06 | FBRT-NoMemory | 20.0 | 4.90 | 11 | 0.445 | failure_and_pass |
| legacy_regression_seed4179803_merge_blind06 | FailureDistance-v2 | 20.0 | 10.00 | 11 | 0.909 | failure_and_pass |
| legacy_regression_seed4179803_merge_blind06 | HistoryRank-UCB-v2 | 20.0 | 10.00 | 11 | 0.909 | failure_and_pass |
| legacy_regression_seed4179803_merge_blind06 | Random (mean, n=10) | 20.0 | 2.80 | 11 | 0.255 | failure_and_pass |
| legacy_regression_seed4179803_merge_brake2 | FBRT-Memory | 20.0 | 4.00 | 4 | 1.000 | failure_and_pass |
| legacy_regression_seed4179803_merge_brake2 | FBRT-NoMemory | 20.0 | 2.20 | 4 | 0.550 | failure_and_pass |
| legacy_regression_seed4179803_merge_brake2 | FailureDistance-v2 | 20.0 | 4.00 | 4 | 1.000 | failure_and_pass |
| legacy_regression_seed4179803_merge_brake2 | HistoryRank-UCB-v2 | 20.0 | 4.00 | 4 | 1.000 | failure_and_pass |
| legacy_regression_seed4179803_merge_brake2 | Random (mean, n=10) | 20.0 | 0.60 | 4 | 0.150 | failure_and_pass |
| legacy_regression_seed4179803_slow_front_brake2 | FBRT-Memory | 20.0 | 18.50 | 68 | 0.272 | failure_and_pass |
| legacy_regression_seed4179803_slow_front_brake2 | FBRT-NoMemory | 20.0 | 18.60 | 68 | 0.274 | failure_and_pass |
| legacy_regression_seed4179803_slow_front_brake2 | FailureDistance-v2 | 20.0 | 14.30 | 68 | 0.210 | failure_and_pass |
| legacy_regression_seed4179803_slow_front_brake2 | HistoryRank-UCB-v2 | 20.0 | 18.00 | 68 | 0.265 | failure_and_pass |
| legacy_regression_seed4179803_slow_front_brake2 | Random (mean, n=10) | 20.0 | 13.80 | 68 | 0.203 | failure_and_pass |

## Applicability and interpretation

`HAS_FAILURE_MEMORY`, `PASS_ONLY_HISTORY`, and `NO_COMPATIBLE_HISTORY` describe the source view actually supplied to each run. `NO_TARGET_FAILURE_IN_POOL` means recall is not applicable. `BASELINE_AT_ORACLE_CEILING` marks runs whose B=20 discoveries equal the complete target failure pool. Missing files are listed as `MISSING_CACHE`; no missing episode was generated.

At B=20, Memory versus NoMemory and the existing baselines are:

| Task | Failure pool | Memory | NoMemory | Δ vs NoMemory | HistoryRank-UCB | FailureDistance | Random mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact_regression_mobil_ref_to_rear_guard_off | 12 | 11.30 | 11.30 | 0.00 | 7.20 | 10.00 | 2.80 |
| compact_regression_ppo_ref_to_obs_age020 | 0 | NA | NA | NA | NA | NA | NA |
| cross_agent_mobil_to_next | 0 | NA | NA | NA | NA | NA | NA |
| cross_agent_ppo_ref_after_mobil | 19 | 17.00 | 17.00 | 0.00 | 12.00 | 15.00 | 4.30 |
| legacy_regression_seed4179801_merge_blind06 | 7 | 7.00 | 4.00 | 3.00 | 7.00 | 7.00 | 1.30 |
| legacy_regression_seed4179801_merge_brake2 | 4 | 4.00 | 1.80 | 2.20 | 4.00 | 4.00 | 0.70 |
| legacy_regression_seed4179801_slow_front_brake2 | 65 | 18.80 | 18.60 | 0.20 | 18.00 | 14.90 | 13.20 |
| legacy_regression_seed4179802_merge_blind06 | 12 | 8.30 | 5.80 | 2.50 | 11.00 | 9.00 | 2.20 |
| legacy_regression_seed4179802_merge_brake2 | 5 | 4.80 | 4.00 | 0.80 | 5.00 | 4.00 | 1.50 |
| legacy_regression_seed4179802_slow_front_brake2 | 68 | 18.80 | 18.80 | 0.00 | 18.00 | 17.00 | 12.10 |
| legacy_regression_seed4179803_merge_blind06 | 11 | 10.10 | 4.90 | 5.20 | 10.00 | 10.00 | 2.80 |
| legacy_regression_seed4179803_merge_brake2 | 4 | 4.00 | 2.20 | 1.80 | 4.00 | 4.00 | 0.60 |
| legacy_regression_seed4179803_slow_front_brake2 | 68 | 18.50 | 18.60 | -0.10 | 18.00 | 14.30 | 13.80 |

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
- Replay fingerprint: `0784ec6227caacbf22d841f452f182ce69311f02adb955da6c9f8b8383707251`
- Logical queries: 11560
- Model fits: 23586
- Input files fingerprinted: 691
- Missing inputs: `[]`
- See `manifest.json` and `task_inputs.jsonl` for detailed provenance.
