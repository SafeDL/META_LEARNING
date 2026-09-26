# FBRT interaction development revision v2

This separate bank was frozen before physical execution. `protocol.json`
records seed `4179902`, SHA-256 of
`method_chains/failure_memory_regression/configs/interaction_catalogue.yaml`, the 128-candidate fingerprint,
and `target_outcome_filtering: false`. The v1 bank and its labels remain
unchanged. The revision was motivated by v1 event chronology: three of four
IA guard-off collisions occurred before the scheduled 3 s lead brake, and
the PPO IB collision occurred before the measured merge and subsequent brake.
V2 increases initial clearances and concentrates event offsets around the
scheduled interaction window. This is one complete development revision,
not a search retaining only favorable cases.

All five builds completed the same 128 cases at 20 Hz in `metadrive`:

| Build | Valid ego collisions | Inconclusive |
|---|---:|---:|
| `mobil_rear_guard_off_v2` | 0 | 0 |
| `mobil_ref_v2` | 0 | 0 |
| `mobil_rear_state_age` | 0 | 0 |
| `ppo_ref_v2` | 0 | 0 |
| `ppo_obs_age020_v2` | 0 | 0 |

The event mechanisms executed: in the age build, all 64 IA cases logged lead
braking and a rear target-speed increase; all 64 IB cases logged measured
merge completion and braking after that completion. Ego initiated a lane
change after the 3 s IA event in 40/64 cases. The age build recorded use of
an aged rear prediction in all 128 cases and at least one rear-guard rejection
in 125 cases. Three scenario IDs have a different ego lane-change decision
time between reference and age builds, but none caused a collision. The empty
target failure pool is therefore not explained by an inactive age mutation.
Every regression/cross-agent failure pool in this v2 bank is empty. Reporting
equal discovery scores as an algorithm improvement would be invalid.

The engine is still highway-env; the existing frozen PPO inference remains
on CPU. The machine exposes an RTX 4090 D. No driving policy was retrained.
The latest code passes 45 tests in
`method_chains/failure_memory_regression/tests`. A full replay of the frozen
legacy bank with independent source and target feature dictionaries is in
`../interaction_v1/legacy_replay_target_only_corrected/repair_report.md`;
its aggregate result remains mixed.

```powershell
conda run -n metadrive python -m method_chains.failure_memory_regression.interaction --root results/method_chains/failure_memory_regression/interaction_v2 --catalogue method_chains/failure_memory_regression/configs/interaction_catalogue.yaml --seed 4179902 --builds mobil_rear_guard_off_v2 mobil_ref_v2 mobil_rear_state_age ppo_ref_v2 ppo_obs_age020_v2
```

The runner resumes the separate bank and rejects changed cached case inputs.
Further claims of significant discovery improvement need a new, independently
fixed evaluation contract with an observed nonempty target failure pool.
