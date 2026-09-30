# NL-IDM release change log

All three versions use `ProfiledIDMVehicle`, the same physics and collision
endpoint, and the same scenario manifest. The exact parameters and build
fingerprints are frozen in each run's `version_lineage.jsonl`.

| Release | Parent | Development purpose | Parameter changes from parent |
|---|---|---|---|
| `nl_v0` | none | Reference baseline | `time_wanted=1.5`, `desired_gap=5.0`, `comfort_acceleration=2.5`, `max_brake=5.0`; no emergency TTC safeguard |
| `nl_v1` | `nl_v0` | Efficiency and normal response calibration | `time_wanted: 1.5→1.2`, `desired_gap: 5.0→4.0`, `comfort_acceleration: 2.5→2.8`, `max_brake: 5.0→5.5` |
| `nl_v2` | `nl_v1` | Short-TTC predictive braking | `emergency_ttc: 0→3.0`, `emergency_brake_gain: 0→8.0`, `emergency_max_brake: 6.0→11.0`; all V1 following parameters inherited |

These are predeclared engineering changes; any observed number of regressions
or improvements is an experimental result, not a reason to redefine a release.
