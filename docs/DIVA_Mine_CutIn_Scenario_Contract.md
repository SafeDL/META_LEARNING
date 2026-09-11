# DIVA-Mine Cut-in physical scenario contract

The active Cut-in Logical scenario is

`(initial_gap_m, ego_initial_speed_mps, relative_speed_mps, cutin_start_offset_m, cutin_path_length_m)`

with the discrete `candidate_index` selecting the left or right red vehicle.

| Parameter | Meaning | Range |
|---|---|---:|
| `initial_gap_m` | Initial longitudinal distance from the red vehicle's rear bumper to the ego vehicle's front bumper. | 7–16 m |
| `ego_initial_speed_mps` | Ego speed at reset. | 7–13 m/s |
| `relative_speed_mps` | Red initial speed minus ego initial speed. | −2.5–2.5 m/s |
| `cutin_start_offset_m` | Spatial position at which the red vehicle begins its prescribed lateral path, measured from the legal merge window start. | 35–600 m |
| `cutin_path_length_m` | Longitudinal length of the prescribed Cut-in path. | 30–130 m |

The red initial speed is derived once at reset: `v_red = v_ego + relative_speed_mps`. The main vehicle speed is constrained to 7–13 m/s. The red vehicle has no separate speed range: its derived speed must only remain positive, and it then tracks that prescribed value without reading the main vehicle state.

Every tuple must have an initial vehicle distance of at least 2 m and a path long enough for the derived red speed. The reference trajectory, action projector and shield all use the same lateral-acceleration limit, \(0.6g=5.88399\ \mathrm{m/s^2}\). For the 3.5 m lane shift, the reset check is

\[
L_{\rm cutin}\ge 1.05v_{\rm red}\sqrt{\frac{3.5\times5.7735026919}{5.88399}}.
\]

This prevents a prescribed speed and path length from demanding excessive lateral acceleration. A short path can therefore be valid at a low red speed and invalid at a high red speed.

Before every rollout, the executor converts the declared initial vehicle distance to a reference-point separation using both vehicle lengths. After reset it measures the actual distance and speeds from MetaDrive and aborts on any disagreement. The lateral path must lie fully in the legal merge window. The common tracker, action projector and shield limit steering rate, lateral acceleration, acceleration, braking and jerk.

The test completion position is derived, not sampled: the ego must pass `cutin_start_s_m + cutin_path_length_m + 30 m` while remaining in the target lane. The 30 m follow-through confirms the post-merge state while both vehicles remain on the legal corridor. A tuple that leaves no such position on the route is rejected at reset; its declared five parameters are never silently changed.

Collisions and near misses after reset are valid test outcomes. Initial overlap, invalid speed/path combinations, out-of-corridor lane changes and unbounded commands are invalid and excluded from learning.

Continuous vulnerability response is a learning signal derived from already-recorded challenge-phase telemetry. It does not alter collision, near-miss, validity, route, completion, or physical execution semantics.
