# Highway-env SUT compatibility record

All common-scenario executions use `ExternalCutInEnv` (`external_ego_v1`): a
five-action `DiscreteMetaAction` (`LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`,
`SLOWER`), 20 Hz physics, 5 Hz policy decisions, and a scheduled background
traffic vehicle.  Ego actions are dispatched before background actions; ego is
not passed to `road.act()`, preventing an automatic controller from replacing
the external command. `ego_collision` and `background_collision` are recorded
separately.

| SUT | Native contract | Common-scene adapter | Outcome |
|---|---|---|---|
| IDM+MOBIL | highway-env `IDMVehicle` | Native ego; its MOBIL lane policy runs once per physics frame | retained |
| VI-TTC | highway-env TTC finite-MDP model | Bellman value iteration over the current 6 s TTC grid | retained |
| MCTS-CV | Explicit constant-velocity prediction model | 24 rollouts/action, 6-decision horizon; no scheduled future traffic data | retained |
| PPO-ECE | `highway-fast-v0`; upstream five-by-six `x,y,vx,vy,sin_h,cos_h` observation and source clock/config | Same observation encoding; frozen SB3 prediction | retained |
| DQN-ECE | Same source contract | Not admitted: SB3 2.3.0 cannot load the SB3 2.7.0 checkpoint (20/20 errors) | rejected |
| DDQN-ECE | Same source contract; tensor-only 30-to-256-to-256-to-5 network | Successfully adapted, but screened out after 33/96 ego collisions | rejected |

The retained PPO checkpoint is not stored in the working tree. Run
`python -m replications.highway_sut_selection.cli fetch` to download and verify
it before a PPO rerun; the resulting `external_assets/` cache is disposable.
Rejected DQN/DDQN runtime code and checkpoint files were deleted after their
screening evidence was written. No policy was trained or fine-tuned.
