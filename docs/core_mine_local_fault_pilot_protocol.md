# Frozen 20 Hz local-regression pilot (opportunity test only)

The global IDM parameter-change studies showed that historical TTC margin
ranking is strong and that target-feedback residuals do not outperform a
source-only mode-quantile baseline. This pilot tests a different, explicitly
local failure mechanism before designing any selector or inspecting target
outcomes. These are **seeded software-fault surrogates**, not actual ADS
releases, sensor simulations, or evidence of real-world safety.

Use the unchanged `idm_ref` build and the 80 scenarios from development seed
20291202 in `docs/core_mine_idm_revision_pilot_protocol.md`. Its 80 source
executions were completed and event-free. All builds receive the same
scenario and simulator seed `20291202 + scenario_index`. Internal ego action
and physics are 20 Hz; ego never changes lane. Replay all 80 on each of the
following three predeclared mutants, no target-dependent filtering:

1. `merge_blind06`: when a front vehicle is first observed while its lateral
   center is at least 0.25 m away from the ego center, ignore that front
   vehicle for 0.60 s. This is a delayed cut-in association surrogate.
2. `merge_brake2`: under the same first-observation trigger, limit commanded
   deceleration to 2 m/s² for 0.80 s. This is a temporary cut-in braking-limit
   surrogate.
3. `slow_front_brake2`: whenever an observed front vehicle travels below
   18 m/s, limit commanded deceleration to 2 m/s². This is a speed-conditional
   braking-limit surrogate.

All other IDM parameters remain `idm_ref`. The trigger reads only observable
vehicle state, never scenario mode, scenario controls, or future trajectory.
Record activation counts, ego collision, physical near miss (TTC < 1.5 s or
polygon clearance < 1 m without collision), event-free completion, and
new-failure counts by mode for *all three* mutants. The event labels must use
the same semantics as the earlier IDM study. This is an opportunity and
mechanism pilot only: **no B=50 method result or novelty claim** can be made
from these 80 cases. If fewer than two mutants have new failures in at least
two modes, stop this family; do not tune the thresholds or bounds after seeing
the pilot. If the gate passes, freeze independent development and validation
seeds and compare every fixed mutant against source-only margin, source-only
mode quantile, direct published-prioritizer approximations/implementations,
target-only learning, and random selection at B=50.
