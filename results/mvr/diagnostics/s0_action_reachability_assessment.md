# S0 vehicle-residual reachability

Scope: no RL training; 3 validation tasks x 4 fixed initial conditions x 5
constant `[delta_steering, delta_acceleration]` residuals = 60 paired simulator
episodes. Every residual shares the same sampled candidate, x0, and seed with
its base counterpart.

| Family | Valid critical behavior | Direct residual evidence |
| --- | --- | --- |
| Cut-in | Yes: base 2/4; acceleration-brake 2/4 target collisions | Brake lowers median TTC from 3.28 s to 2.68 s and distance from 4.37 m to 3.41 m. |
| Merge | No, 0/20 | Brake lowers median TTC from 15.00 s to 12.59 s, but the fixed x0 set does not enter a critical interaction. |
| Roundabout | No, 0/20 | All tested x0 values stay far from interaction (median distance about 50 m). |

Across the 12 paired initial conditions, acceleration-brake reduces TTC by
0.648 s and separation by 0.450 m on average. Steering effects are small for
this screen.

This verifies the v2 physical action interface and preserves legal Cut-in
critical behavior. It is not a Stage1 acceptance result: Merge and Roundabout
require Logical Scenario/x0 calibration before shared-policy training.
