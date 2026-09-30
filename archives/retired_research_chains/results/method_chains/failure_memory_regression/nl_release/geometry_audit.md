# Standstill-gap geometry audit

The installed highway-env `RoadObject.lane_distance_to` returns the
difference between projected **vehicle center** coordinates. `Vehicle.LENGTH`
is 5.0 m. The first two NL chains used `desired_gap=5.0` at V0, leaving
approximately zero bumper clearance when the IDM vehicle settles behind a
stationary 5.0 m target.

In the independently measured calibration bank, all 25 S02
`fbrt_cutout_static` cases collided with the static actor under V0 and each
efficiency candidate. A diagnostic replay of the frozen
`nl:calibration:S02:c0:04:04` case under V0 showed the center distance
decrease from 15.4 m at 8.0 s to 5.3 m at 11.0 s, while ego speed decreased
from 7.3 to 0.7 m/s. The static-actor collision was logged at 11.5 s.
This supports a geometry/standstill-gap explanation for that scenario family's
all-fail result; it does not retroactively invalidate the measured collision
labels or justify dropping S02 from those banks.

The separate `nl3_v0→nl3_v1→nl3_v2` development chain therefore uses an
8.0 m baseline center gap and a 6.0 m V1 center gap, keeping at least nominal
bumper clearance. Its manifest and results are separate from the frozen
original confirmation bank. Whether this change yields a useful bidirectional
test task remains an empirical question.

On the same diagnostic S02 case and seed, the measured V0 collided at 11.5 s,
while `nl3_v0`, `nl3_v1`, and `nl3_v2` completed without collision; their final
center separations were 7.85, 6.19, and 5.94 m. Small negative final speeds
were observed for `nl3_v0` and `nl3_v2` (−0.23 and −0.26 m/s), so these
collision-free outcomes must not be described as fully correct stopping or
complete driving-task repair without a separate functional endpoint.
