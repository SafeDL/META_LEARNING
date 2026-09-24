# Longitudinal-controller revision pilot (development only)

The discrete VI configuration pilot at seed 20291118 was negative and remains
archived. This is a separate, physics-motivated hypothesis: a software
revision that delays detection of a newly encountered leader or weakens
maximum braking may introduce failures near the old controller's safety
boundary. It is a **seeded parameter-regression study**, not evidence of a
real deployed software release. All builds use the same profile-controlled
IDM implementation, 20 Hz low-level actions, 20 Hz physics, and no ego lane
changes. This avoids the 5 Hz external-control confound but narrows the ODD
to longitudinal/cut-in interactions.

Freeze the controller builds before results:

| Build | Reaction delay | Maximum braking | Other parameters |
|---|---:|---:|---|
| `idm_ref` | 0.0 s | 5.0 m/s2 | IDM desired time 1.5 s, gap 5 m, acceleration 3 m/s2, target speed 27 m/s |
| `idm_delay07` | 0.7 s | 5.0 m/s2 | identical to reference |
| `idm_brake3` | 0.0 s | 3.0 m/s2 | identical to reference |
| `idm_delay07_brake3` | 0.7 s | 3.0 m/s2 | identical to reference |

At development seed 20291202, generate 16 scrambled Sobol candidates in each
of five modes with four continuous dimensions: initial gap, relative speed,
timing, and intensity. Fix the physical parameter bounds (m; m/s):
fast intrusion [18,45],[-8,-2]; cut-in braking [18,50],[-7,-1]; lead braking
[20,50],[-7,-1]; stop-and-go [18,50],[-6,-1]; slow-lead following
[20,50],[-8,-2]. Timing/intensity map [0.05,0.95]. Every build executes the
same 80 scenarios with seed `20291202 + scenario_index`.

Reference-safe means the ego did not collide, TTC never fell below 1.5 s,
vehicle-outline clearance never fell below 1 m, and the episode completed.
Report the eligible count and modes, plus each target build's new failures,
ego collisions, and failure modes in this fixed subset. Do not select a
favorable build or change bounds after seeing the pilot. A later formal
B=50 study is warranted only if this pilot shows sufficient headroom in at
least two distinct builds and modes; it must then freeze new independent
seeds, direct baselines, and all target builds including zero-opportunity
cases. This pilot cannot establish novelty or method effectiveness.
