# PPO weight-inheritance release chain

`study_plan.json` freezes three 75-scene training increments, a separate
75-scene validation manifest, and a separate 225-scene final confirmation
manifest before any continued PPO training. All physical context tuples are
disjoint. The existing 5×6 Kinematics / Discrete(5) checkpoint is the base.
`FBRTTrainingEnv` accepts the current PPO action at 5 Hz and advances exactly
four 20 Hz frames through the same physical `_advance` as evaluation. A
checkpoint-driven test verifies identical training/evaluation traces and
actions over consecutive decisions. The training reward is fixed progress
minus an ego-collision penalty.

V0 continued the imported checkpoint for 2048 decision steps; V1 inherited
V0 for 4096 steps on half new and half old **resampled scenarios**; V2
inherited V1 for 4096 steps on half second-new and one quarter of each older
scenario set. All rollouts were newly collected under the current policy.
The last checkpoint after each fixed budget was selected, without testing
which checkpoint helps the FBRT selector. The three stages used 10,240 PPO
decision steps and 40,878 actual physics steps; a separate non-release
preflight used 64/255 more. `training/*.json` records source/destination
checkpoint and policy-weight hashes, exact physical counts, and scenario
pool hashes. The source-weight hash of V1 equals V0's output, and V2's input
equals V1's output. The two later checkpoints change policy weights.

The 75-case diagnostic validation bank has 225 valid paired episodes. V0,
V1, V2 have 35, 31, 31 ego collisions respectively. V0→V1 has four
improvements and zero regressions; V1→V2 has no collision-label changes.
At 20 improvement queries the proposed margin/frontier selector finds 0/4,
while static risk finds 3/4. This is a negative validation result for
transfer to the PPO SUT; it was **not** used to change checkpoints, method,
or final manifest. See `validation/diagnostic_evaluation/` for all methods.

`confirmation/` has a frozen three-version, 12-method protocol and a source
snapshot made before its target responses were measured. The completed
225-scene bank has 5 regressions/10 improvements on V0→V1 and 5/12 on V1→V2.
The proposed margin/frontier method finds 5/5 and 5/5 regressions, but only
0/10 and 2/12 improvements. It does **not** show bidirectional superiority on
PPO weight updates. See `confirmation/confirmation_findings.md`; this result
must be interpreted separately from the non-learning NL-IDM release chain.
