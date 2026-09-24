# Independent 20 Hz online confirmation, frozen before target execution

This is a small physical confirmation of the simpler v8 mechanism, not a
replacement for its four-controller, three-seed validation. Use the fixed
scrambled-Sobol v8 scenario generator at new seed **20291007**: five modes x
64 candidates with the same bounds. Physics and external decisions both run
at 20 Hz. The source controllers are `idm_mobil` (internal 20 Hz) and
`mcts_cv` (called at 20 Hz); the never-precomputed target is `vi_ttc` at
20 Hz. These are still different algorithms, **not true software versions**.

Run the two source controllers on all 320 candidates. The only qualification
gate is at least 50 source-safe candidates spanning all five modes, defined
by no ego collision and no near miss in either source, with both source
episodes completing normally. If the gate fails,
stop and report it without changing seed or bounds. Do not inspect any target
response to decide whether the seed is qualified.

After qualification, compare the frozen `HistoryMargin-Residual`,
`HistoryMargin-Static`, `TargetOnly-Residual`, and one seeded `RandomSafe`
order. Every method runs its own 50 physical target episodes sequentially,
even if another method has already tested the same scenario; no cross-method
target outcome cache is read. Each method is restricted to the identical
source-safe eligible set, and its first ten mode-support selections count
toward 50. GP length 0.30, amplitude 0.15, noise 0.05 remain unchanged.

Report new target failures, ego collisions, failure modes, and early discovery
for all four methods. There is only one seed and two source systems, so no
population inference or all-three-source-safe claim is allowed. MCTS-CV's
rollout time-step mismatch also remains; this confirmation checks actual
physical selection, not controller quality.
