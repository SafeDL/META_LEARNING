# PPO weight-inheritance confirmation protocol

The 225-case confirmation manifest was written and hashed in
`../study_plan.json` before V0/V1/V2 training. The three checkpoints were
then continued in sequence with fixed decision-step budgets; the last
checkpoint after each budget was selected without reading validation or final
transition labels. The complete three-build protocol here was frozen before
any PPO paired outcome was measured. Validation uses a separate 75-case
manifest and is diagnostic only.

The selector is exactly the margin/frontier revision prospectively tested on
the NL-IDM fourth bank. Its primary comparison on PPO is
`directed_margin_frontier` versus `static_risk`, `static_boundary`,
`coordinate_residual`, `static_margin_coverage`, and
`coordinate_margin_frontier`. The previously developed Laplace and UCB
variants are also reported. All 12 frozen methods use the same parent bank,
target oracle, 20 regression + 20 improvement query budget, alternation,
tie rule, and one target execution per query. The two directions and both
weight updates are reported separately; absent true changes are NA.

The PPO result is an independent **SUT update mechanism**, but the method and
scenario families were developed using earlier NL-IDM banks. It is not an
independent method-development lineage or proof of population-level driving
safety. Prior NL confirmation successes and failures remain separate.

Frozen source SHA-256: selector
`71f57997208a6e279d7379ae9823ce7126e5e993970a26f352456057ad04f7bf`;
PPO deterministic loader
`c7adb39df79065f12880874dea94a4c0b4d9ab0b782d350ce397e63a2e880a2e`;
physical runner
`ce4667d58ddf025c3204dabbe892919dcfcabd31fa819bd2adce376c4d099274`.
