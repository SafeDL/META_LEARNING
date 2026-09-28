# NL-IDM release-chain experiment

This directory is independent of the frozen v2/v3 regression banks. The same
`ProfiledIDMVehicle` and physical scenario manifest are used for `nl_v0`,
`nl_v1`, and `nl_v2`. V1 calibrates normal following for efficiency; V2 inherits
V1 and adds only a short-TTC safeguard. The release profiles are in
`sut_algorithms/highway_env/nl_release.py`.

The `development/` run uses one context per family and a 5×5 grid. It is for
checking the protocol and diagnosing model behavior, not confirmatory evidence.
The `confirmation/` run was frozen at three contexts per family and a 7×7 grid
before its target outcomes were read. The grid is finite; it is not an estimate
of real-world crash probability.

From the repository root, with the `metadrive` environment active:

```powershell
python -m methods.failure_memory_regression.bidirectional freeze --root results/method_chains/failure_memory_regression/nl_release/confirmation --split confirmation --resolution 7
python -m methods.failure_memory_regression.bidirectional measure --root results/method_chains/failure_memory_regression/nl_release/confirmation
python -m methods.failure_memory_regression.bidirectional evaluate --root results/method_chains/failure_memory_regression/nl_release/confirmation
```

`measure` resumes after interruption and checks the scenario and build
fingerprints of existing rows. `evaluate` requires a complete three-version
bank. The selector gets complete parent labels and queries target labels only
through `TargetOracle`. Unknown target outcomes remain eligible until queried
and consume budget. Both directions share a target-response posterior for the
adaptive methods; regression scores are target failure probabilities, while
improvement scores are their complement. Each method uses the same fixed
alternating direction schedule and no-repeat budget.

The current predictor uses a strongly regularized logistic MAP fit with a
plug-in probability; it does not integrate posterior uncertainty. The exact
feature and prior settings for each run are in `selector_config.json`.

`transition_truth.csv` is evaluator-only. The report keeps regression and
improvement separate and records NA where a direction has no true changes.
Random repeats reuse the same physical episodes and are not independent
physical trials. The first release chain is a protocol and mechanism test;
the objectives document requires a later PPO weight-inheritance chain before
making claims about learning-policy updates.

The completed 441-scene confirmation bank contains no regressions. It has
3 improvements for V0→V1 and 8 for V1→V2. At 20 improvement queries, the
directed residual finds 3/3 and 6/8, while static parent risk finds 3/3 and
7/8; static risk also has higher early-discovery area on both transitions.
This is a negative result for the proposed advantage, not evidence that the
new method wins. The complete record remains in `confirmation/`.

Subsequent development is recorded separately. `calibration/` measured V0
and three efficiency settings on 75 new scenes (300 physical episodes), then
selected `nl_eff_c2` by the frozen progress/collision rule, without using
tester results. `chain2_development/` (5×5) contained no changes. Its finer
`chain2_development_fine/` (7×7) contained one regression and four
improvements. The exploratory context-specific Laplace residual tied static
risk on the sole regression and found all four improvements earlier
(area 0.352 versus 0.314); the sample is too sparse for a conclusion.
`chain2_confirmation/` is the completed independent frozen bank for that
candidate. It contains 2 regressions on V0→V1 and 9 improvements on V1→V2.
The context-specific Laplace selector found all 9 improvements within 20
queries (early area 0.452 versus 0.238 for static risk), but found neither
regression; the original directed selector found one. Thus the second
confirmation supports an improvement-direction gain, not bidirectional
superiority. See its frozen protocol, full response bank, and `exploratory/`
replay. The family-level paired analysis has only three family clusters and
cannot establish statistical significance.

`chain2_confirmation_ucb/` freezes a later UCB revision and a new three-context
bank. Its `method_selection.md` records the revision and the earlier negative
results before the new physical outcomes were measured. This is a separate
sequential confirmation, not a reanalysis of the previous bank. The completed
bank contains six V0→V1 regressions and 21 V1→V2 improvements. UCB ties static
risk at 1/6 regressions, but finds 16/21 improvements versus static risk's
12/21 (early area 0.905 versus 0.648). It therefore remains a negative result
for the bidirectional claim. Details and the frozen selector source snapshot
are in `chain2_confirmation_ucb/confirmation_findings.md`.

`margin_revision_development.md` explains the next parent-TTC context sweep
and observed-regression frontier. Old-bank replays there are post-hoc.
`chain2_confirmation_margin/` is its separate, frozen 441-scene confirmation
bank. The completed bank has 3 regressions and 17 improvements, all in S08.
The new method finds 3/3 and 13/17 at 20 direction queries, versus static
risk's 0/3 and 6/17 and ordinary coordinate residual's 0/3 and 7/17. This
is a descriptive bidirectional gain on the finite bank; with one changing
scenario family it is not statistical or cross-family superiority. The
coordinate method with the same margin coverage and frontier also finds 3/3
regressions, while only 7/17 improvements; the directed feature mainly
contributes to improvement discovery. See
`chain2_confirmation_margin/confirmation_findings.md` and its full replay.

The separate `chain3_development/` tests a controller geometry correction:
the original 5 m IDM standstill gap was a center-to-center distance for
approximately 5 m vehicles, leaving no bumper margin behind the static actor.
See [`geometry_audit.md`](geometry_audit.md). The corrected baseline makes all
49 S02 development scenes collision-free; this chain has seven improvements
and no regressions. The exploratory model's improvement area is 0.600 versus
0.438 for static risk, but this remains development evidence only.
