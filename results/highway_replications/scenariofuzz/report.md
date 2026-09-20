# ScenarioFuzz-H paper-aligned multi-system report

## Completion verdict

The highway-env adaptation now covers the paper's executable core, six-system LOSO generalization, repeated discovery experiments, history-size growth, filter baselines, cost accounting, and paper-shaped result figures. It remains an environment adaptation, not a reproduction of the paper's CARLA numbers, 54 manually summarized patterns, 58 bugs, code coverage, weather, perception, or urban-map claims.

## Experimental evidence

- 768 genuine source/audit highway-env episodes in the universal bank.
- Six target-specific LOSO graph SEMs; each uses 640 source episodes and excludes its target SUT.
- 18 complete online campaigns (6 SUTs × 3 seeds), each with four components, two Nm protocols, and 160 target executions.
- Physical execution ledger for this suite: 768 universal-bank episodes + 2880 online target episodes + 180 history-sweep target episodes = 3828 episodes.
- Mean paper-protocol collisions: RMS-H 3.00, 2SMS-H 2.11, RMS+SEM-H 18.94, 2SMS+SEM-H 19.17.
- Full-method collision-count change versus RMS-H: +538.9%; wall-time change: -49.1%. These are measured outcomes, not tuned targets.
- Frozen excluded-target Graph-SEM mean AUPRC 0.903, Brier 0.096, precision 0.821, recall 0.851.

- Frozen filter comparison (mean AUPRC / Brier): RBF-Filter-H 0.887/0.107; MLP-SEM-H 0.788/0.144; RandomForest-Filter-H 0.930/0.078; Graph-SEM-H 0.903/0.096.
- SUT-C history-size discovery means (collisions @20): 100 records=20.00, 300 records=20.00, 510 records=20.00.

## Interpretation

Three algorithm seeds quantify selection randomness; they do not create additional independent SUTs. The six controller profiles provide the cross-system dimension. Rejected candidates remain unlabeled, target audit truth is used only after selection for evaluation, and fixed-pool results remain separate from these online campaigns. A non-graph filter matching or exceeding the graph SEM on this low-dimensional task is a valid negative result about graph necessity, not an implementation failure.

Behavioral coverage is shown instead of code coverage because the Python controller/harness does not expose the paper's instrumented ADS modules. Collision modes and trajectory clusters are interaction patterns, not distinct software bugs. Collision episodes terminate early, so part of the measured wall-time reduction comes from shorter failed episodes; the CSV retains candidate-generation, SEM-inference, simulation-wall, and simulated-driving-time fields for separate interpretation.

## Reproduction

Regenerate the suite with `python -m replications.scenariofuzz_highway_env.scenariofuzz.paper_experiments ...`; regenerate figures only with `python -m replications.scenariofuzz_highway_env.scenariofuzz.paper_figures --suite-dir ...`.
