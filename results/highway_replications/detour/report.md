# DETOUR-Scenario-H static replication

- implementation_validated: see the DETOUR test suite.
- mechanism_observed: D1 frozen within-SUT split, D2 LOSO hierarchy traces, per-node count snapshots, branch sampling, road compression, and the full 36-cell stop grid were generated.
- project_utility_observed: mean collision failure ratio at B=20: {'DETOUR-static': 0.6120833333333332, 'Random-within-SUT': 0.19583333333333333, 'Nearest-Failure-global': 0.5333333333333333, 'DETOUR-within-SUT-static': 0.5020833333333334, 'Random': 0.18875, 'Nearest-Failure-global-within-SUT': 0.575}.
- history execution costs: `history_costs.csv` records D1/D2 history and candidate counts by target.
- safe-neighbor early stops in 36-cell grid: 76/4320.
- original_numbers_reproduced: no - this is an input-feature highway-env adaptation, not the paper's road-suite experiment.

Target outcomes were excluded from selection and used only for offline evaluation.
