# Highway-env paper replications

This directory contains independent adaptations of representative testing
methods. Every implementation uses the repository's `highway_env_benchmark`
simulator, SUT profiles, and response-bank schema. Method-specific logic stays
inside its package; datasets and final artifacts stay under
`results/highway_replications/`.

## Reproduction contract

1. Preserve the source paper's information boundary, update rule, baselines,
   ablations, and conclusion-bearing measurements whenever highway-env
   supports them. Name every unsupported feature as a deviation.
2. Use `benchmark.yaml` for cross-method data and budgets. The shared bank has
   three interaction modes, 64 Sobol candidates per mode, and six SUTs: 192
   scenarios and 1,152 actual simulator responses.
3. Compare only like tasks. Failure discovery uses collision precision and
   recall at budgets 5/10/20. Performance estimation uses collision-rate error
   at the same budgets. Paper-specific metrics remain in each method report.
4. Keep source packages free of generated results, caches, dated folders, and
   superseded attempts. Each method has one canonical result directory.

## Layout

```text
replications/
  benchmark.py                 shared bank builder and evaluator
  benchmark.yaml               shared data/evaluation contract
  adate_highway_env/           AdaTE adaptation
  detour_highway_env/          DETOUR adaptation
  fst_highway_env/             FST similarity adaptation
  scenariofuzz_highway_env/    ScenarioFuzz adaptation
  highway_sut_selection/       heterogeneous driving-policy qualification

results/highway_replications/
  shared/                      common response bank and manifest
  adate/                       AdaTE paper-specific and shared-pool results
  detour/                      DETOUR results on the shared pool
  fst/                         FST results on the shared pool
  scenariofuzz/                deterministic paper-aligned suite
  evaluation/                  cross-replication records, figures, and report
  sut_selection/               retained SUT bank and screening evidence
```

## Run

```powershell
conda run -n metadrive python -m replications.benchmark build
conda run -n metadrive python -m replications.benchmark evaluate
conda run -n metadrive python -m pytest replications -q -p no:cacheprovider
```

Each method README contains its exact reconstruction command.
