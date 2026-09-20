# ScenarioFuzz-H replication

This directory is a self-contained adaptation of *Dance of the ADS* to the
repository's real `highway-env` Cut-in harness. It preserves local topology
seeds, two-stage mutation, a node-GAT plus line-graph-edge-GAT SEM, threshold
filtering, real simulation feedback, frequency-aware scheduling, source-only
history isolation, ablations, and collision trajectory post-analysis.

The canonical suite runs six target-specific leave-one-SUT-out models, three
algorithm seeds, both candidate-count protocols, history-size growth, and
execution-preknown RBF/MLP/random-forest controls. It produces highway-env
analogues of paper Figures 6, 7, 10, 11, 12, 14, 15 and Tables 2 and 3. The
earlier nondeterministic model, compact single-target trial, and intermediate
pool directories were removed after this deterministic suite passed.

It intentionally does not add weather, colors, pedestrians, traffic lights,
or other variables that have no physical/perception effect here. The driving
score is explicitly the highway adaptation `1 - vulnerability`. Results are
not claims about the paper's CARLA systems or its 60.3%/103%/54/58 figures.

Run the tests:

```powershell
conda run -n metadrive python -m pytest replications/scenariofuzz_highway_env/tests -q
```

Run the complete paper-aligned, multi-system suite (idempotent at completed
model/campaign boundaries):

```powershell
conda run -n metadrive python -m replications.scenariofuzz_highway_env.scenariofuzz.paper_experiments `
  --pool-config replications/scenariofuzz_highway_env/configs/pool.yaml `
  --online-config replications/scenariofuzz_highway_env/configs/online.yaml `
  --output results/highway_replications/scenariofuzz `
  --repeats 3
```

Regenerate only the paper-aligned tables, figures, report, and validation from
the saved CSV/JSONL/NPZ data:

```powershell
conda run -n metadrive python -m replications.scenariofuzz_highway_env.scenariofuzz.paper_figures `
  --suite-dir results/highway_replications/scenariofuzz
```

The cross-method evaluator applies each frozen LOSO SEM to the shared response
bank and names that control `ScenarioFuzz-SEM-Pool-H`. It is explicitly a
fixed-pool ranking and is not relabeled as online fuzzing.
