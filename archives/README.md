# Archived baselines

This directory contains frozen comparison implementations that are not part of
the active method. Current simulator code stays in `metadrive_benchmark/` and
`highway_env_benchmark/`; current composed methods stay in `method_chains/`.

- `pearl_learning/`: merge-only PEARL baseline and its contract tests.
- `sac_scenario_mining/`: legacy SAC scenario-mining baseline.

Do not add new method code to this directory. Run archived PEARL tests from the
repository root with:

```powershell
conda run -n metadrive python -m pytest archives/pearl_learning/tests -q
```

Archived source code is kept separate from results. The current `results/` tree
does not contain a `pearl_learning/` result archive; the PEARL commands above
test the archived package. Its reconstruction commands do not refer to
retained formal outputs.
