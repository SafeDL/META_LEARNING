# Archived baselines

This directory contains frozen comparison implementations that are not part of
the active method. The current map-aware framework stays in `mvr/`.

- `pearl_learning/`: merge-only PEARL baseline and its contract tests.
- `sac_scenario_mining/`: legacy SAC scenario-mining baseline.

Keep new method code and experiments in `mvr/`. Run archived PEARL
tests from the repository root with:

```powershell
conda run -n metadrive python -m pytest archives/pearl_learning/tests -q
```

Historical outputs remain under `results/` and are not moved by this archive.
