# Repository Guidelines

## Project Structure & Module Organization

`mvr/` is the active map-aware MVR implementation; keep its simulator contracts, policy/context modules, training code, and tests together under that package. `archives/pearl_learning/` and `archives/sac_scenario_mining/` are frozen legacy baselines and should not receive new method features. Store design notes in `docs/`, durable experiment artifacts in `results/`, and reusable cross-package helpers in root `tools/`.

## Build, Test, and Development Commands

Use the MetaDrive Conda environment:

```powershell
conda run -n metadrive python -m pytest mvr/tests -q
conda run -n metadrive python -m pytest archives/pearl_learning/tests -q
conda run -n metadrive python -m compileall -q mvr archives/pearl_learning archives/sac_scenario_mining
```

The first command exercises active MVR contracts, including headless MetaDrive fixtures. Run the second before changing shared legacy utilities. Compile after structural edits when a formatter or linter is unavailable.

## Coding Style & Naming Conventions

Write Python 3 with four-space indentation and standard PEP 8 layout. Order imports as standard library, third-party packages, then project modules. Use `PascalCase` for classes, `snake_case` for functions and variables, and `UPPER_CASE` for constants. Prefer small direct functions over compatibility wrappers, fallbacks, or extra indirection. Keep comments for non-obvious constraints only. Do not add optional CLI flags when a script constant or config entry is sufficient.

Delete unused fields, functions, wrappers, caches, and scripts once references and tests confirm they are dead; do not preserve forwarding shims. Never commit `__pycache__`, temporary files, or regenerable large outputs.

## Testing Guidelines

Use `pytest`; name tests `test_<behavior>.py` and focus assertions on public contracts. Add a focused unit test for every new schema or policy path, plus a headless MetaDrive test when changing physical scenario application. Preserve fixed-budget accounting and do not claim method performance until the relevant G1–G8 gates have executable evidence.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects such as `Implement ...`, `Add ...`, and `Refactor ...`. Keep commits scoped to one coherent change. PRs should explain the affected module and contract, list verification commands/results, link the relevant issue or design note, and include updated artifacts only when they are durable, reproducible evidence.
