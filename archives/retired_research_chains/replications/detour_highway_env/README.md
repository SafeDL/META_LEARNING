# DETOUR-Scenario-H replication

This package adapts DETOUR's hierarchy, retrieval, branch sampling, and early
stopping rules to the repository's highway-env scenario bank. It imports the
shared simulator response schema and keeps target labels outside selection;
target outcomes are used only by the offline evaluator.

The sole canonical artifact directory is
`results/highway_replications/detour/`. It contains the frozen D1/D2 protocols,
20-seed controls, complete 36-cell stopping grid, selection traces, figures,
GIFs, manifest, deviations, and report.

```powershell
conda run -n metadrive python -m replications.detour_highway_env.detour.experiment `
  --config replications/detour_highway_env/configs/detour.yaml `
  --bank results/highway_replications/shared/response_bank.npz `
  --output results/highway_replications/detour
```

Straight-road input features replace the paper's curvature-distance road
features; collision replaces lane-departure failure. These declared deviations
are reported in the result archive rather than presented as original-paper
numbers.
