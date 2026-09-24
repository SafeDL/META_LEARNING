# AdaTE highway-env replication

This package adapts AdaTE to the repository's real highway-env harness. It
implements response-mixture adaptation, DenseRL source-model transfer,
target-hidden updates, simplex QP, importance sampling, and diagnostic replay.
The simulator and SUT definitions come from `highway_env_benchmark`; no environment
or controller copy is maintained here.

Canonical artifacts are under `results/highway_replications/adate/`:

- `shared_pool/`: common three-mode, six-SUT failure-discovery comparison;
- `mixture/`: original A0 mechanism result;
- `dense/`: two-seed, three-target A1 confirmation;
- `rare_event/`: passing-domain calibration and screening;
- `report.md`: conclusions and adaptation boundaries.

The qualitative pilot directories were removed after the cross-target and
rare-event experiments superseded them.

```powershell
conda run -n metadrive python -m replications.adate_highway_env.adate.experiment `
  --stage mixture `
  --config replications/adate_highway_env/configs/shared_pool.yaml `
  --bank results/highway_replications/shared/response_bank.npz `
  --output results/highway_replications/adate/shared_pool
```

The DenseRL confirmation is rebuilt with `configs/dense_cross_target.yaml` and
written to `results/highway_replications/adate/dense`. This is a
mechanism-faithful highway-env adaptation, not a claim to reproduce the paper's
overtaking data, NDE/NADE distribution, or original numeric table. Detailed
differences are in `adate/deviations.md`.

