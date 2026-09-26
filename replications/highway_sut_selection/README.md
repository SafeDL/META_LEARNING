# Highway-env SUT selection

This package qualifies heterogeneous driving policies against the shared
`highway_sim_env` CutIn simulator. The simulator extension lives in
`highway_sim_env/envs/external_cutin.py`; retained policy implementations and
their registry live in `sut_algorithms/highway_env/`.

The active set contains IDM+MOBIL, VI-TTC, MCTS-CV, and the verified PPO-ECE
checkpoint. DQN-ECE and Double DQN-ECE were removed after screening. Their
rejection reasons and checkpoint metadata remain in the formal report.

```powershell
conda run -n metadrive python -m pytest replications/highway_sut_selection/tests -q -p no:cacheprovider
conda run -n metadrive python -m replications.highway_sut_selection.cli fetch
conda run -n metadrive python -m replications.highway_sut_selection.cli qualify
conda run -n metadrive python -m replications.highway_sut_selection.cli common
conda run -n metadrive python -m replications.highway_sut_selection.cli audit
```

Formal results are written to `results/highway_replications/sut_selection/`.
External checkpoints are an on-demand cache: `fetch` downloads and verifies the
retained PPO checkpoint before a rerun, and the cache may be deleted afterward.
