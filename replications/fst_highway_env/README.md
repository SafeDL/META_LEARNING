# FST-Similarity-H on highway-env

This directory is a paper-based reproduction of Li et al., *Few-Shot Testing
of Autonomous Vehicles With Scenario Similarity Learning* (IEEE T-ITS, 2025).
It uses the repository's real `CutInEnv` response bank and implements the
paper's learned cross-attention, query-axis normalization, `w = S p`, and
source-surrogate minimax estimation objective.

For numerical stability at an exact query/self-key match, the configured
inverse-distance logit is evaluated as `1/sqrt(||q-k||^2 + epsilon^2)`. This
has the intended self-logit `1/epsilon` while avoiding an undefined derivative
at zero distance; `epsilon` is an engineering completion not reported by the
paper.

The paper optimizes continuous scenario coordinates with gradient descent.
The P0 reproduction works on the finite executable bank and therefore uses a
fully disclosed discrete single-swap optimizer. It recomputes the whole
attention matrix and all weights after every proposal. No target response is
used until every set and weight vector has been frozen and hashed.

The main reference distribution is uniform over the 192 frozen candidates.
It is a benchmark distribution, not a naturalistic-driving exposure model and
not a real-world crash rate.

## Data coverage

The canonical run uses the complete shared domain: 192 scenarios balanced over
`fast_intrusion`, `cutin_braking`, and `lead_braking`. Training reads all
SUT-A/B/C collision responses; early stopping reads all SUT-D responses; SUT-E/F
labels remain hidden until each n=5/10/20 set and weight vector has frozen.
The discrete single-swap search evaluates every legal replacement at each
visited position, while the combinatorial set space is sampled rather than
exhaustively enumerated.

The 50 repetitions are deterministic response-bank algorithm replays. They
measure initialization and set-design variation and are not 50 independent AVs
or 50 newly simulated response banks.

## Run

```powershell
conda run -n metadrive python -m pytest replications/fst_highway_env/tests -q
conda run -n metadrive python -m replications.fst_highway_env.fst.train `
  --config replications/fst_highway_env/configs/similarity.yaml `
  --output results/highway_replications/fst
conda run -n metadrive python -m replications.fst_highway_env.fst.experiment `
  --run-dir results/highway_replications/fst `
  --budgets 5 10 20
conda run -n metadrive python -m replications.fst_highway_env.fst.visualize `
  --run-dir results/highway_replications/fst
```

`target_estimates.csv` contains held-out SUT-E/SUT-F performance-estimation
results for CMC, uniform sampling, importance sampling, handcrafted similarity,
learned-similarity random sets, learned-similarity optimized sets, and a
no-similarity optimized control. The PNG figures are rebuilt from persisted
CSV/NPZ/JSON artifacts. Both GIFs contain actual `CutInEnv(render_mode='rgb_array')`
frames rather than hand-drawn trajectories.
