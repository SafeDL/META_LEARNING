# Highway-env replication benchmark

The shared domain contains 192 scenarios (fast_intrusion: 64, cutin_braking: 64, lead_braking: 64) and 1152 actual simulator responses across six SUTs.

Failure-discovery methods are compared only under the shared candidate pool. AdaTE includes static and adaptive variants; DETOUR LOSO and the ScenarioFuzz fixed-pool control are static. DETOUR's within-SUT D1 protocol is retained in the records but excluded from the primary table because it observes target-SUT history. FST remains in the separate performance-estimation task, so its MAE is not mixed with discovery recall.

## Collision recall at B=20

| Method | Protocol | Mean recall | SD |
| --- | --- | ---: | ---: |
| AdaTE-Mixture-H-staticK4 | shared_pool_adaptive | 0.616 | 0.289 |
| Uniform-Mixture-H | shared_pool_static | 0.616 | 0.289 |
| AdaTE-Mixture-H-sequential | shared_pool_adaptive | 0.583 | 0.255 |
| ScenarioFuzz-SEM-Pool-H | shared_pool_static | 0.522 | 0.162 |
| DETOUR-static | shared_pool_static | 0.360 | 0.091 |
| Nearest-Failure-global | shared_pool_static | 0.326 | 0.076 |
| Random | shared_pool_static | 0.103 | 0.062 |

## Collision-rate estimation MAE at B=20

| Method | Mean MAE | SD |
| --- | ---: | ---: |
| NoSimilarity-Optimized | 0.0332 | 0.0093 |
| FST-RandomSet | 0.0370 | 0.0314 |
| Handcrafted-Similarity | 0.0401 | 0.0298 |
| FST-Similarity-H | 0.0517 | 0.0488 |
| Uniform | 0.0595 | 0.0109 |
| IS | 0.0628 | 0.0571 |
| CMC | 0.0651 | 0.0188 |

The shared benchmark measures project utility on a common highway-env domain. Paper-specific conclusions, ablations, and deviations remain in each method's own final report.
