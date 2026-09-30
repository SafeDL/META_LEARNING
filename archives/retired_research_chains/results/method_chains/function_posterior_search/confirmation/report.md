# Independent confirmation of Function Posterior Search

This is a frozen physical confirmation experiment. It uses five previously unused scenario seeds, 600 scenarios per seed, six functions, and 18 targets arranged as a 3 (source coverage) x 3 (functional heterogeneity) x 2 factorial. No target outcome was used to define a controller, scenario, source-only hyperparameter, or baseline.

Primary endpoint: target-macro critical-event Recall@50 for adaptive-support posterior search versus Function-Conditioned Mining. Random uses 20 repeats; all other methods are deterministic. Collision OR near miss is the critical-event oracle.

## Overall results

| Method | B=10 | B=20 | B=30 | B=50 |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.0164 | 0.0330 | 0.0499 | 0.0834 |
| DETOUR | 0.0699 | 0.1416 | 0.1991 | 0.3064 |
| Mining | 0.0210 | 0.1061 | 0.1913 | 0.3606 |
| Mining-DETOUR | 0.0215 | 0.1067 | 0.1920 | 0.3612 |
| AdaTE Global | 0.0851 | 0.1698 | 0.2510 | 0.4143 |
| Function-Conditioned Mining | 0.0825 | 0.1677 | 0.2527 | 0.4183 |
| Posterior Search (adaptive support) | 0.0853 | 0.1704 | 0.2557 | 0.4260 |

## Recall@50 by source coverage

| Coverage | Function-Conditioned Mining | Posterior search | Difference |
| --- | ---: | ---: | ---: |
| exact | 0.3645 | 0.3683 | +0.0038 |
| interpolated | 0.4750 | 0.4906 | +0.0155 |
| unseen | 0.4152 | 0.4192 | +0.0040 |

## Recall@50 by functional heterogeneity

| Heterogeneity | Function-Conditioned Mining | Posterior search | Difference |
| --- | ---: | ---: | ---: |
| global | 0.4587 | 0.4682 | +0.0095 |
| partial | 0.3983 | 0.4037 | +0.0054 |
| full | 0.3978 | 0.4062 | +0.0084 |

## Confirmatory comparison

Adaptive posterior search minus Function-Conditioned Mining at Recall@50: +0.0078, hierarchical 95% bootstrap interval [+0.0046, +0.0116], with 45 wins, 41 ties, and 4 losses over 90 seed-target units.

## Descriptive function audit at Recall@50

| Function | Event prevalence | Function-Conditioned Mining | Posterior search |
| --- | ---: | ---: | ---: |
| fast_intrusion | 20.5% | 0.5124 | 0.4719 |
| cutin_braking | 17.3% | 0.4828 | 0.4714 |
| lead_braking | 16.2% | 0.3206 | 0.4460 |
| stop_and_go | 15.4% | 0.2541 | 0.4341 |
| slow_lead_following | 31.6% | 0.6305 | 0.4440 |
| passing_cutin | 26.2% | 0.2091 | 0.4426 |

## Descriptive scenario-regime audit at Recall@50

| Regime | Event prevalence | Function-Conditioned Mining | Posterior search |
| --- | ---: | ---: | ---: |
| core | 19.6% | 0.4190 | 0.2636 |
| benign | 0.0% | n/a | n/a |
| boundary | 35.7% | 0.3930 | 0.4228 |
| outside | 53.4% | 0.4384 | 0.5852 |

## Event prevalence and interpretation

- exact: mean 155.7 critical events per 600 scenarios (25.9%).
- interpolated: mean 102.2 critical events per 600 scenarios (17.0%).
- unseen: mean 123.9 critical events per 600 scenarios (20.7%).

Exact coverage is the method-favourable mechanism check; interpolated coverage tests model misspecification inside the source hull; unseen coverage is an adverse negative control. Global targets are another negative control. Broad superiority is supported only if the overall paired interval excludes zero without relying solely on the exact or fully heterogeneous cells.

The gain is not uniform across the scenario space. Posterior search is weaker in the core regime and stronger at boundary/outside conditions; it also trades recall among functions. The evidence therefore supports stress-oriented prioritization, not a claim that every function or operating region improves. Overall critical-event prevalence is moderate rather than ultra-rare.

The simulator remains a straight-road highway-env harness with scheduled traffic and does not establish real-world safety or certification. Passing scenarios use their existing three-vehicle implementation; timing and intensity are stored but do not alter that mode's fixed schedule.
