# CoRe-Mine cached experiment

The study uses five existing frozen response banks: two for development and three for validation. Each target retains 500 scenarios from the five two-vehicle functions; `passing_cutin` is excluded as its own three-vehicle setting. Selectors receive only source data and outcomes revealed by the cache oracle.

Primary endpoint: CVS@20, a fixed 4x4 gap × relative-speed grid per function. A collision contributes 1 and a near miss 0.5; repeats do not increase archive value.

## Validation summary

| Method | CVS@20 | Collision@20 | Severity@20 | CVS@50 | Cost (s/campaign) |
| --- | ---: | ---: | ---: | ---: | ---: |
| FPS-Risk | 7.657 | 19.944 | 19.972 | 10.870 | 0.035 |
| FPS-Severity | 5.657 | 19.981 | 19.991 | 10.880 | 0.036 |
| FPS-Balanced | 6.935 | 19.296 | 19.565 | 9.926 | 0.041 |
| FPS-Marginal | 10.880 | 16.167 | 17.009 | 13.750 | 0.283 |
| MeanResidual-Marginal | 9.963 | 15.500 | 16.861 | 12.574 | 0.324 |
| CoRe-Marginal | 10.667 | 15.685 | 16.648 | 13.935 | 0.334 |
| TargetOnlyGP-Marginal | 2.250 | 2.704 | 3.361 | 3.657 | 0.327 |
| Random | 2.999 | 2.806 | 3.405 | 6.262 | 0.000 |

## Matched comparisons

The strongest matched marginal baseline is **FPS-Marginal** (CVS@20=10.880); CoRe-Marginal is 10.667 (difference -0.213, -2.0%) with severity 16.648 versus 17.009.
Paired seed→target bootstrap for the CVS difference: [-0.463, +0.074] (n=54); individual episodes were not treated as independent samples.
Ablations: NoComposition CVS@20=8.861, NoNull CVS@20=10.704; FPS-Marginal is the pre-registered NoResidual comparator.

The second pre-listed development check tested FPS-Marginal λ={0, 0.10, 0.30}. None retained 90% of FPS-Severity's SeveritySum@20; results are retained in `objective_trials.csv`.

## Decision

Outcome: **no reliable gain in this round**. All development trials are retained; validation used the frozen configuration. New physical confirmation was not authorized: CoRe lost to its strongest matched baseline and the simpler coverage rule did not meet the pre-set severity guardrail.
