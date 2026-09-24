# Corrected-geometry v5 research decision

The corrected experiment establishes that the test environment can produce
real ego collisions and physical near misses. It does **not** establish an
incremental advantage for the full CoRe-Mine method over its strongest matched
alternatives at the requested budget of 50 target tests.

The independent qualification seed passed all three prespecified branch and
dispersion checks. Across the nine held-out validation target units, the
mean physical event rate in the 320-candidate pool was 23.58%.

| Method | True events@50 | Ego collisions@50 | CVS@50 | Event modes@50 |
|---|---:|---:|---:|---:|
| MeanResidual-Risk | 44.11 | 28.56 | 16.11 | 4.67 |
| MeanResidual-Marginal | 38.44 | 25.78 | 16.44 | 4.78 |
| CoRe-Risk | 43.67 | 27.00 | 16.06 | 4.56 |
| CoRe-Marginal | 37.11 | 26.22 | 17.28 | 4.67 |
| FPS-Risk | 42.44 | 28.78 | 17.11 | 4.67 |
| FPS-Marginal | 33.11 | 24.00 | 16.11 | 4.67 |
| TargetOnlyGP-Marginal | 21.67 | 12.78 | 10.56 | 3.44 |
| Random | 11.92 | 6.32 | 7.53 | 4.04 |

CoRe-Marginal adds 0.83 CVS over MeanResidual-Marginal, with hierarchical
paired bootstrap interval [-0.56, 2.39]; it finds 1.33 fewer true events on
average. CoRe-Risk finds 0.44 fewer events than MeanResidual-Risk. Hence the
composition component is not supported. MeanResidual-Marginal substantially
improves over target-only and FPS-Marginal on events, but risk-only versions
find more actual events. Marginal coverage has no robust independent gain on
this candidate pool. The source-informed risk policies already spend 44/50
queries on verified events, leaving little discovery headroom.

After validation, a separate development-only trial penalized similarity to
verified prior events with strengths 0–0.75. Its CVS stayed between 16.17 and
16.58 while true event count fell from 44.00 to as low as 40.50; this idea is
not advanced as an effective method (`novelty_development.csv`). These results
do not revise the frozen v5 validation claims.

The next study should predeclare a physically valid but sparser proposal and
new held-out seeds. The testing question remains budgeted discovery of true,
nonredundant risks for a new controller. The method contribution will be
retained only if it improves the matched alternatives on independent data.
