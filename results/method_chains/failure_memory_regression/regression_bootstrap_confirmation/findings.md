# Stage 5: regression-only context bootstrap

## Protocol and integrity

The frozen `nl-regression-bootstrap-bidirectional-v1` and
`ppo-regression-bootstrap-bidirectional-v1` protocols each cover 1,089 new
scenes and 3,267 paired physical executions. Both response banks completed
the full budget. A second `measure` pass reported 3,267 cached episodes and
zero new physical episodes for each chain. The source, runner, checkpoint,
manifest, and protocol hashes were checked before measurement; the result
files and frozen source snapshots are retained beside this report.

The primary, `directed_regression_bootstrap_ucb`, probes one minimum-TTC
parent-pass case per eligible context in the regression direction. It then
uses the directed, offset-calibrated posterior with a one-standard-deviation
UCB. In the improvement direction it uses the calibrated posterior mean from
the first query. The coordinate counterpart and prior-stage methods were
evaluated on the same physical response bank. End-point labels were withheld
from selection and used only for offline replay and evaluation.

## Directional results

Counts below are discoveries within each 20-query directional budget (or the
available true changes when fewer than 20 existed). Early area summarizes
the ordering of true changes within the available discovery pool.

| Chain | Transition | Direction | True changes | Primary | Key comparison |
|---|---|---:|---:|---:|---|
| NL-IDM | V0 → V1 | regression | 20 | 12/20; area 0.429 | coordinate regression bootstrap 10/20; coordinate role-gated 14/20 (area 0.552); directed bootstrap-offset 12/20 |
| NL-IDM | V1 → V2 | improvement | 70 | 16/70; area 0.771 | directed offset-calibrated 17/70 (area 0.824); static risk 15/70 (area 0.843); directed residual 16/70 (area 0.857) |
| PPO | V0 → V1 | improvement | 42 | 8/42; area 0.219 | directed offset-calibrated 10/42 (area 0.562); static role coverage 10/42 (area 0.500); static risk 1/42 |
| PPO | V1 → V2 | regression | 10 | 6/10; area 0.119 | directed offset-calibrated 10/10 (area 0.371); coordinate regression bootstrap 7/10; static risk 4/10 |
| PPO | V1 → V2 | improvement | 34 | 6/34; area 0.414 | coordinate and directed bootstrap-offset 8/34 (area 0.438); directed offset-calibrated and coordinate role-gated 6/34 |

The regression-only bootstrap improves on the unbootstrapped calibrated method
for the NL regression pool (12 vs. 0 discoveries), but does not beat the
coordinate role-gated control there. It loses to the directed calibrated
method on PPO regressions (6 vs. 10) and to that method and role coverage on
PPO V0 → V1 improvements (8 vs. 10). In NL improvements it finds one fewer
change than the unbootstrapped calibrated method. The direction-specific
bootstrap therefore does not provide a consistent bidirectional advantage.

Across the five nonempty direction-by-transition tasks, the primary finds
48 of 176 changes (27.3%). The strongest predeclared aggregate controls are
`directed_bootstrap_offset_ucb` at 45/176 and `directed_offset_calibrated_ucb`
at 43/176. This is a descriptive signal in favor of the primary under the
suite-wide 20-query directional budget, but it does not erase transition-level
losses or establish a statistically reliable effect.

## Inference and limits

Family-clustered comparisons use three scene families. Holm-adjusted
sign-flip p-values are 1.0 throughout and are descriptive only; three clusters
cannot establish a reliable cross-family effect. In particular, apparent
NL regression gains are concentrated in one family. These results support a
specific observation—that parent-only context probes can help some regression
pools—but not an overall superiority claim.

The main cost is opportunity cost within the fixed query budget: a context
probe spends a query before the target-feedback posterior can rank cases. The
asymmetry avoids that cost in improvement tasks, yet gains still reverse
between NL and PPO. The next method revision should test whether independent
direction-conditioned calibration, instead of pooling both directions'
target labels into one response model, improves this transfer. That proposal
is exploratory and requires a new frozen confirmation split before any claim.

Post-confirmation development replays do not support that direction-isolation
hypothesis: it reduced NL regression discoveries from 12 to 0 and PPO V0 → V1
improvement discoveries from 8 to 2. A prequential static/directed mixture and
a symmetric flip-probability model also underperformed the primary on those
tasks. These are tuning diagnostics on the same bank, not new confirmation
results; the prototypes remain exploratory only.
