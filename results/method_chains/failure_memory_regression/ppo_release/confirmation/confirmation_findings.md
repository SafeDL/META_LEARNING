# Frozen PPO confirmation: regression gain, improvement failure

The 225-case, three-context-per-family manifest was hashed before PPO
training. V0/V1/V2 checkpoints were selected by fixed last-step budgets, and
the 12-method selector protocol and source snapshot were frozen before the
675 target episodes were measured. Every build/scene result is valid and
matches the frozen checkpoint, scenario, seed, and execution contract. No
unknown transition labels remain. The study includes 10,240 on-policy
training decision steps (40,878 physics steps), plus 64/255 preflight steps.

| Weight update | Direction | True changes | Directed margin frontier @20 / area | Static risk | Coordinate residual | Same-coverage coordinate frontier |
|---|---|---:|---:|---:|---:|---:|
| V0→V1 | Regression | 5 | 5 / 0.281 | 0 / 0.000 | 0 / 0.000 | 5 / 0.252 |
| V0→V1 | Improvement | 10 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 |
| V1→V2 | Regression | 5 | 5 / 0.262 | 5 / 0.181 | 5 / 0.186 | 5 / 0.262 |
| V1→V2 | Improvement | 12 | 2 / 0.052 | 1 / 0.005 | 3 / 0.090 | 2 / 0.048 |

The revised method finds all 10 regressions across updates but misses most
improvements. It is **not** bidirectionally superior on the learning-policy
chain: it ties the standard baselines on the first improvement task and loses
to ordinary coordinates on the second. The first update has S01 regressions
and S01/S02 improvements; the second has S02 regressions and improvements.
V0/V1/V2 have 78/73/66 ego collisions on the 225 scenes, but those totals
conceal the bidirectional point flips. Random selector repeats use the same
physical bank and are not independent trials.

The failure mechanism is visible from parent-only data: many improvements
occur inside S02 contexts with almost entirely parent-fail grids, leaving
few binary pass/fail edges. The present context/TTC coverage rule applies
only to regression turns; improvement turns can spend their budget elsewhere.
This interpretation was made **after** confirmation and is a diagnostic, not
permission to change the frozen result. Any symmetric improvement acquisition
must be treated as a new method and tested on a new, independent bank.

The complete evaluator-only truth, query order, directional/context summaries,
family-paired statistics, cost ledger, and figures are in `evaluation/`.
