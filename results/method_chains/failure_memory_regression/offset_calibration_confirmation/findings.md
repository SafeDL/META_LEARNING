# Context-local parent-offset calibration findings

## Complete frozen banks

NL-IDM and inherited PPO each completed the disjoint 1,089-scene manifest and
3,267 paired physical executions. The frozen comparison used a 40-query
budget, alternating directions where available, and ten random replays of the
same physical truth bank. Random replays are not independent physical samples.

The predeclared primary, `directed_offset_calibrated_ucb`, estimates a
context-local correction to the parent-risk logit slope from queried target
labels. It uses a shared target posterior and a deterministic +1 posterior
standard-deviation score for regressions; improvements use the posterior-mean
complement. Its coordinate ablation uses the same prior and acquisition rule.

| Chain and transition | Direction | True changes | Primary | Coordinate calibration | Static risk | Coordinate residual | Target-only |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NL V0→V1 | Regression | 51 | 0 | 0 | 0 | 0 | 3 |
| NL V0→V1 | Improvement | 2 | 2 | 2 | 2 | 2 | 0 |
| NL V1→V2 | Improvement | 83 | 18 | 18 | 20 | 20 | 1 |
| PPO V0→V1 | Improvement | 46 | 6 | 6 | 1 | 1 | 11 |
| PPO V1→V2 | Regression | 21 | 11 | 13 | 8 | 7 | 1 |
| PPO V1→V2 | Improvement | 13 | 2 | 2 | 0 | 0 | 0 |

The primary beat static risk and coordinate residual on PPO V1→V2 regressions,
but its coordinate calibration ablation found 13/21 versus 11/21. Both
calibration variants tied on the NL improvement transitions; the static and
ordinary coordinate methods each found 20/83 in NL V1→V2. The primary missed
all 51 NL regressions, while target-only found 3. On PPO V0→V1 improvements,
target-only found 11/46 versus 6/46 for the primary. Thus local slope
calibration helps selected task directions but does not consistently beat
simple or coordinate-only controls, and the directed edge block adds no
repeatable gain.

Paired family tests use only the three interaction families from one related
release chain. Every Holm-adjusted comparison is 1.0; these coarse tests are
descriptive and cannot establish population-level superiority. Full
transition summaries, query traces, truth maps, prefix curves, figures, and
cost ledgers are in `nl/` and `ppo/`.

## Next hypothesis

Both stage two and stage three used score ranking without a context bootstrap.
The missed NL regressions and PPO's task-dependent ranking suggest a cold-start
coverage problem: an uncertain target context may receive no feedback before
the risk posterior concentrates elsewhere. The next development candidate
will reserve one parent-only query in each eligible context and then use the
locally calibrated posterior for the remaining budget. This opened bank is
development evidence only; confirmation requires another disjoint manifest.
