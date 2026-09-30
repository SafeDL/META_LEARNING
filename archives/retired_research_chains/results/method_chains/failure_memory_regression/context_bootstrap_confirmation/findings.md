# Context-bootstrap calibrated confirmation findings

## Complete frozen banks

NL-IDM and inherited PPO each completed all 1,089 scenes in the disjoint
manifest, for 3,267 paired physical executions per chain. Selectors used the
frozen 40-query budget and ten random replays reuse the same physical truth;
they are not independent samples.

The predeclared primary, `directed_bootstrap_offset_ucb`, probes one
parent-only sample in each context with an eligible pool for each direction,
then ranks the remaining budget with the locally calibrated shared posterior.
Its coordinate ablation uses the same bootstrap and posterior rules.

| Chain and transition | Direction | True changes | Primary | Coordinate bootstrap | No-bootstrap coordinate calibration | No-bootstrap directed calibration | Static risk | Target-only |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NL V0→V1 | Regression | 47 | 16 | 16 | 11 | 11 | 4 | 1 |
| NL V0→V1 | Improvement | 2 | 2 | 2 | 2 | 2 | 2 | 1 |
| NL V1→V2 | Improvement | 73 | 15 | 15 | 20 | 20 | 16 | 0 |
| PPO V0→V1 | Regression | 10 | 9 | 10 | 1 | 1 | 1 | 1 |
| PPO V0→V1 | Improvement | 29 | 6 | 5 | 8 | 6 | 1 | 2 |
| PPO V1→V2 | Regression | 17 | 10 | 10 | 3 | 3 | 5 | 3 |
| PPO V1→V2 | Improvement | 47 | 10 | 10 | 16 | 17 | 1 | 0 |

Context bootstrap improved regression discovery in the new NL V0→V1 and both
PPO transitions compared with the stage-three score-only calibration. It found
16/47 NL regressions versus 4 for static risk, 9/10 PPO V0→V1 regressions, and
10/17 PPO V1→V2 regressions versus 3 for the directed no-bootstrap candidate.
The directed primary tied its coordinate ablation on four of six nonempty
directions and was one discovery behind it on PPO V0→V1 regressions.

The same 9-context bootstrap used budget that improved versions could have
spent on posterior ranking. The primary found only 15/73 NL V1→V2 improvements
versus 20 without bootstrap, and 10/47 PPO V1→V2 improvements versus 16–17
without bootstrap. It therefore improved regression search but did not
improve bidirectional discovery consistently. Family-level tests have only
three related clusters; every Holm-adjusted p-value is 1.0. The results are
descriptive and do not establish population-level superiority.

Full reports, query traces, truth maps, prefix curves, figures, and cost
ledgers are under `nl/` and `ppo/`.

## Next hypothesis

Bootstrap appears useful for the regression direction and costly for
improvement search. The next candidate will reserve context probes only for
regression queries, then begin improvement selection with posterior ranking.
It will use the same local offset calibration and a coordinate ablation, and
must be confirmed on another disjoint physical manifest. These completed
banks are development evidence and will not be reused as confirmation.
