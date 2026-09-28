# Core confirmation findings

## Protocol completion

The preregistered `directed_residual` selector was evaluated on two complete,
disjoint physical banks: NL-IDM (`nl/`) and weight-inherited PPO (`ppo/`). Each
bank contains 1,089 frozen scenes and 3,267 paired physical executions. The
fixed 40-query replay, 20 per direction when available, generated 1,440
logical queries per chain; the 10 random repeats reuse the same physical truth
bank. Prefix metrics and figures are evaluator-only derivatives of those
banks.

The primary shared-offset posterior did not show consistent improvement over
the predeclared static-risk and coordinate-residual baselines:

| Chain and transition | Direction | True changes | Directed residual | Coordinate residual | Static risk |
| --- | --- | ---: | ---: | ---: | ---: |
| NL V0→V1 | Regression | 16 | 0 | 0 | 0 |
| NL V1→V2 | Improvement | 25 | 13 | 13 | 14 |
| PPO V0→V1 | Regression | 0 | NA | NA | NA |
| PPO V0→V1 | Improvement | 26 | 2 | 2 | 2 |
| PPO V1→V2 | Regression | 1 | 0 | 0 | 0 |
| PPO V1→V2 | Improvement | 20 | 0 | 0 | 0 |

`target_only` found the single PPO V1→V2 regression and 3 of its 20
improvements. Its better result on that pair is consistent with harmful
transfer from the parent-risk offset, but one chain and one regression cannot
establish a general effect. On NL V1→V2, a post-hoc context-local Laplace
variant found 18/25 improvements, but it did not help on regression discovery
or PPO and its family-level comparison is underpowered.

There are only three family clusters per related release chain. The exact
family sign-flip tests have coarse resolution and all multiplicity-adjusted
comparisons are non-significant. Neighboring grid cells and random selector
repeats are not independent samples. These results reject a claim that the
current directed feature model is already superior; they do not show that the
method family can never be improved.

## Development direction

Post-hoc feature replays found that several context-local UCB variants with
fixed context probes can discover changes in the NL bank. Those variants
override the posterior ranking with coverage probes, so they are not valid
confirmations of the paper's stated selector. A new candidate,
`directed_context_ucb_pure`, has been implemented with per-context local
residual features and a deterministic one-standard-deviation upper bound for
regression ranking. It uses one shared target-response posterior, parent
offset, and queried target labels; it has no forced context probe or frontier
override. On post-hoc replay it found 19/25 NL V1→V2 improvements, but 0/16
NL regressions, 2/26 PPO V0→V1 improvements, and 0/20 PPO V1→V2 improvements.
This is not consistent superiority and is development evidence only.

The stage-two development hypothesis is that a more weakly regularized local
context intercept can correct negative transfer from a miscalibrated parent
offset while retaining directed features. It must be tested against a
coordinate-context UCB ablation, the original directed residual, target-only,
and static risk on a newly frozen physical manifest. The stage-two method and
protocol are separate from the completed core confirmation.

## Reproducible artifacts

- `nl/report.md`, `nl/summary_by_direction.csv`, `nl/summary_by_context.csv`,
  `nl/transition_truth.csv`, `nl/prefix_discoveries.csv`,
  `nl/paired_family_statistics.json`, `nl/figures/`
- The corresponding files under `ppo/`
- `nl/exploratory/` and `ppo/exploratory/`: post-hoc comparison of previously
  explored role, margin, and frontier selectors; no new physical episodes
- `nl/exploratory_context_ucb_pure/` and `ppo/exploratory_context_ucb_pure/`:
  development replay of the newly coded pure UCB variant; no new physical
  episodes
