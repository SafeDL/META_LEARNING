# Context-local posterior UCB confirmation findings

## Complete frozen banks

NL-IDM and weight-inherited PPO each completed the same disjoint 1,089-scene
manifest and 3,267 paired physical executions. Ten methods used the same
alternating 20-per-direction schedule and 40-query budget. The 1,520 logical
queries per chain include 10 random replays of the same physical truth bank;
those repeats are not independent trials.

The predeclared primary, `directed_context_ucb_loose`, uses a context-local
target-risk posterior with fixed parent offset, context-intercept prior
variance 4, and a deterministic +1 standard-deviation regression score.
Improvement queries use the posterior-mean complement. It does not use forced
context probes or neighborhood-frontier overrides.

| Chain and transition | Direction | True changes | Primary | Static risk | Coordinate residual | Directed residual | Key comparator |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| NL V0→V1 | Regression | 17 | 0 | 0 | 0 | 0 | target-only 2 |
| NL V1→V2 | Improvement | 33 | 19 | 8 | 11 | 11 | coordinate-context UCB 19 |
| PPO V0→V1 | Regression | 0 | NA | NA | NA | NA | no observed regression |
| PPO V0→V1 | Improvement | 35 | 1 | 1 | 1 | 1 | target-only 2 |
| PPO V1→V2 | Regression | 24 | 2 | 2 | 2 | 2 | target-only 4 |
| PPO V1→V2 | Improvement | 25 | 9 | 4 | 4 | 5 | coordinate-context UCB 9 |

The primary improved discoveries over static risk and coordinate residual in
NL V1→V2 and PPO V1→V2 improvements, but it tied the coordinate-context UCB
ablation on discovery counts in both tasks. On PPO V1→V2, its early area was
below both coordinate- and directed-context UCB despite the same 9/25 final
discoveries. It missed all NL V0→V1 regressions and did not beat target-only
on either nonempty regression task. Its PPO V0→V1 improvement result was
below target-only and tied static risk.

Thus the results support a task-specific gain from context-local adaptation,
but do not support a general superiority claim or an incremental gain from
directed edge features. All family-level sign-flip comparisons are descriptive
only (three families within one related chain) and multiplicity-adjusted
p-values are 1.0. The full reports, prefixes, truth maps, query traces, and
paired-family summaries are under `nl/` and `ppo/`.

## Next model change

The remaining failure mode is negative transfer from a fixed parent-risk
offset. The next development candidate will let queried target outcomes
calibrate the parent offset with a context-local residual slope, while
retaining the shared posterior and deterministic acquisition. It must be
tested on a new manifest and against both coordinate and directed versions;
these completed banks are now development evidence and will not be reused as
confirmation for that change.
