# Highway-env replication results

This is the only retained result root for the representative paper
replications.

- `shared/`: 192 balanced scenarios and 1,152 real simulator responses.
- `adate/`: paper-specific AdaTE results plus the shared-pool run.
- `detour/`: complete DETOUR D1/D2 runs, traces, figures, and GIFs.
- `fst/`: trained similarity model, 50-repeat evaluation, figures, and GIFs.
- `scenariofuzz/`: deterministic six-target paper-aligned suite and replays.
- `evaluation/`: normalized cross-method records, coverage matrix, figures,
  report, and validation.
- `sut_selection/`: qualified IDM+MOBIL, VI-TTC, MCTS-CV, and PPO-ECE policies,
  their common-scenario response bank, and risk-structure audit.

Read `evaluation/report.md` for the comparable results. The shared benchmark
does not replace each paper's own conclusions: it separates failure discovery
from performance estimation and marks differences in target-history access.
