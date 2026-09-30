# Mode-calibration development decision: do not validate these variants

The pre-outcome protocol is
`docs/core_mine_boundary_calibration_development_protocol.md`. It separates
the two development seeds (20291230, 20300315) from the previously inspected
IDM validation seeds. Both source-only gates passed; all three fixed IDM
mutants were retained. Every selector used the same 320 source-safe candidates
and B=50 charged target queries per seed/target. The first ten mode-support
queries count inside B=50.

| Method | New failures@50 | Ego collisions@50 | CVS@50 |
|---|---:|---:|---:|
| ModeQuantile-Static | 30.50 | 24.33 | 14.58 |
| MarginModeRate | 30.50 | 24.17 | 14.83 |
| MarginModeCal | 30.17 | 23.67 | 14.83 |
| HistoryMargin-Static | 28.33 | 22.67 | 14.42 |
| HistoryMargin-Residual | 26.00 | 21.50 | 13.42 |
| TargetOnly-Residual | 20.00 | 15.67 | 10.08 |
| RandomSafe (ten orders) | 6.48 | 4.57 | 5.05 |

`ModeQuantile-Static` is the strongest static comparator: simply rank the
old controller's continuous TTC response *within each mode*, without any
new-controller feedback. `MarginModeRate` ties its 30.50 mean failures but
finds slightly fewer ego collisions; `MarginModeCal` is slightly worse on
both. On seed 20291230, the quantile-only static method finds 31.00 failures
versus 30.67/30.00; on seed 20300315 it finds 30.00 versus 30.33/30.33.
Thus neither adaptive variant improves on the strongest static baseline in
both seeds. The frozen progression gate fails and the new validation target
seeds **must not be executed for these variants**.

This is a substantive negative result. The simpler explanation is that the
three global IDM parameter changes preserve most of the within-mode
source-margin ordering; normalizing the old margin across modes captures
the useful signal. Sparse target feedback does not demonstrably add value.
It remains a hypothesis, not a proved causal mechanism. [SPECTRE](https://github.com/simplexity-lab/SPECTRE) already
prioritizes ADS regression scenarios using previous-version execution
attributes, including collision and demand, so "reuse historical risk" by
itself is not a novelty claim. The next study, if pursued, must target a
genuinely different regression mechanism with local or non-monotone behavior,
predeclare a representative set of faults rather than selecting winners, and
compare against a direct previous-version prioritizer. Keep all failed
studies visible; do not rebrand the quantile baseline as the proposed method.

Raw development evidence is `analysis50.json` and `records.csv`. The two
source and target banks and gate manifests are under
`results/method_chains/core_mine/studies/idm_revision_validation/{seed}/`.
