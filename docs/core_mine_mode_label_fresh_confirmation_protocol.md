# Fresh B=50 confirmation of mode-label allocation

Frozen on 2026-09-24 before any source or target outcomes for seeds
20330911, 20330925, 20331009, and 20331023. These seeds do not appear in
the inspected CoRe-Mine source or target studies. Do not replace a seed,
alter the scenario grammar, or tune a selector after qualification.

## Test question

When a target controller produces only 50 charged tests, do its early
collision and near-miss labels help allocate the remaining tests across
functional interaction modes, beyond static historical rankings and a
standard UCB1 mode-selection rule?

This tests the simpler mode-label method. It does not validate the original
composition-plus-GP method or establish a result for real software releases.

## Frozen scenarios and budget

Use the existing source-safe Sobol proposal: 320 candidates per seed, 64 in
each of five modes. The two cut-in modes keep two lanes; the three
longitudinal modes use one lane. Execute both historical controllers,
`idm_mobil` and `fvdm_ref`, on all 320 candidates before target execution.
Keep a candidate only if both sources complete without ego collision or
near miss. The source-only gate requires at least 50 eligible candidates
overall and 24 in every mode for every seed. If a seed fails, stop the study;
do not replace it.

Targets are `vi_ttc` and `fvdm_delay05_brake3`. Both simulation physics and
ego decisions run at 20 Hz. Each method executes 50 distinct eligible
scenarios in sequence per seed and target. The common first-ten-query
mode-support rule counts against the budget. No target result may be
computed or revealed before its charged query.

The four selectors are:

1. `ModeLabelShift-Risk`: rank by the historical mean continuous response,
   plus each mode's average revealed target-severity minus historical-score
   residual. Target feedback is the label `0.5 * (collision or near miss) +
   0.5 * collision`.
2. `SourceMeanRaw-Static`: rank all eligible candidates by the unadjusted
   mean historical response.
3. `ModeQuantile-Static`: rank candidates by their within-mode percentile
   in the historical response.
4. `ModeUCB1`: after the common first ten queries, choose the mode maximizing
   its observed mean target-severity label plus `sqrt(2 * log(t) / n_mode)`;
   within that mode, choose the highest historical-response candidate.

The UCB1 comparator uses the canonical bounded-reward exploration bonus.
Because candidate removal changes the available scenarios within a mode,
this is an empirical fixed-pool comparator, not a claim that the usual
stationary-bandit regret guarantee applies.

## Endpoints and decision rule

Primary endpoint: distinct actually executed ego-collision-bearing cells
in the fixed 4 x 4 physical `(initial gap, relative speed)` grid within each
mode at B=50. Also report ego collisions, collision-or-near-miss failures,
number of collision modes, CVS, and 3 x 3 / 5 x 5 cell sensitivity. Cells
are not software bugs. Verify that repeated target scenarios have exactly
matching physical outcomes across selectors.

The proposed method passes only if it exceeds each of the three comparators
on the primary endpoint for both targets, has a positive four-seed
cluster-bootstrap 95% lower endpoint for every paired comparison, and has
at least 90% of the strongest comparator's mean ego-collision count. Its
3 x 3 and 5 x 5 means must not both be below all three comparators. Report
all outcomes regardless of the gate.

The comparison with UCB1 directly tests whether the observed gain is more
than ordinary adaptive allocation across modes. A positive result remains
specific to this simulator, these two targets, and this scenario grammar.

## Prior-art boundary

Regression prioritization for autonomous driving already includes semantic
coverage and scenario rarity on replayed recordings (STRaP,
ESEC/FSE 2022, DOI 10.1145/3540250.3549152). Recent simulation-feedback
fuzzing also uses observed ADS behavior to guide scenario selection and
mutation (SimADFuzz, TOSEM 2026, DOI 10.1145/3744242). UCB1 itself is a
standard online allocation rule (Auer et al., *Machine Learning*, 2002,
DOI 10.1023/A:1013689704352). These methods are not identical to this fixed
candidate, source-safe task, but they make broad claims about online
feedback or test prioritization insufficient. The UCB1 comparator is
therefore mandatory, and a positive result alone will not establish
algorithmic novelty.

## Reproduction

```powershell
conda run -n metadrive python -m method_chains.core_mine.mode_label_fresh_confirmation --stage sources --workers 2
conda run -n metadrive python -m method_chains.core_mine.mode_label_fresh_confirmation --stage targets --workers 2
conda run -n metadrive python -m method_chains.core_mine.mode_label_fresh_confirmation --stage analyze
```
