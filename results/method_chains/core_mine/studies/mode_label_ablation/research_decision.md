# Development ablation: target TTC term is not the observed gain

The protocol in `docs/core_mine_mode_label_ablation_protocol.md` was
fixed before running this new method. It reuses four **already inspected**
fresh-confirmation seeds and two targets, so this is development, not
independent confirmation. `ModeLabelShift-Risk` physically executed its
own 50 distinct source-safe cases in each of eight seed-target units:
**400 new charged target episodes**. All decisions and dynamics ran at
20 Hz. The audit checks the eight new ledgers, source-only gates, eligible
indices, event definitions, and exact outcome agreement on 481 scenarios
repeated across the label method and three stored comparators.

| Method | VI/TTC 4 x 4 collision cells@50 | FVDM-revision cells@50 | Overall cells@50 | Overall ego collisions@50 |
| --- | ---: | ---: | ---: | ---: |
| `ModeLabelShift-Risk` (no target TTC term) | 24.00 | 20.25 | **22.125** | 41.500 |
| `ModeShift-Risk` (continuous target response) | 24.00 | 20.00 | 22.000 | **41.625** |
| `ModeQuantile-Static` | 20.50 | 19.75 | 20.125 | 37.000 |
| `SourceStatic-Marginal` | 18.50 | 18.75 | 18.625 | 35.125 |

The predeclared necessity gate for continuous **target** feedback fails:
the continuous method does not beat label-only on collision cells in
either target. Per unit, their selected sets overlap in 47–50 of 50
cases, and their first 10–14 queries are identical. At 3 x 3 resolution
they tie overall at 16.125 cells; at 5 x 5 the label method is 28.000
versus 27.875. New failure totals are 45.625 versus 45.750. None of
these small differences supports a material target-TTC benefit.

The two methods keep the same continuous **historical** TTC-derived
source response and the same within-mode ordering. Their only
difference is whether the revealed target response includes its
continuous TTC term. Because each update adds a constant within a
mode, the adaptive behavior here is cross-mode test-budget allocation,
not learning a new local ordering inside a mode. Relative to the static
mode-quantile comparator, both adaptive methods shift queries toward
the one-lane `lead_braking` and `stop_and_go` modes, which have many
collisions in this grammar. This is useful *within this benchmark* but
does not prove a new residual-modeling mechanism.

The earlier binary Bayes-factor switch failed because it asked whether
mode event rates differed enough to justify switching **from** static
ranking. This new result asks a different question: once mode shifts are
always applied, are target TTC values needed to compute them? The answer
here is no. The two findings are compatible. Neither justifies
retrofitting the failed switch threshold to the observed targets.

Research decision: retire the claim that continuous target-margin
residuals explain the confirmed benefit. The minimum supported
description is historical continuous-margin ranking with online
mode-level **event-label** calibration. Its novelty versus prior
regression prioritization and bandit-style allocation remains unproven.
Any stronger algorithmic claim must beat this simpler ablation and
strong static allocation baselines on new held-out targets or real
version pairs. Do not relabel these reused seeds as fresh confirmation.

Reproduce the new physical ablation (Conda environment `metadrive`):

```powershell
conda run -n metadrive python -m method_chains.core_mine.mode_label_ablation --stage targets --workers 2
conda run -n metadrive python -m method_chains.core_mine.mode_label_ablation --stage analyze
conda run -n metadrive python -m pytest method_chains/core_mine/tests/test_mode_label_ablation.py -q
```

Per-unit outcomes and paired differences are in `analysis50.json`; the
charged query ledgers remain under seed/target subdirectories.
