# B=50 clean-source robustness decision

The pre-outcome protocol is
`docs/core_mine_fvdm_source_robustness_protocol.md`. This study repeats the
20 Hz heterogeneous transfer task on three fresh seeds (20320405,
20320419, 20320503), replacing the previous simplified MCTS-CV historical
source with the 20 Hz internally controlled `SM-Strong-FVDM` profile. The
other historical source remains `idm_mobil`; the target remains VI/TTC.
The two history executions have different environment outer-step rates,
but **both ego controllers and both physics loops update at 20 Hz**. The
VI/TTC target's external action is also selected at 20 Hz.

Each seed began with the same outcome-blind five-mode × 64 Sobol proposal.
Both historical systems ran all 320 cases before any target outcome. The
common historical-event-free gates passed with 273, 264, and 266 eligible
cases across all five modes. Six frozen methods each physically executed
50 distinct eligible VI/TTC scenarios in order, for 900 charged target
episodes. No target bank was precomputed or shared between selectors. The
full trace and cross-method replay audit passed; 220 distinct scenarios
tested by multiple methods returned identical events and numeric safety
responses. This table is the three-seed mean; `RandomSafe` used one fixed
order per seed.

| Method | New failures@50 | Ego collisions@50 | Failure modes@50 | Early AUC | CVS@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HistoryMargin-Residual | 35.00 | 15.00 | 4.00 | 0.683 | 13.33 |
| CoRe-Residual | 34.00 | 16.00 | 4.00 | 0.681 | 14.00 |
| TargetOnly-Residual | 31.67 | 16.33 | 2.33 | 0.612 | 12.83 |
| HistoryMargin-Static | 26.67 | 9.33 | 4.33 | 0.623 | 10.50 |
| ModeQuantile-Static | 25.00 | 8.67 | 4.33 | 0.560 | 9.67 |
| RandomSafe | 8.33 | 2.33 | 2.67 | 0.193 | 4.83 |

The stronger static comparator on all three seeds was raw historical
margin. HistoryMargin-Residual gained +7, +7, +11 new failures over it:
mean **+8.33** with descriptive three-seed bootstrap interval [7, 11].
It gained +2, +6, +2 over target-only residual learning: mean **+3.33**
[2, 6]. CoRe gained +6, +4, +12 over the stronger static comparator
(mean +7.33 [4, 12]) and +1, +3, +3 over target-only (mean +2.33 [1, 3]).
However, CoRe minus HistoryMargin-Residual is −1, −3, +1: mean **−1.00**
[−3, 1]. Its +1 collision and +0.67 CVS advantages over mean residual do
not change the primary new-failure result. Therefore the **predeclared
source-robustness gate for the full composition method fails**. Do not
advertise function-hypothesis composition as a source-invariant gain or
cherry-pick the earlier IDM/MCTS source pair.

The narrower, consistently replicated result is that transferring a
continuous historical safety response and **locally correcting it with
charged target feedback** beats both static historical ranking and a
target-only GP in this heterogeneous VI/TTC replay task. The earlier
IDM/MCTS-source study had a large residual-vs-static advantage but only
+0.67 mean residual-vs-target-only failures [−3, 3]; the clean-source
study provides stronger evidence for an independent contribution from
historical knowledge (+3.33 [2, 6]). Pooling all six seeds would obscure
the two different source configurations and is not used to repair the
failed composition gate.

Limits remain severe: one VI/TTC target algorithm, two synthetic historical
source pairs, one two-vehicle simulator, and no real release pair. The
controller-family difference is a proxy for software revision, not a real
deployment update. Intervals with three seeds per source pair are
descriptive. No faithful published regression-prioritizer implementation
could be executed on this bank without redefining its required attributes;
the primary-source boundary is in
`results/method_chains/core_mine/studies/prior_art/scope.md`. The result establishes
a useful *conditional experimental mechanism*, not publication-grade broad
novelty, fault independence, or road-safety improvement.

Reproduction: run
`conda run -n metadrive python -m method_chains.core_mine.fvdm_source_robustness --stage sources --workers 2`,
then `--stage targets`, `--stage analyze`, and
`python -m method_chains.core_mine.plot_fvdm_source_robustness` in the same
environment. Raw banks/gates and 18 physical query ledgers, the audited
`analysis50.json`, and two figures are retained here.
