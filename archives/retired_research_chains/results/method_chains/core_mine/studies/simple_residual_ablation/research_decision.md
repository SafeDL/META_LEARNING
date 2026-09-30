# Developmental decision: local GP versus simple online correction

This comparison was designed **after** the two heterogeneous VI/TTC
replication results had been inspected. It uses their already qualified
source banks and seeds, but physically executes 600 additional target
episodes: two new methods x three seeds x two source configurations x 50
unique charged queries. It is not an independent confirmation. The protocol
is `docs/core_mine_simple_residual_ablation_protocol.md`; all methods obey
the same source-safe gate, B=50, ten-query mode support, 20 Hz target
decision/physics cadence, and outcome-reveal boundary. The analysis audits
all twelve new traces and deterministic repeats against the previously run
GP traces.

| Source pair | Method | New failures@50 | Ego collisions@50 | Early AUC | CVS@50 |
| --- | --- | ---: | ---: | ---: | ---: |
| IDM/MCTS | Historical mean + GP residual | 33.67 | 16.67 | 0.672 | 13.67 |
| IDM/MCTS | Historical mean + mode-average shift | 32.67 | 13.67 | 0.660 | 12.17 |
| IDM/MCTS | Historical mean + kernel-weighted shift | 30.33 | 15.33 | 0.598 | 11.67 |
| IDM/FVDM | Historical mean + GP residual | 35.00 | 15.00 | 0.683 | 13.33 |
| IDM/FVDM | Historical mean + mode-average shift | 35.33 | 12.33 | 0.784 | 12.17 |
| IDM/FVDM | Historical mean + kernel-weighted shift | 30.00 | 11.33 | 0.644 | 11.00 |

For the primary *new failure count* used in the earlier protocols, the GP
does **not** clearly beat simple mode-level recalibration. Per-seed
`ModeShift - GP` failure differences are `[0, -3, 0]` for IDM/MCTS and
`[+2, +1, -2]` for IDM/FVDM. Therefore a claim that a local GP is
necessary for finding more collision-or-near-miss cases is rejected.

The secondary **ego-collision** contrast is more consistent:
`ModeShift - GP` is `[-5, -3, -1]` and `[0, -4, -4]` in the two source
configurations. CVS favors the GP in five of six seeds. This suggests, but
does not prove, that a locally varying correction may prioritize more
severe regressions than a constant within-mode shift. Because this endpoint
was promoted after inspecting developmental results, it requires a newly
frozen, fresh-seed, collision-primary confirmation. The current six seeds
cannot be reused as its confirmatory evidence. The kernel-shift comparator
underperformed the GP on both total failure and collision means, but that
does not rescue a general GP-necessity claim against ModeShift.

Even successful fresh-seed confirmation would establish only a conditional
method-component benefit within the same two-vehicle VI/TTC simulator. It
would not make GP regression itself novel, nor validate real software
releases or real-world safety. Full unit metrics and cross-method
repeat audit are in `analysis50.json`.
