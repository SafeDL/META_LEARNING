# Cross-family retrospective decision: ModeShift is conditional

The unchanged `ModeShift-Risk` rule was replayed against the already
executed and previously inspected full target banks from the frozen IDM
and FVDM same-family revision studies. This is **retrospective
development**, not fresh confirmation and not 2,700 new physical target
episodes. Every one of 18 seed-target units was retained. A selective
oracle revealed only each selector's 50 charged cases; the first-ten
mode-support rule, old-safe candidate pool, and tie-breaking were matched.
The methods were ModeShift, raw historical response ranking, and static
within-mode historical response percentile. Raw per-unit query ledgers
are in `records.csv`; the script and protocol are
`method_chains/core_mine/mode_shift_cross_family_replay.py` and
`docs/core_mine_mode_shift_cross_family_replay_protocol.md`.

| Revision family | Method | New failures@50 | Ego collisions@50 | CVS@50 |
| --- | --- | ---: | ---: | ---: |
| IDM, 3 builds x 3 seeds | ModeShift-Risk | 30.11 | 25.78 | 14.78 |
| IDM | HistoryMargin-Static | 29.33 | 25.44 | 15.06 |
| IDM | ModeQuantile-Static | **31.11** | **26.44** | **15.61** |
| FVDM, 3 builds x 3 seeds | ModeShift-Risk | 25.44 | 16.56 | 12.67 |
| FVDM | HistoryMargin-Static | 26.33 | 17.00 | 13.56 |
| FVDM | ModeQuantile-Static | **27.00** | **17.44** | **13.61** |

The development diagnostic is negative for unconditional online
calibration. ModeShift trails the strongest static comparator by 1.00
new failure per IDM seed-build unit and 1.56 per FVDM unit. It is
better than raw historical response by 0.78 in IDM, but worse by 0.89
in FVDM. All six builds were included; the target-specific figures are
preserved in `analysis50.json`.

This does not undo the prospective, fresh-seed mixed-lane confirmation:
ModeShift exceeded the strongest static comparator there on collision-
region discovery and collision count. Rather, the two experimental
regimes expose an unresolved *decision problem*: when does the early
new-version feedback add information beyond a good static historical
ranking, and when does it merely perturb that ranking? A general
method claim now requires a pre-specified, observable rule for deciding
whether and how strongly to update, validated prospectively across
both kinds of controller change. Choosing static for one family and
ModeShift for another after seeing the results would be invalid.

Limitations: the two old validation suites use precomputed target banks
and a wider two-lane scenario proposal than the later prospective
mixed-lane study. The different endpoint (new collision-or-near-miss
count, rather than collision cells) is the one frozen for those suites.
This diagnostic should not be pooled numerically with the later study
or described as independent evidence of algorithmic novelty.
