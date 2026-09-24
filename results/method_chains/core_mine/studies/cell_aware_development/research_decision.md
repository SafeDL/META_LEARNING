# Cell-aware ModeShift development decision

The single fixed `CellAware-ModeShift` rule and its pass gate were specified
in `docs/core_mine_cell_aware_development_protocol.md` before its eight
new B=50 campaigns. The source banks and original comparators were reused,
but all 400 target episodes were newly and sequentially executed, not
read from a target-outcome bank. The audit checks 50 unique eligible queries
in every campaign and exact agreement on 214 cross-method repeated cases.

| Method | 4 x 4 collision cells@50 | 3 x 3 cells@50 | 5 x 5 cells@50 | Ego collisions@50 | New collisions or near misses@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CellAware-ModeShift | **24.375** | **16.000** | 21.250 | 24.375 | 31.250 |
| ModeShift-Risk | 21.125 | 15.500 | **25.875** | **39.500** | **45.125** |

The 4 x 4 gain is +3.25 cells per seed-target unit; it is positive on
both targets (+4.5 VI/TTC, +2.0 FVDM revision), with descriptive seed-
cluster bootstrap interval [2.0, 4.5]. However, actual ego collisions
fall by 38.3%, far beyond the predeclared 10% maximum loss. The 5 x 5
grid reverses the coverage ranking. This makes the apparent primary gain
grid-sensitive and expensive in true hazard yield. **The development gate
fails; no fresh confirmation or penalty tuning is warranted for this
rule.**

This result cautions against presenting one arbitrary cell resolution
as a complete measure of distinct dangerous cases. Future result tables
should keep collision count, several cell resolutions, and an independent
continuous coverage metric side by side. The broader testing task and
ModeShift's benefits remain testable, but the tested cell penalty is not
an established contribution.
