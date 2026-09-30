# Developmental hierarchical-residual decision

After inspecting the frozen mixed-lane study, one fixed, mechanistic
alternative was tested: a mode-wide Gaussian intercept plus the original
local Matérn residual on historical mean safety response. The exact
covariance and stop gate were written in
`docs/core_mine_hierarchical_residual_development_protocol.md` before
running its eight new B=50 target campaigns. These 400 physical target
episodes used the already qualified four source banks, the same 20 Hz
controllers/physics, and no target outcome cache. The audit verifies all
eight unique-query ledgers and deterministic repeated episodes against
the previously run methods.

| Target | Hierarchical collision cells@50 | ModeShift cells@50 | Original mean-GP cells@50 | Hierarchical ego collisions@50 | ModeShift ego collisions@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| VI/TTC | 21.25 | **22.75** | 19.75 | 38.75 | **41.00** |
| Delayed/brake-limited FVDM | 17.75 | **19.50** | 17.00 | 30.50 | **38.00** |

The hierarchical model improves on the original local-only GP by 1.125
collision cells per target-seed unit on average, but it is **1.625 cells
worse** than the simple ModeShift, with descriptive seed-cluster interval
[-2.5, -0.625]. It is worse on both target systems, and it finds fewer
actual ego collisions. The predeclared fresh-seed confirmation gate
therefore **fails**. Do not tune a second covariance on the same inspected
seeds or describe the hierarchical posterior as an established method
contribution. The simplest explanation remains that a constant per-mode
target calibration is sufficient or better in this specific mixed-lane
regression suite.

This is a development result, not a universal statement about Gaussian
processes. Its evidence is limited to two synthetic target controllers and
one road simulator. The full paired matrices, response ledgers, and
cross-method replay audit are preserved in `analysis50.json`.
