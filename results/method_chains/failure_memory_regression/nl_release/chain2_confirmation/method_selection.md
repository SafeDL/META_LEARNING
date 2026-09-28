# Method and comparison freeze

This confirmation run was frozen after a separate efficiency calibration and
two development grids. `nl2_v1` inherits the candidate selected solely by
mean vehicle progress under the fixed collision constraint; `nl2_v2` inherits
its normal-following parameters and adds only the predeclared TTC safeguard.

On the finer 7×7 development grid, V0→V1 had one regression and V1→V2 had
four improvements. The exploratory `directed_context_laplace` method tied
static parent risk on the sole regression and achieved improvement early area
0.352 versus 0.314 for static parent risk, with both finding 4/4 by query 20.
The regression sample is too small for a claim of superiority. The new method
is therefore a **candidate** for an independent confirmation, not an
established winner. The coarse 5×5 development grid had no changes and remains
part of the record.

The primary comparisons in this frozen bank are the new method against
`static_risk` and `coordinate_residual`, separately for regression and
improvement. All eight methods in `protocol.json` share parent pools, the
same 20+20 alternating budget, the same target oracle, and identical
tie-breaking. The scenario manifest has three predefined physical contexts
per family, a 7×7 grid, and no overlap of exact context identities with the
development or first confirmation run. Target results were not read before
this freeze. A zero-change direction will be NA, not a win. This is a second
study after a documented negative first confirmation and must not erase it.

The new model uses context-specific parent-edge features, a Gaussian
zero-mean residual prior (variance 0.25 for three global terms, 1.5 for local
terms), and a deterministic Laplace logistic-normal predictive approximation.
It has no target label access except charged oracle queries. The original
`directed_residual` remains as an ablation.
