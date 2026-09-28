# Independent UCB revision confirmation

The preceding `chain2_confirmation/` bank is a frozen negative result for
bidirectional superiority: `directed_context_laplace` found 9/9 improvements
but 0/2 regressions at 20 queries. Both missed regressions were in a parent
pass context below the static-risk top 20. That completed bank is a diagnostic
source for this **new** method, not confirmation evidence for it.

Before measuring this bank, `directed_context_ucb` was fixed to use the same
context-specific target-response posterior as `directed_context_laplace`.
On regression turns it ranks candidates by `sigmoid(MAP target logit + one
posterior logit standard deviation)`; on improvement turns it uses the same
deterministic Laplace predictive probability as the prior method. It keeps
the same 20+20 no-repeat schedule, parent history, target oracle, scenario
grid, and tie rule as every baseline. It has no fixed context coverage slots.

The revision tied the prior method on both 7×7 development tasks (one
regression, four improvements) and on the separate geometry development task
(seven improvements, no regressions). This is sparse development evidence.
The `confirmation3` contexts were defined before the failed second
confirmation was read and have no exact context identity overlap with earlier
development or confirmation manifests. All nine methods in `protocol.json`
are frozen. The primary comparison is UCB versus `static_risk`,
`coordinate_residual`, and `directed_context_laplace`, reported separately
for regression and improvement. Zero-change directions remain NA.

The prior two confirmation results and this one must all remain in the study
record; sequential revisions cannot be described as one preregistered test.
