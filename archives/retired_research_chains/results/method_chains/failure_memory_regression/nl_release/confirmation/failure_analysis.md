# Confirmation failure analysis

The 7×7, nine-context NL-IDM confirmation bank is complete: 441 scenes for
each of three related controller versions, with no unknown paired outcomes.
V0→V1 has 3 improvements and no regressions; V1→V2 has 8 improvements and
no regressions. This chain does not provide a two-direction discovery task.

For the two nonempty improvement tasks, the directed residual finds 3/3 and
6/8 at 20 direction queries, versus 3/3 and 7/8 for static parent risk.
Its early-discovery area is 0.262 and 0.281, versus 0.271 and 0.329 for static
risk. All improvements lie in one component for V0→V1 and two components for
V1→V2 under the frozen four-neighbor grid; both methods touch all components
by 20 queries. The proposed directional feature therefore adds no measured
benefit on this confirmation bank.

The V1 update combines shorter following gaps with a slightly larger normal
braking limit. Which individual parameter caused the absence of regressions
is unknown without a separate ablation. The current regularized MAP model
also shares local edge coefficients across contexts; with few changed labels,
its target adaptation may be too weak or may transfer between unrelated
contexts. These are hypotheses for development, not conclusions established
by this confirmation run.

The next study must keep this result intact. A new release calibration should
be chosen for independent efficiency/comfort requirements on a development
set, then frozen before a new full-bank confirmation. Model revisions should
be selected on development data, including context-specific residuals and a
deterministic posterior-predictive approximation. The present confirmation
bank can explain failure modes but cannot become a fresh holdout for a tuned
success claim. The PPO weight-inheritance stage remains gated by a credible
NL bidirectional protocol and is not claimed as completed.

After this negative result was known, the later
`directed_context_laplace` candidate was replayed on this bank as an explicitly
**post-hoc diagnostic** (`exploratory/`). It tied static risk on the first
improvement task (3/3, area 0.271) and found 8/8 on the second (area 0.562),
versus static risk's 7/8 (area 0.329). This is useful for debugging model
behavior but cannot be relabeled as independent confirmation, and the bank
still has zero regressions. The separate `chain2_confirmation/` protocol was
frozen before its target results were read.
