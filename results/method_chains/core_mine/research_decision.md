# Research decision

**Question.** Under a fixed target-test budget, can function-level source responses and selectively revealed target feedback discover severe, non-redundant vulnerability regions better than point-risk search?

**Evidence.** At B=20, CoRe-Marginal CVS is 10.667, versus FPS-Marginal at 10.880; severity is 16.648 versus 17.009.

**Method.** CoRe combines function-specific hypotheses, a Matérn-5/2 residual GP, predictive-evidence weights, and marginal archive coverage; NoComposition, NoNull, and FPS-Marginal/NoResidual are retained in the evidence.

**Conclusion.** no reliable gain in this round. The λ check found no simple marginal rule that kept 90% of the severity baseline, so physical confirmation is not warranted. This is cache evidence only, not a real-world safety claim.
