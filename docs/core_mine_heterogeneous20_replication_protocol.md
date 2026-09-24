# Frozen multi-seed heterogeneous-controller confirmation at B=50

This is a prospective replication of the **single-seed** VI/TTC result in
`docs/core_mine_online20_protocol.md`. The prior seed 20291007 supplied
design information and must not be pooled into the new primary analysis.
Freeze this file and the selector implementation before any VI/TTC outcome
at the new seeds. It tests transfer across distinct controller algorithms,
**not** actual successive software releases.

Use new source/target simulator seeds **20320111, 20320125, 20320208**.
For each seed, generate the unchanged v8 scrambled-Sobol pool of five
functional modes × 64 scenarios using `sparse_scenarios`. Execute both
`idm_mobil` and `mcts_cv` on all 320 candidates at 20 Hz external decisions
and 20 Hz physics. A candidate is eligible only if both source episodes
complete with neither ego collision nor near miss. The source-only gate is
at least 50 eligible scenarios spanning all five modes. If it fails for
any seed, stop the entire validation without changing seeds, bounds,
sources, target, or eligibility. Preserve and report all gate results.

Only after all three source gates pass, compare these frozen selectors:

1. `HistoryMargin-Residual`: mean historical continuous response plus
   mode-local Matérn-5/2 target residual GP, original fixed hyperparameters.
2. `CoRe-Residual`: per-mode historical-hypothesis evidence weighting,
   null branch, and the same residual GP.
3. `HistoryMargin-Static`: original mean historical response, no residual.
4. `ModeQuantile-Static`: percentile rank of mean source response within
   each mode among eligible scenarios, no target feedback.
5. `TargetOnly-Residual`: same GP/target feedback with constant 0.50 prior.
6. `RandomSafe`: one deterministic source-safe random order per seed.

Every selector physically executes its own 50 **distinct** VI/TTC target
scenarios sequentially; no target outcome bank, cross-method target cache,
or oracle label lookup is permitted. Repeated scenarios between selectors
must yield identical deterministic event and numeric safety responses. The
same mode-support rule applies to all methods for the first ten queries,
which count inside B=50. The target receives 20 Hz high-level decisions.
No method may inspect another method's target trace before or during its
campaign. Run each seed/method as a separately saved atomic campaign; do
not repeat a campaign merely to improve its result.

Primary endpoint: new target ego collision or near miss found by B=50 among
scenarios safe for *both* sources. Secondary endpoints: ego collisions,
early-discovery AUC, failure modes, and independently partitioned CVS.
Report every seed and all six methods. Primary comparisons are
`HistoryMargin-Residual` minus (i) the stronger of the two frozen static
selectors per seed, and (ii) `TargetOnly-Residual`; paired differences and
descriptive seed bootstrap intervals accompany means. A standalone
transfer-plus-feedback efficacy claim requires a positive mean difference
against **both** of those comparators, no omitted seed, and no material
collision-count regression. Composition is retained as a contribution only
if `CoRe-Residual` itself improves on mean residual under the same budget.
No data-dependent rerun, retuning, or method switch after this validation.

The MCTS-CV source's known one-step rollout time-scale mismatch is a
threat to controller fidelity and must be disclosed. The study's limited
two-vehicle road model and lack of actual released ADS versions also limit
its claim. Even a positive outcome does not alone prove publication-grade
novelty relative to published regression prioritizers.
