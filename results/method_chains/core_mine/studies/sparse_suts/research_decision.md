# CoRe-Mine sparse-risk multi-SUT: diagnosis and decision

## Result

This preregistered-v2 rerun does **not** establish an advantage for CoRe-Marginal over the strongest matched geometry baseline.  At budget 20, CoRe-Marginal and FPS-Marginal both obtain CVS = 4.292.  Their paired difference is 0.000 with bootstrap interval [-0.167, +0.167].  The no-composition ablation is 4.417, so this data does not support a compositional-residual benefit either.

The physical bank is nevertheless sparse overall: its target critical-event rate is 10.57% (500 candidates x 5 seeds x 4 leave-one-out targets), versus the earlier dense bank.  Thus sparsity alone was not the missing experimental condition.

## Root cause

The immediate limiting factor is the **functional-scenario proposal**, expressed jointly with the response overlap it induces; it is not simply that the four retained SUT implementations are poor choices.

* The outcome-independent 90:10 benign:challenge construction makes nearly every challenge candidate critical for all policies: IDM 99.6%, VI 100.0%, MCTS 90.8%, PPO 94.4%.
* Benign candidates are completely safe for IDM, MCTS, and PPO.  Only VI retains benign near misses (4.2%).
* Event-set Jaccard overlaps are 0.912 (IDM/MCTS), 0.940 (IDM/PPO), and 0.937 (MCTS/PPO).  These three policies have zero target-only failures per seed.  VI supplies the sole non-shared signal, averaging 19 target-only failures per seed.

The SUT roster therefore has architectural diversity and one useful counterfactual response (VI); it passed retention and is worth keeping.  But the current five functional modes reduce its behavioural diversity to one shared, easily found stress tail plus one VI-only near-miss pattern.  FPS can cover that tail directly, leaving no broad conditional residual for CoRe to exploit.

## Next experimental design (freeze before collecting outcomes)

Keep the four SUTs as a baseline, but rebuild the functional proposal around several **continuous borderline bands** per mode rather than a benign/challenge split.  Vary headway, relative speed, manoeuvre onset, and cut-in duration independently so that risk pockets can differ by controller.  Add interaction modes that stress planning horizon, braking aggressiveness, and learned-policy uncertainty separately.

Before running the full comparison, use a small, separate qualification bank with gates fixed in advance:

1. At least two leave-one-out targets must have source-safe target failures, distributed over at least two functional modes.
2. No three-SUT subset may have all pairwise event Jaccard overlap above 0.90.
3. Critical events must not be concentrated almost entirely in one declared stress stratum; report rates by mode and continuous parameter band.
4. The full test bank remains outcome-blind for selection, with new seeds and no post-hoc retuning.

Only if these gates hold is another CoRe efficacy comparison diagnostic of compositional transfer rather than of a trivially shared stress tail.
