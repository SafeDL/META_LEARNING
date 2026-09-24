# Current research position: source-safe regression testing at B=50

## The testing problem in ordinary language

An automated-driving controller has changed. An earlier controller ran a
large set of scenarios without an ego collision or near miss, but that does
not guarantee the new controller will pass them. We can afford only **50
executions of the new controller**. Which old-passing scenarios should we
replay first to reveal new collisions and near misses?

This is a fixed-suite regression-test prioritization problem. It is not a
claim to find every dangerous road situation or to certify real-world
safety. The old pass/fail label is identical for every eligible candidate,
so the useful historical signal is the *continuous safety margin* of each
old pass, such as minimum time to collision (TTC), not an old failure label.

## What was actually tested

Every study starts from 320 outcome-blind proposed scenarios: 64 in each of
five predefined functional modes. Historical controllers execute the whole
pool first. A case is eligible only if the historical runs complete without
ego collision or near miss. Each compared selector then chooses 50 distinct
eligible cases, with the same first-ten-query mode-support rule. Those ten
queries are **inside**, not in addition to, B=50. A new failure means an
ego collision or near miss of the target controller on a source-safe case;
it is not a count of independent software bugs.

The earlier two-lane heterogeneous-controller replications physically execute target
queries sequentially, at 20 Hz ego decisions and physics, with no target
outcome bank. Both compare six frozen selectors: source margin alone,
within-mode margin percentile, target-only Gaussian-process (GP) residual,
historical mean margin plus a mode-local target-feedback residual,
source-hypothesis composition plus the same residual, and random replay.
The source pairs are (IDM/MOBIL, MCTS-CV) and (IDM/MOBIL, Strong-FVDM);
the target is VI/TTC in both. Each source pair has three fresh seeds, so
each replication charges 3 x 6 x 50 = **900 physical target episodes**.
The second source pair was frozen before its target outcomes and addresses
a known internal MCTS rollout time-scale mismatch in the first pair.

| Historical sources -> new target | Best historical static, failures@50 | Historical mean + local residual, failures@50 | Composition + local residual, failures@50 | Target-only residual, failures@50 | Random, failures@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| IDM/MOBIL + MCTS-CV -> VI/TTC | 20.33 | 33.67 | 36.67 | 33.00 | 7.00 |
| IDM/MOBIL + Strong-FVDM -> VI/TTC | 26.67 | 35.00 | 34.00 | 31.67 | 8.33 |

These are three-seed means, not population-level guarantees. In the first
source configuration, composition finds 16.33 more new failures than the
strongest static comparator and 3.67 more than target-only at B=50. In the
cleaner FVDM-source configuration, the simpler historical-mean residual
finds 8.33 more than the strongest static comparator and 3.33 more than
target-only. Composition is **one fewer** failure per seed on average than
the historical-mean residual in that second configuration. Thus the frozen
source-robustness gate for composition **failed**. We cannot claim its
source-hypothesis weighting is a generally beneficial component.

The earlier same-family IDM and FVDM *parameter-revision* studies favored
static historical margin ranking, especially the within-mode quantile for
IDM. Consequently, feedback correction is not a universal recommendation
for every controller change. The supported conditional observation is:
when a new controller behaves differently enough from the historical
controllers in this simulator, its first charged outcomes can correct a
misleading historical margin ranking. This distinction is observational;
the current method does **not** automatically diagnose the type of change
or switch between static and adaptive strategies.

The later developmental simple-residual ablation makes the method claim
narrower. A constant within-mode correction of the historical margin
matches the GP residual on total new-failure count (32.67 versus 33.67
for IDM/MCTS; 35.33 versus 35.00 for IDM/FVDM). The GP found more ego
collisions, so a six-seed collision-primary confirmation was frozen and
run on fresh IDM/FVDM-source scenarios. Its 1,500 physical target episodes
passed the charged-query and deterministic-repeat audits. At B=50, the
historical-mean GP found 15.17 ego collisions and 35.17 total failures
per seed, versus ModeShift's 12.83 and 35.67. But TargetOnly-GP found
**16.83** ego collisions and 33.17 total failures. The predeclared
GP-component confirmation gate therefore **failed**: history plus GP is
not the strongest collision finder, even though it improves over simple
mode-level correction and static history on collisions. The GP's larger
functional failure-mode count than target-only (4.33 versus 2.17) comes
from near misses in the other modes, not collision diversity: **all
observed ego collisions in the confirmation occurred in only
`cutin_braking` and `fast_intrusion`**.

## Mixed-lane development and fresh B=50 confirmation

The two-lane result exposed a scenario problem: in three longitudinal
tasks the new controller could often escape by changing lanes. A separate
development pilot therefore changed only `lead_braking`, `stop_and_go`,
and `slow_lead_following` to one lane; the two cut-in tasks stayed on two
lanes. The subsequent four-seed, two-target comparison was frozen before
those seeds' source or target outcomes. In each seed, two old controllers
ran all 320 proposals. The source-only safety gate retained 275, 267,
269, and 270 eligible cases. Eleven selection methods then ran their own
50 distinct target episodes, sequentially, for each seed and target:
4 x 2 x 11 x 50 = **4,400 charged target executions**. The first ten
mode-support queries count within the 50. Target outcomes were not
precomputed or exposed to selectors.

The primary endpoint was the number of distinct collision-bearing cells
in a fixed 4 x 4 physical grid within each functional mode, not just a
near-miss count or a visualization score. At B=50, the simple historical-
margin-plus-mode-average-shift selector found 22.75 cells and 41.00 ego
collisions per seed on VI/TTC, and 19.50 cells and 38.00 ego collisions
on the delayed-braking FVDM revision. The planned mean-plus-local-GP
marginal selector found 20.25/34.25 and 16.50/26.75, respectively.
The original full CoRe composition plus marginal selector was worse
still (14.50 and 10.75 cells). Thus the revised scenarios make the
regression-testing problem observable across modes, **but do not validate
the proposed full algorithm**. A subsequent pre-specified development
test of a hierarchical residual also failed to beat the simple mode
shift on either target; it has not earned a fresh confirmation campaign.

A second developmental test penalized already discovered collision cells.
It raised 4 x 4 collision-cell count from 21.125 to 24.375, but reduced
actual ego collisions from 39.5 to 24.375 and *lost* on a 5 x 5 grid.
It failed its predeclared collision-retention gate. This is why the
fixed-grid endpoint must always be shown beside true collision yield and
grid-resolution sensitivity rather than interpreted alone.

The development-identified simple `ModeShift-Risk` rule was then tested
on **four entirely new seeds**, with its protocol frozen before any source
or target outcome at those seeds. All source-only gates passed. Across
seven selectors, two target controllers, and four seeds, there were
2,800 new charged B=50 target episodes; 670 repeated scenarios agreed
exactly across methods. ModeShift found 22.0 collision cells and 41.625
actual ego collisions per 50 queries on average. The strongest frozen
static mode-quantile comparator found 20.125 cells and 37.0 collisions;
target-only GP found 13.125 cells and 20.375 collisions. Its paired
cell advantage over mode quantiles was +1.875, with descriptive four-seed
cluster-bootstrap interval [1.25, 2.625]. The benefit persisted at
3 x 3 and 5 x 5 grid resolutions and met the frozen effectiveness gate.
However, the FVDM revision advantage over mode quantiles was just
+0.25 cell per seed, with three ties; the larger benefit arose in VI/TTC.
This confirms a *conditional simple calibration advantage*, not a
universal gain from online updates. The full CoRe method again lost.

An unchanged ModeShift selector was also replayed retrospectively on
the older, fully evaluated same-family IDM and FVDM revision banks. It
found 30.11 versus 31.11 new failures@50 for static mode quantiles in
IDM, and 25.44 versus 27.00 in FVDM. Because those target banks and
earlier aggregate results had already been inspected, this is a
development diagnostic, not another independent confirmation. It
reinforces the boundary: the useful *next research question* is how to
decide from early charged tests whether a new controller has changed
enough to warrant online mode correction. A post-hoc family label is
not an acceptable switching rule.

One fixed decision rule was tested developmentally: use static mode
quantiles for the first ten queries, then switch to ModeShift only when
a Beta-binomial Bayes factor favors different binary new-event rates
across modes. A first implementation accidentally ranked ineligible
proposals and was invalidated; the corrected source-safe implementation
was rerun and its all-static VI/TTC trace matched the frozen static
baseline exactly. The corrected rule protected same-family outcomes
but found only 20.25 VI/TTC collision cells at B=50 versus ModeShift's
22.75, because it never switched on that target. Its frozen development
gate failed. That specific binary-rate **switch criterion** was inadequate
in this suite; its prior/threshold must not be retrofitted to these
results. It does not establish that continuous target TTC feedback is
necessary once mode calibration is always active.

A direct 400-episode physical ablation on the already inspected fresh
seeds removed only the target TTC term from ModeShift's feedback while
keeping continuous historical margins. The label-only method found
22.125 collision cells@50, versus 22.000 for continuous ModeShift;
their selected sets overlapped in 47–50 of 50 cases per seed-target
unit. The predeclared necessity gate for continuous **target** feedback
failed. This is development evidence, not a new held-out confirmation.
It overturns any explanation that credits the current benefit to
learning fine-grained target safety-margin residuals. The tested
adaptive advantage is better understood as mode-level allocation from
event labels on top of historical continuous-margin ranking. See
`docs/core_mine_mode_label_ablation_protocol.md` and
`results/method_chains/core_mine/studies/mode_label_ablation/research_decision.md`.

A subsequent direct static control ranked by the raw mean historical safety
response, with no target feedback. In 400 additional physical executions on
the same already inspected seeds, label-based mode allocation found 22.125
collision cells@50 versus 18.500 for raw static ranking and 20.125 for static
mode quantiles. The paired gain over raw static was +3.625 cells per
seed-target unit, with a four-seed cluster-bootstrap interval of
[2.625, 4.500]. The gain appeared on both targets, but this remains
development evidence because the seeds were reused. It supports the value
of target event labels over the tested historical mean; it does not establish
novelty beyond existing adaptive allocation methods. See
`docs/core_mine_source_raw_static_development_protocol.md` and
`results/method_chains/core_mine/studies/source_raw_static_development/research_decision.md`.

A fresh confirmation then compared the label method with the two static
controls and a fixed-pool UCB1 allocator on four additional seeds and two
targets (2,560 historical-source and 1,600 target episodes). The label
method found 22.875 collision cells@50, versus 22.375 for UCB1. Its paired
advantage over UCB1 was only +0.50 cells, with a cluster-bootstrap interval
of [-0.75, 1.75]; the FVDM-revision target tied UCB1, and UCB1 found more
total failures. The frozen effectiveness gate therefore failed. The data
support an advantage over static allocation, but not an advantage over an
ordinary online allocator or a novel method claim. See
`docs/core_mine_mode_label_fresh_confirmation_protocol.md` and
`results/method_chains/core_mine/studies/mode_label_fresh_confirmation/research_decision.md`.

## Method idea and claim boundary

A defensible working idea is *correction of previously safe history*.
First preserve the old continuous margin on each source-safe scenario.
After every target replay, use its collision/near-miss label to adjust
the functional mode's priority. Use the updated estimate to prioritize the next
remaining case. This explains why an old pass is informative but not
decisive. In the latest mixed-lane B=50 experiment, a single average
correction per mode works better than the tested local GP corrections on
the collision-cell endpoint, but the new ablation shows that the
target's continuous TTC term is not required for this result. The
composition of multiple source
hypotheses and marginal archive objective also failed their frozen
component gates. No complete proposed method currently dominates the
strongest matched baselines across the relevant outcomes.

This is a **fresh-seed effectiveness result for a simple method in a
limited simulator**, not yet a publication-grade algorithmic novelty
result. Historical transfer,
adaptive safety testing, and regression prioritization have prior art;
see `results/method_chains/core_mine/studies/prior_art/scope.md`. A faithful
published-method comparison, a separate target architecture, and actual
software release pairs are missing. The experiment uses a two-vehicle
road model and synthetic controller algorithms. Its five mode labels do
not imply five independently discovered failure mechanisms. Across the
first three composition runs, 108 of 110 detected events came from just
`cutin_braking` and `fast_intrusion`; in the six-seed confirmation, **all
observed ego collisions** came from those same two modes. A separate
target architecture and collision-bearing interaction families are
necessary before claiming general regression-test effectiveness or
different types of collision hazards. The latest mixed-lane test changes
the empirical situation: it finds ego collisions in all five *scenario
labels*, but those labels still do not establish five independent bug
mechanisms.

## What the collision animation demonstrates

The mixed-lane side-by-side replays use the **same scenario and seed**
for historical IDM/MOBIL and the new VI/TTC controller. They show the
first charged ModeShift ego collision in each mode of the frozen B=50
campaign. Both ego decisions and physics update every 0.05 s; the GIF
also writes one frame every 0.05 s, rather than skipping to 0.2 s
snapshots. The target freezes at its collision while the matched
historical run continues to safe completion. An ordinary
lane change is not counted as dangerous: the red border and `EGO COLLISION
REGISTERED` label appear only when the simulator records an ego collision.
The collision occurs at 2.10 s (`fast_intrusion`), 1.80 s
(`cutin_braking`), 1.60 s (`lead_braking`), 2.00 s (`stop_and_go`), and
0.75 s (`slow_lead_following`). The matched historical replay completes
without collision or near miss in each case. The last three are one-lane
rear-end interactions; the first two remain two-lane cut-in interactions.
This verifies *simulator* collisions, not real-world crash likelihood.

Latest five GIFs, three-frame timelines, and numeric replay audit:
`results/method_chains/core_mine/studies/multimode20/gifs/modeshift_collisions_20hz/`.
Earlier two-lane replays, showing only two collision modes:
`results/method_chains/core_mine/studies/heterogeneous20_replication/gifs/core_collisions_20hz/`.
Full protocols and reports:
`docs/core_mine_heterogeneous20_replication_protocol.md`,
`docs/core_mine_fvdm_source_robustness_protocol.md`,
`docs/core_mine_simple_residual_ablation_protocol.md`,
`docs/core_mine_collision_primary_confirmation_protocol.md`,
`results/method_chains/core_mine/studies/heterogeneous20_replication/research_decision.md`,
`results/method_chains/core_mine/studies/fvdm_source_robustness/research_decision.md`,
`results/method_chains/core_mine/studies/simple_residual_ablation/research_decision.md`,
and `results/method_chains/core_mine/studies/collision_primary_confirmation/research_decision.md`.
First mixed-lane frozen protocol and decision:
`docs/core_mine_multimode20_protocol.md` and
`results/method_chains/core_mine/studies/multimode20/research_decision.md`.
Developmental cell-penalty decision:
`docs/core_mine_cell_aware_development_protocol.md` and
`results/method_chains/core_mine/studies/cell_aware_development/research_decision.md`.
Fresh simple-method confirmation:
`docs/core_mine_mode_shift_fresh_confirmation_protocol.md` and
`results/method_chains/core_mine/studies/mode_shift_fresh_confirmation/research_decision.md`.
Retrospective same-family boundary check:
`docs/core_mine_mode_shift_cross_family_replay_protocol.md` and
`results/method_chains/core_mine/studies/mode_shift_cross_family_replay/research_decision.md`.
Corrected development-only switch test:
`docs/core_mine_evidence_gate_development_protocol.md` and
`results/method_chains/core_mine/studies/evidence_gate_development/research_decision.md`.
