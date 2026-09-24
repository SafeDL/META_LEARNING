# Frozen multi-target, collision-diversity regression test at B=50

The development pilot at seed 20330105 showed that changing only the
longitudinal modes to a one-lane road creates collision-bearing,
historically safe cases. That pilot is excluded from this confirmation.
This protocol is frozen **before any source or target outcome** at the
four new seeds **20330406, 20330420, 20330504, 20330518**. No result-
dependent seed, bound, method, or endpoint changes are allowed.

For each seed, generate the unchanged v8 scrambled-Sobol proposal of
five modes x 64 cases and the same physical bounds. Use two lanes for
`fast_intrusion` and `cutin_braking`; use one lane for `lead_braking`,
`stop_and_go`, and `slow_lead_following`. The one-lane modes keep the
original lead schedule, 7 s horizon, collision logic, polygon clearance,
and TTC threshold. This separate environment grammar does not replace
prior two-lane results. Run both historical systems (`idm_mobil` external
and `SM-Strong-FVDM` internal) on all 320 cases before any target run.
Eligible cases must complete without ego collision or near miss under
**both** historical controllers. The source-only gate requires at least
50 eligible cases total and at least 24 in **every** mode in **every**
seed. If it fails, stop without replacing seeds or retuning.

The two frozen targets are externally controlled `vi_ttc` and the
internally controlled synthetic revision `fvdm_delay05_brake3`. The
external ego decisions, internal profile control, and physics all run
at 20 Hz; the internal environment outer-step call is 5 Hz but invokes
four 20 Hz ego/physics updates. Target outcomes are never precomputed.
For every target x seed x method, the method physically executes its own
50 **distinct** eligible cases sequentially. Repeated cases between
methods must agree on event labels and numerical safety responses. The
same first-ten-query mode-support rule applies to every method and
counts inside B=50.

Frozen methods (11 total):

1. `MeanGP-Risk`: source-mean response plus mode-local GP residual;
   greedily select predicted event probability.
2. `MeanGP-Marginal`: same posterior, select expected severity-aware
   marginal archive coverage over the **eligible source-safe suite** with
   the original fixed `lambda=0.10`.
3. `CoReGP-Risk`: source-hypothesis composition, null branch, and local
   residual; select event probability.
4. `CoReGP-Marginal`: same complete posterior plus marginal coverage.
5. `TargetGP-Risk`: same local GP with constant 0.50 prior, no source
   response; select event probability.
6. `TargetGP-Marginal`: target-only GP plus the identical marginal rule.
7. `SourceStatic-Risk`: source-mean response without target model update;
   select event probability.
8. `SourceStatic-Marginal`: same static source predictions plus verified
   target-event archive updates in the marginal acquisition only.
9. `ModeShift-Risk`: source mean plus the fixed simple within-mode
   average target residual from the previous ablation; rank its score.
10. `ModeQuantile-Static`: within-mode source-mean percentile, no target
    feedback after the common mode-support prefix.
11. `RandomSafe`: one deterministic eligible random order per target/seed.

The **primary endpoint** is `CollisionCells@50`: distinct fixed 4 x 4
normalized `(initial_gap, relative_speed)` cells within each mode that
contain at least one actually executed **ego collision**. Cell boundaries
come from the source-blind proposal's fixed physical bounds, not from
target outcomes; out-of-range values receive overflow cells. This is
independent of the selection method's RBF coverage kernel. It gives no
credit for near misses or repeated collisions in the same cell. Also
report `CollisionModes@50`, actual ego collision count, new collision-
or-near-miss count, severity-weighted CVS, early discovery AUC, query
distribution, and wall-clock cost. Never call a cell a distinct software
bug.

The main component contrasts are `MeanGP-Marginal` minus each of
`MeanGP-Risk` (value of marginal selection), `TargetGP-Marginal`
(value of historical response), `SourceStatic-Marginal` (value of local
target correction), `ModeShift-Risk` (value beyond simple calibration),
and `ModeQuantile-Static` (strong static ranking). A full-method
contribution requires positive mean primary differences against **all**
five, a positive seed-cluster bootstrap 95% lower endpoint against the
first three, a positive mean within **both** targets, and no more than
10% mean ego-collision-count loss versus the best of those five
comparators. Otherwise report which component or simpler version remains
supported. `CoReGP-Marginal` earns a composition claim only if it beats
`MeanGP-Marginal` on the primary endpoint in both targets, with positive
cluster-bootstrap lower endpoint; its risk-only counterpart isolates the
acquisition effect. All eleven methods and every seed/target are reported
regardless of gate outcome.

Bootstrap units are the four seeds, retaining both target systems within
each sampled seed; intervals are descriptive given the shared simulator.
The two targets are different synthetic controllers, not real released
ADS versions. A positive result would support a conditional method
component in this mixed one-/two-lane Highway-env benchmark, not road
safety certification or publication-grade generalization by itself.
