# Frozen mixed-lane, two-target B=50 research decision

The protocol `docs/core_mine_multimode20_protocol.md` was frozen after a
separate developmental pilot, before outcomes at seeds 20330406, 20330420,
20330504, and 20330518. The two historical controllers physically ran all
320 proposed cases per seed. The common source-safe gates passed with 275,
267, 269, and 270 eligible scenarios; each seed retained at least 48
eligible cases in every mode. The cut-in modes used two lanes and the
longitudinal modes used one lane. Ego control and physics updated at
20 Hz. Every one of 11 selectors physically executed 50 distinct,
sequential target episodes for each of two target controllers and four
seeds: **4,400 charged target episodes**. No target outcome bank was
precomputed. The analysis audited the 88 ledgers and exact agreement on
798 scenarios repeated across methods; `analysis50.json` retains every unit.

| Method | VI/TTC collision cells@50 | FVDM-revision collision cells@50 | VI ego collisions@50 | FVDM ego collisions@50 |
| --- | ---: | ---: | ---: | ---: |
| MeanGP-Risk | 19.75 | 17.00 | 34.75 | 28.50 |
| MeanGP-Marginal (planned main method) | 20.25 | 16.50 | 34.25 | 26.75 |
| CoReGP-Risk | 20.00 | 17.00 | 35.25 | 28.50 |
| CoReGP-Marginal (original full method) | 14.50 | 10.75 | 25.00 | 19.50 |
| TargetGP-Marginal | 16.50 | 11.50 | 25.00 | 17.00 |
| SourceStatic-Marginal | 18.25 | 19.50 | 31.00 | 36.25 |
| ModeShift-Risk | **22.75** | **19.50** | **41.00** | **38.00** |
| ModeQuantile-Static | 20.25 | 19.25 | 35.25 | 36.75 |
| RandomSafe | 12.75 | 8.25 | 15.00 | 8.75 |

The predeclared **MeanGP-Marginal component gate fails**. It ties
MeanGP-Risk overall on collision cells (18.375 each), loses to
SourceStatic-Marginal on the FVDM revision (16.5 versus 19.5), and loses
to the simple ModeShift-Risk on both targets (20.25 versus 22.75 for VI;
16.5 versus 19.5 for FVDM). The **CoRe composition gate also fails**:
CoReGP-Marginal averages 5.75 fewer collision cells than MeanGP-Marginal
across the eight seed-target units, with seed-cluster descriptive bootstrap
interval [-8.25, -3.5]. Source transfer itself matters here: both source-
informed static and adaptive methods substantially exceed the matched
target-only GP on the fixed B=50 collision-cell endpoint. But local GP
correction and the marginal acquisition do not explain the best results.

The mixed-lane scenario redesign directly addresses the old opportunity
failure. In the developmental pilot, source-safe VI/TTC cases produced
actual ego collisions in all five functional modes, including 9/24 in
one-lane `lead_braking`, 12/24 in one-lane `stop_and_go`, and 2/24 in
one-lane `slow_lead_following`. The delayed-braking FVDM target produced
collisions in four modes. In the frozen comparison, ModeShift-Risk finds
on average 4.75 collision-bearing modes at B=50 across target-seed units,
versus only two collision-bearing modes in the old two-lane VI/TTC study.
This supports the **scenario-design diagnosis**, not a universal method
claim: allowing an unconstrained escape lane made three longitudinal
functional labels weak tests of collision regression for VI/TTC.

For visual audit, the directory `gifs/modeshift_collisions_20hz/` contains
the first **charged B=50 ego collision per mode** from ModeShift-Risk on
VI/TTC, replayed beside historical IDM/MOBIL with identical scenario and
seed. All five replays match the charged target outcomes numerically; the
historical controller completes without collision or near miss. Both ego
control and physics run at 20 Hz, and each GIF frame represents 0.05 s.
The target freezes on its recorded collision while the historical run
continues to its safe completion near 7 s; `*_outcome.png` shows both
end states together. A quick lane change alone
does not constitute a hazard: the red border is drawn at the simulator's
recorded ego collision. The collision times are 2.10 s (`fast_intrusion`),
1.80 s (`cutin_braking`), 1.60 s (`lead_braking`), 2.00 s (`stop_and_go`),
and 0.75 s (`slow_lead_following`). See `manifest.json` for charged-query
identifiers, clearances, and replay frequency, and the timeline PNGs for
legible before/during/collision frames. These are simulator events, not
five independently verified software defects.

The next method hypothesis is motivated by the failure pattern, not
claimed as established: when target changes cause a roughly mode-wide
response shift, the old local-only GP correction may decay back to an
incorrect source prior away from its few observed cases. A hierarchical
residual that includes a mode-level intercept plus a local component is a
minimal, testable revision. It must beat the already strong ModeShift,
ModeQuantile, and target-only controls on **fresh** seeds before being
called effective. If not, the simplest supported result remains an
empirical regression-test ranking study, not a new algorithm.

Limits: the targets are synthetic VI/TTC and FVDM-parameter revision
controllers, not actual released ADS builds. The one-lane and two-lane
scenes are straight-road Highway-env simulations with one scheduled lead
vehicle. The four seeds are not independent road environments. Collision
cells are physical parameter regions, not distinct software bugs or
real-world crash rates. The result does not support deployment safety
claims.
