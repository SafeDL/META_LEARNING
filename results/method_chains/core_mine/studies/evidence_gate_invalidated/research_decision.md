# INVALID RUN: static comparator candidate-set mismatch

This first implementation accidentally ranked the static expert among
all 320 source proposals instead of only the candidates that passed the
historical safety gate. Consequently it did **not** implement the frozen
static expert in the mixed-lane campaigns; for example, it stayed static
for all 50 VI/TTC queries but did not reproduce the static comparator's
selection. The raw 400 target episodes and `analysis50.json` are retained
for provenance, but the pass/fail conclusion below is **withdrawn** and
must not be cited as method evidence. The corrected, protocol-matched
run is separately stored in `../evidence_gate_development/`.

## Superseded preliminary interpretation (invalid)

The static/ModeShift decision rule was frozen in
`docs/core_mine_evidence_gate_development_protocol.md` before executing
its campaigns. It compares a shared Bernoulli new-event rate across
functional modes with independent rates, using exact Beta(1,1)
integrated likelihoods and equal prior odds. The first ten charged
queries use static mode quantiles. After that, BF(H1/H0) > 1 selects
the unchanged ModeShift score; otherwise the static score is used.
No controller-family label or uncharged target outcome enters the gate.

The test replayed all 18 archived same-family IDM/FVDM seed-target units
selectively and physically executed 400 new mixed-lane target episodes
at 20 Hz on the four previously qualified source banks. Each mixed-lane
campaign made 50 distinct charged queries. The analysis verified exact
agreement on 652 repeated physical scenarios against the existing
ModeShift and static-quantile traces.

| Regime / metric | EvidenceGate | Best static quantile | ModeShift |
| --- | ---: | ---: | ---: |
| IDM new failures@50 | 31.11 | 31.11 | 30.11 |
| FVDM new failures@50 | 27.44 | 27.00 | 25.44 |
| Mixed-lane VI/TTC collision cells@50 | 19.75 | 20.25 | **22.75** |
| Mixed-lane FVDM revision collision cells@50 | 19.25 | 19.25 | **19.50** |

The predeclared development gate **fails** because collision-cell
discovery is below ModeShift on both mixed-lane targets. The gate uses
the static expert for all 50 VI/TTC queries and for 41.25/50 queries
on average in the mixed-lane FVDM revision; in the same-family IDM and
FVDM suites it uses static for 47.33 and 40.44 queries, respectively.
It does activate the adaptive expert in some campaigns, but too rarely
where ModeShift adds value. VI/TTC actual ego collisions fall to 35.5
versus ModeShift's 41.0 in the original mixed-lane B=50 study.

The mechanism explains the failure without inventing a positive claim:
binary event prevalence can be similar across modes even when continuous
target-minus-history safety responses and collision severity differ.
The Bayes factor therefore chooses a shared-rate interpretation and
keeps a static ranking in the very setting where the continuous
ModeShift rule performed best. The fixed BF prior/threshold will not be
retuned on these inspected outcomes or taken to fresh confirmation.

This result strengthens the need to define the intended transfer unit
and diagnostic signal before another algorithmic claim. A future
pre-specified method would need to test *continuous safety-margin
miscalibration* or directly estimated marginal discovery value, rather
than assuming that cross-mode binary event-rate heterogeneity is the
right proxy. Any such development must use new, independent seeds
before confirmation and preserve the strong static and ModeShift
comparators. No full CoRe novelty or broad release-regression benefit
follows from this negative test.
