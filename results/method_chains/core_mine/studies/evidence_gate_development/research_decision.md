# Corrected evidence-gate development decision

The first implementation accidentally ranked static percentiles over
ineligible source proposals. Its output remains under
`../evidence_gate_invalidated/` and is explicitly invalidated.
This corrected run ranks only the exact source-safe eligible pool, while
preserving the frozen Beta(1,1) Bayes-factor model, BF > 1 threshold,
first-ten static prefix, B=50 budget, and endpoints. The analysis has
an executable contract: whenever the gate remains static for all 50
queries, its chosen indices must exactly match the frozen
`ModeQuantile-Static` trace. This contract passed for all VI/TTC units.

The study uses 18 selective cached same-family IDM/FVDM units and 400
new physical 20 Hz mixed-lane target episodes. Every mixed-lane selector
made 50 distinct eligible queries; 695 repeated physical comparisons
against the existing selectors agreed exactly on event labels and
continuous safety values.

| Regime / metric | EvidenceGate | Static mode quantile | ModeShift |
| --- | ---: | ---: | ---: |
| IDM new failures@50 | 31.11 | 31.11 | 30.11 |
| FVDM new failures@50 | **27.44** | 27.00 | 25.44 |
| Mixed-lane VI/TTC collision cells@50 | 20.25 | 20.25 | **22.75** |
| Mixed-lane FVDM revision collision cells@50 | 19.50 | 19.25 | 19.50 |

The predeclared development gate **fails**: same-family protection and
collision-count retention pass, but the gate does not match ModeShift
on the VI/TTC collision-cell endpoint. It selects the static expert
for all 50 VI/TTC queries, and on average 47.5 of 50 FVDM-revision
queries. IDM and FVDM same-family campaigns use static on average
47.33 and 40.44 queries. Mean mixed-lane ego collisions are 36.25
versus ModeShift's 39.5, within the predeclared 10% tolerance, but
this does not rescue the primary failure.

The mechanism is informative: binary event-rate heterogeneity between
modes is not the same as a useful change in continuous safety margin
or collision severity. In VI/TTC, the event-rate Bayes factor does
not favor separate mode rates even though ModeShift's continuous
target-minus-history correction finds more collision regions. The
fixed prior and threshold will not be retuned on these inspected
outcomes or taken to fresh confirmation.

This development cannot establish method novelty. A future selector
would have to predict the *decision value* of correction from charged
continuous responses, retain strong static and ModeShift baselines,
and pass fresh multi-regime confirmation. The synthetic scene and
controller limitations of the earlier reports remain unchanged.
