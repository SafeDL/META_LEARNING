# Exploratory cross-family replay of the unchanged ModeShift rule

This is a **development-only retrospective replay**, not fresh validation.
The full target outcome banks and aggregate reports for the three IDM and
three FVDM validation seeds were inspected before this protocol. The
purpose is to determine whether the same ModeShift rule that passed the
new mixed-lane study is robust to previously studied same-family software
parameter revisions. No additional target episodes are claimed.

Reuse every one of the 3 seeds x 3 target builds in the IDM revision
validation and the 3 seeds x 3 target builds in the FVDM revision
validation: 18 fixed seed-target units. Each candidate pool consists of
320 historically event-free scenarios selected by the original source-
only gate. Use the original response encoding and the unchanged first-ten
mode-support rule. A selective cache oracle reveals only each method's
own 50 distinct target queries, sequentially. Reuse the same exact
ModeShift formula from `simple_residual_ablation.corrected_scores`:
historical source response plus the within-mode average of previously
observed target-minus-source response. Do not tune any hyperparameter.

Matched comparators in this diagnostic are raw historical response
ranking and within-mode historical response percentile. These use the
same source-safe pool, 50-query budget, and tie-breaking. The older
frozen reports remain authoritative for their own GP and random
comparisons; this replay does not overwrite them.

Primary descriptive endpoint: new ego collision or predefined near miss
count at B=50. Report ego collision count separately and every target
build. No seed, build, or unfavourable unit may be excluded. If ModeShift
loses to the best static comparator in either family, treat this as
evidence that its confirmed advantage is conditional on controller/
scenario shift, and seek a pre-specified *change-type decision rule*
before claiming general regression-testing effectiveness. If it wins
both, the rule still needs a new prospective cross-family campaign.
