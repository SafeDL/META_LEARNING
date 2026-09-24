# Development-only evidence gate for static versus adaptive history

This protocol is fixed before executing the new evidence-gated campaigns.
The 18 same-family target banks and four mixed-lane source banks have
already been inspected; this is **development**, not independent
confirmation. The rule uses no controller-family name, mutant identity,
target oracle, or uncharged target response to decide whether to update.
Do not tune its threshold or prior after examining the outcomes below.

Implementation audit note: the first run mistakenly computed static
percentiles using all 320 proposals rather than the *source-safe eligible
subset*. That run is invalid and retained only for provenance under
`results/method_chains/core_mine/studies/evidence_gate_invalidated/`. The corrected
run uses the exact eligible-subset static comparator specified below,
keeps the same Bayes factor, prior, threshold, budget, and endpoints,
and writes to `results/method_chains/core_mine/studies/evidence_gate_development`. This is a
protocol-conformance bug fix, not an outcome-driven method change.

The selector compares two already implemented experts:

- **Static:** within-mode percentile of the old controllers' continuous
  safety response (`ModeQuantile-Static`).
- **Adaptive:** the unchanged historical mean response plus average
  observed target-minus-history response within each mode
  (`ModeShift-Risk`).

For the first ten charged queries use the static expert with the common
mode-support constraint. After each query, use only the observed binary
new event (ego collision or predefined near miss) and its mode to test
whether target event rates differ across modes. Under H0, all modes
share one unknown Bernoulli event probability; under H1, each mode has
its own. Both hypotheses use uniform Beta(1,1) priors, with prior odds
1:1. Compute the exact beta-binomial Bayes factor from the revealed
ordered binary outcomes; choose the adaptive expert only when BF(H1/H0)
is **strictly greater than 1**, otherwise choose static. The selector
recomputes this after each charged query, so it may switch back. The
common source-safe pool, deterministic tie-breaking, first-ten support
rule, and B=50 budget are unchanged. This is an exploratory
mechanism test, not a theorem that event-rate heterogeneity implies
ModeShift will improve ranking.

Evaluate *all* three IDM and three FVDM revision builds across their
three archived seeds each. Those are selective cached replays of
already executed physical outcomes, with only the selector's 50 labels
revealed to it. Compare B=50 new-failure and collision counts with the
unchanged ModeShift and static-quantile replays. For the mixed-lane
suite, reuse the four qualified historical source banks from
`docs/core_mine_multimode20_protocol.md`; physically execute the gate's
own 50 target episodes for each of the two targets and four seeds
(400 newly charged episodes, 20 Hz control and physics). Compare with
the prior ModeShift and mode-quantile B=50 traces, checking exact
repeated-scenario agreement. The mixed-lane primary metric remains
fixed 4 x 4 collision cells, alongside collision count and 3 x 3/
5 x 5 grid sensitivity.

Development progression requires: (1) mean new failures no worse than
static quantiles in **both** same-family controller families; (2) mean
mixed-lane collision cells at least as high as ModeShift on **both**
targets, with no more than 10% mean ego-collision loss; (3) the rule
must actually select *both* experts during the 50-query campaigns in
at least one unit each, rather than being a disguised fixed expert.
All units and negative outcomes remain reported. If any condition
fails, do not take this exact evidence gate to fresh confirmation or
retune its priors/threshold on these same results.

Even a positive development result would not prove novelty: Bayesian
model comparison, regression-test prioritization, and source transfer
all have prior art. A later prospective fresh-seed test and a faithful
prior-method comparison are still required before an algorithmic claim.
