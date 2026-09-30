# Research decision after v8 and the control-cadence audit

## One-sentence testing problem

When an automated-driving controller changes, a previously passing scenario
suite can contain failures of the new controller. With 50 new executions,
which *historically passing* scenarios should be replayed first? The current
simulation uses different controller algorithms as a **proxy** for a revision;
it does not yet test actual successive software versions.

The useful signal is not an old failure label: all eligible histories are
event-free. Rather, their subcritical, continuous TTC margins differ. The
simple candidate method, **HistoryMargin-Residual**, averages the historical
continuous responses to initialize a risk map, then fits a mode-specific local
GP to the difference between observed target response and that map after each
charged target execution. It selects the next eligible scenario by predicted
target-event probability. The first ten functional-mode support queries are
part of B=50. Neither the GP residual alone nor cross-system transfer alone
is claimed to be a novel mathematical invention.

## What the frozen evidence supports

The v8 protocol was fixed before one qualification and three validation
seeds. Five modes x 64 candidates per seed were executed on four controllers;
all methods used the same source-safe eligible set and 50-query budget. A
source-safe target event means ego collision, or an ego near miss defined by
TTC < 1.5 s or vehicle-polygon clearance < 1 m, while all three other
controllers had neither event. The 3-seed x 2-target primary analysis was
preselected from development as opportunity-bearing; IDM and PPO remain in
the four-target secondary table. V8 used 5 Hz external decisions and 20 Hz
physics. Full-bank outcomes were cached but hidden until each logical query.

| Primary B=50 mean | New target failures | Ego collisions | Failure modes | CVS |
|---|---:|---:|---:|---:|
| HistoryMargin-Residual | 22.33 | 9.33 | 2.50 | 8.50 |
| HistoryMargin-Static | 13.33 | 4.83 | 2.50 | 5.83 |
| TargetOnly-Residual | 18.17 | 7.83 | 2.00 | 7.25 |
| CoRe-Residual | 21.83 | 9.33 | 2.50 | 8.33 |
| RandomSafe (10 repeats) | 5.32 | 2.47 | 1.82 | 3.23 |

Paired hierarchical bootstrap differences in new failures for the simple
method are +9.00 over static history [2.00, 16.67], +4.17 over target-only
[2.33, 6.00], and +0.50 over full CoRe [-0.50, 1.83]. The last interval
does not justify retaining composition. Nor does the equal 2.50 failure-mode
count versus static history justify a diversity claim. Only three validation
seeds and two opportunity-bearing targets underlie these intervals; they
should be read as descriptive uncertainty, not broad population guarantees.

The pooled gain masks strong heterogeneity. For MCTS-CV, the simple method
found 13/10/9 failures across the three seeds versus 13/10/11 for static
history: no improvement. For VI/TTC it found 33/33/36 versus 17/14/15.
IDM and PPO had no historically safe target failures in the selected pools.
Thus **the demonstrated benefit is primarily a VI/TTC conditional result**.
The five displayed v8 cases are real ego-collision replays, not the withdrawn
v4 adjacent-lane false near-miss labels.

`figures/per_target_b50.png` shows the held-out B=50 new-failure counts by
target and method; read its two panels separately, because the average gain
is not shared across controllers.

## Cadence audit prompted by the visual validity concern

The fixed 300 scenarios selected by the v8 simple method (150 per primary
target) were physically re-executed with the same scenario and seed at 10 and
20 Hz high-level decisions; physics remained 20 Hz. The original 5 Hz bank
provides the reference. No scenario was reselected. This is *not* another
B=50 efficacy validation: sources were not rerun at the new rates, so
source-safe eligibility there is unknown. Five example collisions were first
reproduced exactly at 5 Hz.

| Target, fixed 150 scenarios | 5 Hz events / collisions | 10 Hz | 20 Hz | Original events persisting at 20 Hz |
|---|---:|---:|---:|---:|
| MCTS-CV | 32 / 13 | 18 / 6 | 11 / 1 | 9 / 32 |
| VI/TTC | 102 / 43 | 104 / 42 | 102 / 41 | 99 / 102 |

The MCTS-CV contribution is strongly cadence-dependent, while most VI/TTC
failures persist. MCTS-CV's rollout also advances its constant-velocity gap
by `lead.speed - ego.speed` per simulated step, effectively a one-second
increment despite the environment's 0.2-second control step; increasing the
call rate changes this mismatch. PPO's checkpoint was trained at a different
native cadence and environment configuration, so simply running it at 10/20
Hz would not create a sound matched comparison. The present study can claim
only 5 Hz effectiveness, plus selected-case VI/TTC frequency robustness.
`figures/fixed_queries_cadence.png` shows total events and ego collisions on
the *same fixed 150 selected scenarios per target* at each frequency. It is a
controller sensitivity figure, not a new method-comparison figure.

## Independent physical 20 Hz confirmation

A separately frozen single-seed confirmation at seed 20291007 executed both
historical sources (IDM/MOBIL and MCTS-CV) on all 320 candidates at 20 Hz.
Without reading any VI/TTC target outcome, 276 completed source-safe
candidates across all five modes passed the eligibility gate. Each of four
methods then physically executed its own 50 VI/TTC target scenarios in query
order, for 200 target episodes; no target outcome bank was precomputed or
shared between selectors. A trace audit confirms 50 unique eligible queries
per method and identical outcomes on 80 cross-method repeated executions.

| Independent VI/TTC 20 Hz run, B=50 | New failures | Ego collisions | Failure modes | Early AUC |
|---|---:|---:|---:|---:|
| HistoryMargin-Residual | 35 | 16 | 4 | 0.649 |
| HistoryMargin-Static | 18 | 5 | 4 | 0.351 |
| TargetOnly-Residual | 34 | 15 | 2 | 0.624 |
| RandomSafe (one fixed order) | 10 | 2 | 2 | 0.217 |

This physically confirms a large gain from target-feedback correction over
static history at 20 Hz, but only **one additional failure** over target-only
learning. It supports the mechanism's feasibility, not a robust incremental
transfer benefit at higher frequency. It also uses only two heterogeneous
historical algorithms and one target seed, not true software revisions.

## Novelty and readiness verdict

Regression scenario prioritization from old-version outcomes is already the
explicit task of [SPECTRE](https://github.com/simplexity-lab/SPECTRE).
[Corso and Kochenderfer (AAAI 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/16876)
already transfer previous safety-validation knowledge to a changed system.
[STRaP (FSE 2022)](https://arxiv.org/abs/2209.01546) prioritizes ADS regression
recordings by scene coverage and rarity. Hence neither "reuse old tests" nor
"transfer across controllers" is a defensible standalone novelty claim.
The narrower, testable candidate contribution is to use *subcritical
continuous historical safety margins* within a predeclared all-source-safe
candidate set, then sequentially correct them with a small new-controller
budget. V8 supports this mechanism for one controller but does not yet prove
it is novel or generally effective against direct prior-work baselines.

Publication-strength completion remains open. The next frozen study should
use genuine same-family software revisions or explicitly labelled seeded
faults at a matched, cadence-consistent rate; rerun **both** histories and
targets at that rate; report every target, including zero-opportunity cases;
compare a faithful SPECTRE-style historical prioritizer and a transfer-aware
sequential baseline at B=50; and replicate the truly sequential physical
target queries over multiple independently frozen seeds and version pairs.
Until then, the
honest paper description is a promising, simulator-specific ablation result,
not a validated general solution to automated-driving regression testing.

Reproduction: `conda run -n metadrive python -m method_chains.core_mine.source_safe_experiment --stage analyze`,
`conda run -n metadrive python -m method_chains.core_mine.control_frequency_audit --stage examples`,
`conda run -n metadrive python -m method_chains.core_mine.control_frequency_audit --stage all --workers 4`,
`conda run -n metadrive python -m method_chains.core_mine.plot_source_safe_evidence`,
and `conda run -n metadrive python -m method_chains.core_mine.online20_confirmation --stage verify`.
The frozen data are in `analysis50.json`, `validate/records.csv`,
`frequency_audit/selected_queries.csv`, `frequency_audit/summary.json`,
`online20/target_queries.csv`, and `online20/verification.json`.
