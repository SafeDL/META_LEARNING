# Fresh confirmation: mode-label allocation versus static and UCB1

This study tested the frozen protocol in
`docs/core_mine_mode_label_fresh_confirmation_protocol.md` on four new
seeds, 20330911, 20330925, 20331009, and 20331023. The source-only step
executed 2,560 historical-controller episodes. All four seeds passed the
qualification gate, with 270–275 eligible candidates per seed and 51–60
eligible cases in each mode. Four selectors then executed 1,600 charged
target episodes: 50 distinct source-safe cases for each method, seed, and
target. The target results were not precomputed.

The analysis replayed every selector decision using only earlier labels
and verified exact agreement across 427 scenarios repeated by multiple
selectors. All 32 ledgers had the correct seed, target, source hash, and 50
unique eligible queries.

| Method | VI/TTC collision cells | FVDM-revision cells | Mean cells | Ego collisions | New failures | Collision modes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ModeLabelShift-Risk` | 23.00 | 22.75 | **22.875** | 41.500 | 46.500 | 4.75 |
| `SourceMeanRaw-Static` | 19.00 | 21.75 | 20.375 | 36.750 | 45.125 | 5.00 |
| `ModeQuantile-Static` | 20.25 | 21.50 | 20.875 | 37.750 | 46.125 | 5.00 |
| `ModeUCB1` | 22.00 | 22.75 | 22.375 | 40.125 | 47.250 | 5.00 |

The paired difference in 4 x 4 collision cells was +2.50 over raw static
ranking (95% four-seed cluster-bootstrap interval [1.125, 3.375]) and +2.00
over static mode quantiles ([0.500, 3.500]). Both target-specific means
were positive against those two static methods.

Against the predeclared `ModeUCB1` baseline, the gain was only +0.50 cells
per seed-target unit, with interval [-0.750, 1.750]. It was +1.00 on VI/TTC
and exactly 0.00 on the FVDM revision; two seed-target units tied and two
favored UCB1. The label method also found fewer total new failures than
UCB1 (46.50 versus 47.25) and collisions in fewer modes on average (4.75
versus 5.00). Its cell advantage came with more queries in `stop_and_go`
(16.00 versus 11.50 of 50) and fewer in `fast_intrusion` (5.12 versus
7.62). The fixed effectiveness gate therefore **fails**: the label method
does not show a reliable advantage over ordinary online mode allocation.

## Research decision

The simple label-calibration rule reproducibly improves collision-cell
coverage over the two tested static rankings in this simulator suite. It
does not establish that historical transfer adds value beyond UCB1, and it
does not find more failures or cover more collision modes than UCB1. This
candidate is not ready to be presented as a novel algorithm.

Autonomous-driving regression prioritization already includes semantic
coverage and scene rarity on replayed recordings in STRaP
([Deng et al., ESEC/FSE 2022](https://doi.org/10.1145/3540250.3549152)).
Simulation-feedback fuzzing also guides scenario selection and mutation
using observed ADS behavior in SimADFuzz
([Yang et al., TOSEM 2026](https://doi.org/10.1145/3744242)). UCB1 is a
standard adaptive allocation rule
([Auer et al., 2002](https://doi.org/10.1023/A:1013689704352)). These
works do not implement the identical source-safe fixed-pool problem, but
they make generic feedback-driven selection insufficient as the claimed
contribution.

The useful next question is whether information from a genuinely paired
old/new controller revision improves regression discovery beyond a
target-only allocator, under the same charged budget. The present target
set has one related FVDM planner mutation and one different controller
family, so it cannot answer that question generally. A next development
must use paired revisions or explicitly controlled planner changes and
measure target-only failures against their historical outcomes. Treat these
four seeds as consumed; do not reuse them for tuning or confirmation.

The complete metrics, unit-level paired differences, and mode allocations
are in `analysis50.json`; the charged episode ledgers are in each
seed-target subdirectory.
