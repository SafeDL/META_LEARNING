# FVDM-family B=50 validation decision

This is the independent-family follow-up frozen in
`docs/core_mine_fvdm_revision_validation_protocol.md`, after the 80-case
opportunity pilot. All three validation seeds passed the source-only gate:
each has at least 113 completed, event-free source cases per functional mode.
The first 64 safe cases per mode fix a 320-case candidate bank before any
target execution. The three target builds all ran against that full bank:
2,880 target episodes, with no inspected build dropped. The ego controller
and physics update at 20 Hz. Every campaign charges exactly 50 distinct
target queries; the initial mode-support queries count toward that limit.

Primary endpoint is the count of old-safe scenarios causing a new ego
collision or predefined near miss (TTC < 1.5 s or polygon clearance < 1 m).
Collision count is reported separately. Means below cover the nine paired
seed-by-build units; `RandomSafe` is averaged over ten fixed repetitions per
unit. Full records are in `records.csv`; individual units, opportunity, and
paired bootstrap output are in `analysis50.json`.

| Method | New failures @50 | Ego collisions @50 | Modes @50 | Early discovery AUC | CVS |
| --- | ---: | ---: | ---: | ---: | ---: |
| ModeQuantile-Static | 27.00 | 17.44 | 5.00 | 0.674 | 13.61 |
| HistoryMargin-Static | 26.33 | 17.00 | 5.00 | 0.652 | 13.56 |
| RiskDiverse-Static | 22.78 | 13.78 | 5.00 | 0.571 | 13.33 |
| HistoryMargin-Residual | 23.44 | 16.33 | 4.89 | 0.539 | 11.44 |
| TargetOnly-Residual | 18.22 | 12.89 | 3.33 | 0.358 | 8.06 |
| RandomSafe | 5.50 | 3.52 | 3.13 | 0.109 | 4.18 |

The target banks contain 322 new failures across the nine units (mean 35.78
per 320-case bank), of which ModeQuantile finds 243 (75.5%) with B=50.
The per-build means for ModeQuantile versus raw historical margin are:

| Target build | ModeQuantile | Raw margin | Risk-diverse |
| --- | ---: | ---: | ---: |
| FVDM delay 0.5 s | 16.00 | 16.33 | 13.67 |
| FVDM max brake 3 | 24.00 | 24.67 | 21.33 |
| FVDM delay + brake | 41.00 | 38.00 | 33.33 |

Paired hierarchical seed/build bootstrap differences for ModeQuantile minus
raw margin are +0.67 new failures [−0.56, 2.00], +0.44 ego collisions
[−0.22, 1.33], and +0.06 CVS [−0.56, 0.94]. The small overall positive
failure mean is driven by the combined mutant; the two single-change builds
favor raw margin slightly. ModeQuantile minus the risk-diverse proxy is +4.22
new failures [1.33, 7.11] and +3.67 collisions [1.22, 6.44]. This proxy is
**not** a faithful implementation of SPECTRE.

The predeclared *mean* family-generalization gate passes because both
comparisons on new failures are positive. A stronger statement that mode
quantiles reliably outperform simple raw historical margin across controller
families is **not supported**: the FVDM interval crosses zero, as do the
collision and CVS intervals. All five modes are already found by the three
static methods on average, so mode-count is saturated and does not establish
incremental value. The defensible result is that historical safety margins
strongly prioritize source-safe regression failures in two synthetic
controller families, and within-mode quantiles improve the IDM family and
the FVDM combined-change build but are not uniformly superior to raw ranking.
The earlier residual feedback architecture is not supported as the method
contribution. These tests do not establish real-ADS generality or
publication-grade novelty over prior regression-prioritization work.
