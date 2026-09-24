# Prospective heterogeneous-controller B=50 replication

The pre-outcome protocol is
`docs/core_mine_heterogeneous20_replication_protocol.md`. Three new seeds
(20320111, 20320125, 20320208) each received 320 outcome-blind Sobol
scenarios spanning five functional modes. At 20 Hz high-level decisions and
20 Hz physics, the two historical controllers (`idm_mobil`, `mcts_cv`) ran
all 320 per seed before the target was queried. The source-only gates passed
with 271, 270, and 269 candidates completed without ego collision or near
miss under **both** sources. VI/TTC target outcomes were never precomputed.

Each of six frozen selectors physically ran 50 distinct target scenarios
sequentially on each seed: **900 charged target episodes**. The same
mode-support rule and budget applied to every selector. All 18 traces passed
the unique-query/source-safe audit; 244 distinct scenarios executed by more
than one selector had identical event, collision, TTC, and clearance values.
The endpoint below counts new ego collisions or ego near misses in the
source-safe set. It is not a count of distinct software defects.

| Method | New failures@50 | Ego collisions@50 | Failure modes@50 | Early AUC | CVS@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CoRe-Residual | 36.67 | 17.67 | 2.67 | 0.759 | 14.00 |
| HistoryMargin-Residual | 33.67 | 16.67 | 2.67 | 0.672 | 13.67 |
| TargetOnly-Residual | 33.00 | 15.00 | 2.00 | 0.622 | 11.83 |
| ModeQuantile-Static | 20.33 | 10.00 | 2.67 | 0.421 | 9.50 |
| HistoryMargin-Static | 13.67 | 7.00 | 2.67 | 0.213 | 7.33 |
| RandomSafe (one order per seed) | 7.00 | 2.33 | 2.33 | 0.174 | 4.33 |

CoRe's new-failure counts by seed are 34, 38, 38. Relative to the strongest
frozen static selector on each seed (ModeQuantile for all three), its paired
advantages are +12, +18, +19 failures: mean **+16.33** with a three-seed
bootstrap interval [12, 19]. Relative to target-only residual learning the
advantages are +4, +5, +2: mean **+3.67** [2, 5]. Relative to the source-
mean residual ablation they are +1, +3, +5: mean **+3.00** [1, 5]. The
corresponding collision advantages are +7.67, +2.67, and +1.00 per 50;
CoRe's CVS advantage over target-only is +2.17 [0.5, 3.0]. These intervals
are *descriptive for just three related simulator seeds*, not broad
population confidence about new ADS releases. The standalone mean-residual
method beats static history strongly but exceeds target-only by only +0.67
failures [−3, 3]. Within this **IDM/MCTS source configuration**, the
complete compositional-residual model has the highest observed yield.
The subsequent pre-frozen Strong-FVDM-source robustness study reverses
its comparison with the mean-residual variant; see
`results/method_chains/core_mine/studies/fvdm_source_robustness/research_decision.md`.
Composition therefore cannot be retained as a source-robust contribution.
Across the three CoRe traces, the 110 detected events comprise 68
`cutin_braking`, 40 `fast_intrusion`, one `slow_lead_following`, and one
`stop_and_go` case; `lead_braking` had no detected new failure. Therefore
the method finds several functional contexts but its yield is concentrated
in two, and the table's mode count must not be narrated as comprehensive
coverage of all five functions.

The evidence supports a narrow mechanism: when the target is a substantially
different controller algorithm, historical response hypotheses with
mode-specific evidence weighting and local target-feedback residuals can
prioritize source-safe failures better than either the historical mean alone
or target-only learning. This study does **not** separate the composition
weights from the null branch within CoRe; that additional ablation is still
needed before attributing the gain to one of them. The earlier same-family
IDM and FVDM revisions instead favored simple static history, so CoRe is
**not** a universal recommendation for every controller update.

Important limitations: the target is a different algorithm, not a real
successive software release; all scenarios are generated in one two-vehicle
Highway-env model. The MCTS-CV historical source has a known rollout
time-step mismatch even though its externally executed decisions and
physics were 20 Hz. The study used one target algorithm and did not run a
faithful implementation of published SPECTRE, STRAP, or iterative RL
transfer baselines; their input/task interfaces differ. These issues block
claims of broad ADS generality or publication-grade superiority. The
[original SPECTRE implementation](https://github.com/ssbse2021/SPECTRE),
[Corso and Kochenderfer's iterative safety-transfer paper](https://ojs.aaai.org/index.php/AAAI/article/view/16876),
and [Uesato et al.'s continuation-based failure search](https://openreview.net/pdf?id=B1xhQhRcK7)
make clear that historical or cross-controller transfer alone is not novel.
The potentially distinctive claim is limited to compositional, local
correction for a **fixed previously-safe regression replay suite** under a
charged B=50 target budget.

Reproduce with
`conda run -n metadrive python -m method_chains.core_mine.heterogeneous20_replication --stage sources --workers 2`,
then `--stage targets`, `--stage analyze`, and
`python -m method_chains.core_mine.plot_heterogeneous20_replication` in the
same environment. Raw per-query traces, source banks, source-only gates,
`analysis50.json`, and the two audited figures are under this directory.
