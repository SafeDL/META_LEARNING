# Fresh-seed confirmation: simple mode calibration, not full CoRe

The protocol `docs/core_mine_mode_shift_fresh_confirmation_protocol.md`
was frozen before source or target outcomes at seeds 20330703, 20330717,
20330731, and 20330814. Two historical controllers physically ran all
320 proposals per seed (2,560 source episodes). The source-only gates
passed, retaining respectively 273, 272, 269, and 270 eligible cases;
every seed retained at least 48 in each mode. Seven selectors then ran
their own 50 distinct, sequential target episodes for each seed and each
of two target controllers: **2,800 charged target episodes**. All ego
decisions and physics were at 20 Hz. No target-outcome bank was
precomputed. The audit checked all 56 ledgers and exact physical outcome
agreement on 670 scenarios repeated across selectors.

| Method | VI/TTC collision cells@50 | FVDM-revision cells@50 | Overall cells@50 | Overall ego collisions@50 | Overall new failures@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ModeShift-Risk | **24.00** | **20.00** | **22.000** | **41.625** | **45.750** |
| ModeQuantile-Static | 20.50 | 19.75 | 20.125 | 37.000 | 45.375 |
| SourceStatic-Marginal | 18.50 | 18.75 | 18.625 | 35.125 | 43.375 |
| TargetGP-Marginal | 15.25 | 11.00 | 13.125 | 20.375 | 24.500 |
| MeanGP-Marginal | 18.75 | 15.00 | 16.875 | 29.625 | 35.625 |
| CoReGP-Marginal | 13.50 | 10.50 | 12.000 | 22.125 | 27.500 |
| RandomSafe | 11.25 | 6.00 | 8.625 | 10.250 | 13.000 |

The frozen **ModeShift effectiveness gate passes**. Against the strongest
static mode-quantile ranking, its paired mean is +1.875 collision cells
per seed-target unit, with descriptive seed-cluster bootstrap interval
[1.25, 2.625]. Against source-static marginal ranking the difference is
+3.375 [2.75, 3.875]; against target-only marginal GP it is +8.875
[7.375, 10.0]. Its mean exceeds every frozen comparator, and its mean
ego-collision yield is highest, so the result does not depend on trading
away collision count for grid coverage. The sensitivity check also
favors ModeShift: 16.125 versus 15.250 cells for mode quantiles at 3 x 3,
and 27.875 versus 25.125 at 5 x 5. The corresponding source-static
figures are 14.375 and 23.375.

The gain is **heterogeneous**. For VI/TTC, ModeShift finds 3.5 more cells
than mode quantiles; for the FVDM revision the gain is only **0.25** cell
per seed, with three paired ties and one +1. That small FVDM difference
does not justify a broad claim that adaptive correction always matters.
The strong result is conditional on this synthetic mixed one-/two-lane
replay grammar and these two target controllers. Four seeds are repeated
samples within one simulator, not four independent driving domains.

The evidence now supports a *simple* idea: use an old controller's
continuous safety margin as the initial priority, then shift it using
the average target-minus-history response within each functional mode
as the 50 tests arrive. It does **not** support the original CoRe
source-hypothesis composition, local GP, or marginal-coverage objective:
their predeclared joint method is markedly inferior on these fresh
seeds. Source transfer and coarse mode calibration have value here;
the original extra components do not.

Claim boundary: collision cells are physical parameter regions, not
distinct software bugs; ego collisions are simulator events, not road
crash risk. SPECTRE and earlier cross-system safety testing already
establish related prior art, so this confirmation alone does not prove
publication-level algorithmic novelty. Actual software-release pairs,
a richer scene grammar, and a faithful available-input comparison to
prior prioritization methods remain missing. The central research
position should therefore be a carefully delimited source-safe
regression-test prioritization study, not a claim that full CoRe-Mine
has been validated.

Reproduction in the project workspace, using the `metadrive` Conda
environment:

```powershell
conda run -n metadrive python -m method_chains.core_mine.mode_shift_fresh_confirmation --stage sources --workers 2
conda run -n metadrive python -m method_chains.core_mine.mode_shift_fresh_confirmation --stage targets --workers 2
conda run -n metadrive python -m method_chains.core_mine.mode_shift_fresh_confirmation --stage analyze
```

The script reuses existing traces on rerun and checks source-only gates,
B=50 uniqueness, and repeated-scenario consistency at analysis.
