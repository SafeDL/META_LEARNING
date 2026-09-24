# Same-family IDM regression validation at B=50: negative residual result

This is a seeded *parameter-change* proxy for software regression, not a
comparison of real deployed releases. The frozen protocol is
`docs/core_mine_idm_revision_validation_protocol.md`. Physics and the ego's
internal longitudinal control both run at 20 Hz. The external environment
step is 5 Hz, but its action is ignored by the profiled IDM ego; the ego
recomputes acceleration on every 0.05 s physics step and never initiates a
lane change. The scheduled other vehicle may cut in.

For each of three held-out seeds, 640 proposals were first run on `idm_ref`.
The first 64 completed, event-free proposals in each of five modes formed a
320-scenario source-safe pool, before any target result was inspected. All
three parameter mutants were run on that same pool. Selectors saw target
labels only at their 50 charged, distinct queries. The first ten mode-support
queries count inside B=50. Ten `RandomSafe` orders were averaged within each
seed/target unit. The qualification seed and all validation source-only gates
passed without bound or seed adjustment.

| Method | New failures@50 | Ego collisions@50 | Failure modes@50 | CVS@50 |
|---|---:|---:|---:|---:|
| HistoryMargin-Static | 29.33 | 25.44 | 5.00 | 15.06 |
| HistoryMargin-Residual | 25.00 | 20.56 | 5.00 | 13.00 |
| RiskDiverse-Static | 23.89 | 20.33 | 5.00 | 14.83 |
| TargetOnly-Residual | 18.89 | 14.89 | 3.67 | 9.72 |
| RandomSafe | 6.92 | 5.49 | 3.51 | 5.59 |

On matched seed/target units, HistoryMargin-Residual minus
HistoryMargin-Static is **-4.33 new failures** (hierarchical bootstrap 95%
interval [-6.22, -2.22]) and **-4.89 ego collisions** ([-6.56, -2.89]). It
loses to static history on new failures for every parameter mutant averaged
over seeds: delay-only 22.33 vs 28.00, brake-only 16.67 vs 18.67, and
combined 36.00 vs 41.33. It beats TargetOnly-Residual by 6.11 failures
([4.56, 8.00]) and RandomSafe by 18.08 ([15.10, 21.21]), but those weaker
comparators do not rescue the required improvement over static history.

The complete 320-scenario pools contain 20-78 new failures per target/seed;
no zero-opportunity unit was omitted. Historical continuous TTC-response
ranking alone has ROC AUC 0.879-0.947 (mean 0.923) against the full-pool
target-failure labels, a *post hoc diagnostic*, not a permitted selection
input. This helps explain why a static margin is already strong. The residual
model may be correcting a useful ranking with noisy early labels; that is a
hypothesis requiring a new development and held-out validation, not a result
established by this study. The paired intervals describe these nine fixed
units and are not broad population guarantees.

The previous heterogeneous-controller v8 GIFs are **5 Hz external-action
examples**, not evidence that five collision modes survive a faster
controller. At 20 Hz high-level calls on those same five scenarios, only
`cutin_braking` remains an ego collision; `fast_intrusion` and
`lead_braking` become near misses, while `stop_and_go` and
`slow_lead_following` have no event. The GIF itself captures 20 physics
frames per second; playback smoothness cannot repair the 5 Hz controller
cadence. The new IDM experiment removes ego lane switching and updates its
longitudinal command every physics frame. Its own five-mode side-by-side
20 Hz replays are in `gifs/comparisons/idm_revision_manifest.json`; each is
the first collision charged to HistoryMargin-Residual in that mode, with a
three-timepoint timeline PNG and a full GIF. All five target collisions and
source-safe outcomes were checked during physical replay. These visualizations
show real simulator collision events but do not reverse the negative method
comparison above.

Verdict: the testing problem and historical-margin signal are credible in
this simulator. The proposed adaptive residual contribution is **not
validated** under the same-family, high-frequency setup. Do not claim an
effective or novel regression-testing method from this result. Any revised
selector must be developed on separate seeds, compared with direct published
baselines, and retested on new frozen validation seeds.

Raw evidence: `analysis50.json`, `records.csv`, and each seed's
`source_bank.npz`, `candidate_pool.npz`, `target_bank.npz`, and `gate.json`.
