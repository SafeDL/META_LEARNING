# Highway-env DIVA-Mine E1--E6 outcome

This record follows the original Highway-env MVP result in `results/diva_highway/cutin_mvp/`. The original result directory is retained unchanged as E1 evidence; the independent E2 controller-bank result is in `results/diva_highway/cutin_mvp_e2/`.

## E1: sequential, boundary-aware diagnostic support

The former one-shot variance-only support selection was replaced by a sequence of four probes. After every probe, the rank-2 Gaussian posterior is updated from only the revealed target outcome. The next probe maximizes posterior predictive information gain multiplied by proximity to the fixed collision/near-miss boundary, `vulnerability = 0.75`.

The existing 6-by-128 bank was replayed offline. The sequential rule matches the retired variance-only rule at mean NDCG@10 **0.987139**, but does not improve on random support in a majority of held-out folds. Its CriticalScore@20 is **16.250**, below the Shared Prior's **17.917**. E1 therefore fails its diagnostic-gain criterion and triggers E2; no new simulation was required for this replay.

## E2: independent heterogeneous controller bank

The new bank retains the 128 shared Sobol Cut-in anchors and two scenario variables, but uses three IDM and three Full Velocity Difference Model (FVDM) controllers. It records exactly `6 × 128 = 768` `(SUT, anchor)` responses in `response_bank_highway_e2.npz`.

| SUT | Controller | Collision | Near miss | Failure rate |
|---|---|---:|---:|---:|
| SUT-A | IDM | 22 | 5 | 21.09% |
| SUT-B | IDM | 10 | 10 | 15.62% |
| SUT-C | IDM | 27 | 5 | 25.00% |
| SUT-D | FVDM | 22 | 21 | 33.59% |
| SUT-E | FVDM | 15 | 6 | 16.41% |
| SUT-F | FVDM | 36 | 27 | 49.22% |

Every SUT is within the required 5%–50% failure-response band. The data also have non-nested risk regions: SUT-A versus SUT-D has 1 anchor with `A − D ≥ 0.2` and 17 with `D − A ≥ 0.2`; SUT-C versus SUT-D has 4 and 14 respectively. Rank-2 EVR is 100% in every LOSO source fold. E2 passes.

## E3: K=4 LOSO ranking

| Method | Mean NDCG@10 |
|---|---:|
| Shared Prior | 0.981283 |
| Random Support + Adaptation | 0.965408 |
| Highest-Risk Support + Adaptation | 0.981283 |
| DIVA Diagnostic + Adaptation | 0.981283 |

DIVA matches the Shared Prior on all six held-out SUTs. It has neither the required mean gain of `+0.02` nor four strict per-SUT gains, so E3 fails. The response bank is not too sparse or homogeneous under E2's checks; rather, the four selected target outcomes do not change this rank-2 posterior's top-10 ranking enough to beat the already strong shared prior.

## E4: B=20 mining evidence

| Method | Mean CriticalScore@20 |
|---|---:|
| Random | 4.346 |
| Shared Prior | 18.167 |
| Highest-Risk Support + Adaptation | 18.167 |
| Random Support + Adaptation | 15.658 |
| DIVA Diagnostic + Adaptation | 16.250 |

E4 was replayed from the completed bank for diagnostic evidence. DIVA is 10.6% below Shared Prior and also below Random Support, so it fails the required `+5%` improvement and 4-of-6 target wins. Because E3 failed, this is not a valid successful downstream mining claim.

## E5: offline Oracle Headroom audit on E2

E5 uses the completed E2 bank only; it triggers **zero** new Highway-env
episodes. `Oracle Latent` projects all 128 held-out vulnerability values onto
the LOSO rank-2 basis, which tests representational headroom. `Oracle K=4`
inspects every candidate target outcome at each selection step and greedily
maximizes the resulting NDCG@10. Both are diagnostic upper bounds, not
baselines and not deployable methods. The four support probes remain part of
the B=20 critical-score budget.

| Method | Mean NDCG@10 | Mean CriticalScore@20 |
|---|---:|---:|
| Shared Prior | 0.981283 | 18.167 |
| Oracle Latent (full target) | 0.981283 | n/a |
| Oracle K=4 (ranking upper bound) | 0.981283 | 15.917 |

Oracle K=4 produces **0/6** strict NDCG wins and **0/6** strict critical-score
wins. Its CriticalScore@20 is 12.4% below the Shared Prior. Oracle Latent also
does not strictly improve mean NDCG. Thus the E2 failure is not caused by a
suboptimal diagnostic acquisition rule: under this low-rank representation,
the single-mode Cut-in bank has no usable target-adaptation headroom.

The former E3 absolute NDCG threshold (`+0.02`) is retired. With Shared Prior
NDCG@10 = 0.981283, its mathematical maximum possible gain is only 0.018717.
Any later ranking gate must instead use normalized remaining-headroom recovery,
`(NDCG_adapted - NDCG_shared) / (1 - NDCG_shared)`, with a 25% threshold and
four positive target-SUT gains.

## E6: dual-mechanism Cut-in bank and Gate 0

Per the E5 stop rule, the scenario was changed without changing the learning
algorithm, total anchor count, SUTs, or fixed-budget protocol. The new bank
contains exactly 64 `fast_intrusion` anchors (0.45 s lane change, then constant
speed) and 64 `cutin_braking` anchors (1.5 s lane change, then fixed 4.5 m/s^2
braking for 1 s), for another exact `6 x 128 = 768` episodes.

Gate 0-A passes: all target failure rates lie in [14.06%, 50.00%]. Gate 0-B
passes: each LOSO source fold has rank-2 EVR between 87.57% and 93.28%.
Gate 0-C fails: even an outcome-oracle top-20 ranking has less than 10%
CriticalScore headroom over the Shared Prior for every target SUT (the maximum
is 7.69% for SUT-B; the requirement is 4/6 targets at at least 10%).

This failure is decisive for this dual-mode configuration: knowing the target
SUT exactly cannot materially increase fixed-budget failure discovery, so no
support acquisition method can establish the claimed adaptation gain. E7 and
E8 were therefore not run.

## Final decision

The predefined stop rule applies. The Highway-env Cut-in MVP establishes
transferable low-rank structure and physically valid heterogeneous failure
regions, but does **not** establish the DIVA-Mine adaptation advantage. Do not
claim a successful DIVA method result, and do not expand anchors, scan `K=1/2`,
add a larger surrogate, change geometry, or transfer this method to MetaDrive
on the basis of these data.

The E2 directory contains the response bank, E1 comparison CSV, LOSO
ranking/mining CSVs, figures, replays, and the E5 oracle-headroom CSV/JSON.
The E6 directory contains the dual-mode response bank and the Gate 0 audit.
