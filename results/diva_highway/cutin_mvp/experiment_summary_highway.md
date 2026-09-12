# Highway-env DIVA-Mine MVP: first mechanism check

Configuration: six IDM-style SUT profiles, 128 shared Sobol Cut-in anchors, 2-D scenario coordinates (`initial_gap`, `relative_speed`), rank-2 source-only SVD, LOSO split, K=4 support, B=20 target tests, and 20 random repeats.

## Integrity checks

- The held-out target row is excluded from every prior fit.
- Only the K selected support outcomes enter the posterior update.
- Support queries are included in B=20, and no scenario is queried twice.
- `mvr/tests/test_diva_highway.py` and `mvr/tests/test_diva_highway_cutin.py` pass; the saved bank has shape `(6, 128)`.

## Results

The mean LOSO rank-2 explained-variance ratio is **95.55%** (per fold: 97.7%, 93.9%, 97.5%, 93.8%, 95.5%, 94.9%). Thus, a strong low-rank structure exists.

| Method | NDCG@10 | Critical Score@20 |
|---|---:|---:|
| Random | — | 3.654 ± 2.009 |
| Shared Prior | 0.987 ± 0.029 | 17.917 ± 2.050 |
| Random Support + Adaptation | 0.987 ± 0.029 | 15.858 ± 1.577 |
| Diagnostic Support + Adaptation | 0.987 ± 0.029 | 16.167 ± 2.357 |

## Decision

This first Highway-env mechanism check does **not** establish the DIVA-Mine advantage. The common prior nearly saturates top-10 ranking quality, and its fixed-budget critical score exceeds both adapted methods. Per the experiment design's stop rule, the appropriate next step is to create genuinely non-nested SUT failure modes or revise the diagnostic acquisition criterion, rather than introduce a more complex surrogate model.
