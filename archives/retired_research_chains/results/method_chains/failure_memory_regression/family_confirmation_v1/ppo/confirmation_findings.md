# Four-family PPO confirmation

The fixed weight-inherited PPO checkpoints completed 600 valid physical
episodes over the 200-scene frozen manifest. The method list, selector and
runner snapshots, physical contexts, and checkpoint digests were fixed before
paired target outcomes were measured. No unknown labels remain.

| Update | Direction | True changes | Multi-frontier directed @20 (area) | Static risk | Static role coverage | Coordinate multi-frontier |
|---|---|---:|---:|---:|---:|---:|
| V0→V1 | Regression | 5 | 5 (0.257) | 0 | 5 (0.286) | 5 (0.257) |
| V0→V1 | Improvement | 2 | 0 (0.000) | 0 | 2 (0.024) | 0 (0.000) |
| V1→V2 | Regression | 0 | NA | NA | NA | NA |
| V1→V2 | Improvement | 4 | 4 (0.157) | 1 (0.019) | 0 | 4 (0.157) |

The first update's regressions occur in S08, and improvements in S02; the
second update's improvements also occur in S02. The proposed method is strong
on the first regression and second improvement tasks but misses every first
improvement, while static role coverage finds both. Its identical-acquisition
coordinate ablation ties it everywhere in this confirmation. Thus this
independent context test does **not** establish bidirectional or
directed-feature superiority. The absent second regression task is NA. The
full truth, logical query order, context and region summaries, physical cost
ledger, family-paired statistics, and figures are preserved here.
