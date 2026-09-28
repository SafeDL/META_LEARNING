# Independent context replication of the bidirectional selector

## Why this replication

Stage five's primary method found 48 of 176 changes across the five
nonempty direction-by-transition tasks, compared with 45 for the strongest
predeclared single-method control. That is a modest aggregate signal. It also
lost on individual PPO transitions, and family-clustered inference over three
families was underpowered. This replication tests whether the aggregate
advantage persists on fresh fixed contexts from the same three supported
families.

The primary selector and all predeclared controls are unchanged. Each chain
uses three new fixed contexts per family, complete 11×11 grids, a 40-query
budget, and identical physical seeds for all methods. The new context tuples
were audited against every earlier frozen manifest. The NL and PPO protocols,
source snapshots, checkpoints, manifests, and configuration must be frozen
before any target outcomes are measured.

This is a replication, not a new parameter search. A positive result must be
reported with task-level outcomes and the family-cluster limitation; an
aggregate count alone cannot justify a general superiority claim.

## Results

Both protocols were frozen before measurement. Each manifest contains 1,089
scenes; source snapshots, manifests, protocols, and PPO checkpoint hashes
passed the audit. The overlap audit excluded paired NL/PPO manifests from the
same split and found no context overlap with earlier experiments. The complete
NL and PPO response banks each contain 3,267 physical episodes, and a replay
audit confirmed that evaluation added zero physical episodes.

The primary selector did not retain a consistent advantage on the fresh
contexts. It lost to coordinate role-gated search on NL V0→V1 regression,
NL V1→V2 improvement, PPO V0→V1 improvement, and PPO V1→V2 improvement; it
beat that control on PPO V1→V2 regression. The three-family statistics are
descriptive and inconclusive. Full findings, including post-confirmation
exploratory screening, are in [`findings.md`](findings.md).
