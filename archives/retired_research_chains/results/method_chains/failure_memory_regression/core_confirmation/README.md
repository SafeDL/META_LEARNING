# Core FBRT confirmation protocol

This protocol implements the algorithm and comparison rules in Sections 2–3
and 7 of `docs/FBRT_Optimization_Objectives.md`. It uses a shared target
failure posterior with the parent model as fixed offset. The directed model
uses same-parent local pass/fail edges; the center and coordinate models
replace only those edge features. `target_only` removes the parent risk offset.
Static risk and static boundary do not adapt to target feedback. All methods
use the same parent pools, fixed alternating schedule, 20 queries per
direction, and one target outcome per query. No coverage slot, direction
specific margin override, or neighborhood-frontier override is part of this
protocol.

NL-IDM and weight-inherited PPO each use the same three scenario families,
three newly frozen physical contexts per family, and complete 11×11 rule
grids: 1089 scenes and 3267 physical executions per release chain. The
scenario tuples are disjoint from every prior development, validation, and
confirmation manifest. The selector, physical runner, PPO loader and
checkpoint hashes are recorded in the child protocols before measuring target
responses.

The confirmatory test is complete. Across NL-IDM and PPO, the stated core
`directed_residual` method did not show consistent superiority over
`coordinate_residual` or static risk. NL-IDM's V0→V1 regression task had 16
changes, but the primary found 0; V1→V2 had 25 improvements, of which it found
13 versus 13 for coordinate residual and 14 for static risk. PPO V0→V1 had 26
improvements, of which the primary, coordinate residual, and static risk each
found 2. PPO V1→V2 had 1 regression and 20 improvements; the primary found
neither, while `target_only` found the one regression and 3 improvements.

The family-level tests are descriptive only (three families from one release
chain); adjusted p-values do not support a superiority claim. All complete
truth banks, query replays, prefix metrics, figures, and statistics are in the
`nl/` and `ppo/` directories. The confirmatory results are not overwritten by
the post-hoc development replays in their `exploratory/` directories.

The PPO `protocol.json` retained the inherited label `nl-bidirectional-v1`.
This is a metadata-only naming error: its frozen `builds`, manifest, PPO hashes,
and selector config identify the PPO chain. `ppo/metadata_erratum.md` records
the correction. New protocol generation now labels PPO chains explicitly.
