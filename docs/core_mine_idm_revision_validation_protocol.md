# Frozen same-family, 20 Hz configuration-regression validation

Frozen after the two separate 80-scenario development pilots and before any
outcome at the seeds below is inspected. The frozen version changes are the
four IDM builds in `docs/core_mine_idm_revision_pilot_protocol.md`. They are
seeded parameter mutations, not actual commercial software releases.
Physics and low-level ego control run at 20 Hz for every build; the ego never
changes lane. This study tests longitudinal/cut-in failure emergence.

## Source-only candidate construction

The reference release `idm_ref` is the sole historical source. For each seed,
generate 128 scrambled Sobol scenarios in each of five modes, independently
seeded by `bank_seed + mode_offset`, with fixed gap and relative-speed bounds
(m; m/s): fast intrusion [10,50],[-8,-2]; cut-in braking [10,50],[-7,-1];
lead braking [12,50],[-7,-1]; stop-and-go [10,50],[-6,-1]; slow-lead
following [12,50],[-8,-2]. Timing/intensity vary over [0.05,0.95]. Execute
all 640 scenarios on the reference controller. In original Sobol order, take
the first 64 *completed, event-free* source scenarios per mode. The resulting
320-candidate pool is fixed before executing any mutant. Do not use source
margin to filter or rank the candidate pool; that signal is reserved for
the competing selectors. Every target and method receives the same 320
candidate indices and source responses. Exact simulator seed is
`bank_seed + original_640_pool_index`.

Qualification seed: **20291230**. If any mode has fewer than 64 completed,
event-free source scenarios, stop the study and report failure; do not
replace the seed or change bounds. Qualification does not require a minimum
mutant failure count. Validation seeds, never used for tuning, are
**20300109, 20300123, 20300206**. Apply the same source-only construction
and gate to each. If a seed fails, report it and stop rather than silently
dropping it.

## Compared selectors and endpoints

Evaluate all three targets: `idm_delay07`, `idm_brake3`, and
`idm_delay07_brake3`. Execute each target once on all 320 selected candidates
to create a physically grounded outcome bank. The selector sees source
responses and only those target responses it queries. Each method is charged
exactly 50 distinct target scenarios; its first ten mode-support queries
count toward the budget. The frozen GP settings are residual length 0.30,
amplitude 0.15, noise 0.05. The matched methods are:

- `HistoryMargin-Residual`: source continuous TTC response plus local GP
  correction, then highest predicted target-event probability.
- `HistoryMargin-Static`: identical source response without correction.
- `TargetOnly-Residual`: identical GP and feedback with a constant initial
  response and no source margin.
- `RiskDiverse-Static`: a fixed 50/50 rank-normalized blend of source margin
  and mode-aware farthest-point distance to already chosen tests; it is a
  *risk/diversity heuristic*, not a faithful implementation of SPECTRE.
- `RandomSafe`: ten seeded repetitions, averaged within each target/seed.

The primary endpoint is new target failures at B=50, counting ego collision
or (without ego collision) TTC < 1.5 s or vehicle-polygon clearance < 1 m.
Report ego collisions, event modes, CVS, early discovery, every target's
complete-pool opportunity and B=50 recall, all per-seed results, and paired
intervals. A method claim requires improvement over **both** static history
and target-only learning without hiding zero- or low-opportunity versions.
No real-world safety or novelty claim follows automatically from a positive
result; direct published-method comparisons remain a separate requirement.
