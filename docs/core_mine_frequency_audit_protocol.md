# Control-cadence sensitivity audit (diagnostic, not a new method validation)

Frozen before the 10/20 Hz outcomes are inspected. The v8 primary result uses
20 Hz physics and 5 Hz externally chosen high-level actions. This audit asks
whether its already selected target failures persist when the *same* MCTS-CV
or VI/TTC implementation is called at 10 or 20 Hz, with physics fixed at
20 Hz. It does not alter v8's B=50 campaigns, qualification gate, or source
eligibility. It does not claim that changing policy frequency is a matched
software revision or a fair new source-safe study.

The fixed audit cohort is the 50 distinct scenarios selected by
HistoryMargin-Residual for each MCTS-CV and VI/TTC validation unit (three
seeds x two targets), 300 scenario-target pairs. Re-run each pair at 10 and
20 Hz with unchanged scenario parameters, simulator seed (`bank_seed + index`),
policy code, 7-second horizon, 20 Hz physics, and corrected ego-collision /
near-miss definitions. The existing 5 Hz bank is the reference; five displayed
collisions also receive a direct 5 Hz reproduction check before the broad
audit. No scenarios are selected or dropped after frequency outcomes are seen.

Report, separately for MCTS-CV and VI/TTC: original 5 Hz failure/collision
counts; 10/20 Hz failure/collision counts; number and fraction of 5 Hz failures
that persist; and newly occurring failures among the same selected points.
The historic controllers are not re-evaluated at the new frequencies, so
10/20 Hz outcomes must not be called source-safe discoveries. PPO is excluded
from cadence manipulation because its saved policy was trained at a fixed
control cadence. MCTS-CV also uses a one-second internal rollout increment,
so frequency sensitivity is partly a model-time-scale mismatch, not solely
reaction latency. Formal higher-frequency efficacy would require a matched
source/target cohort and a cadence-consistent controller implementation.
