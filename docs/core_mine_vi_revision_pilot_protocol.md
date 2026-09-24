# Same-family VI revision opportunity pilot (development only)

Freeze the three changes below before inspecting any new target outcomes.
They are explicitly labelled **seeded planner-configuration mutations**, not
historically deployed software releases. All are variants of the same
value-iteration TTC planner and run with 20 Hz physics and 20 Hz decisions.

| Build | Change relative to reference | Plausible regression mechanism |
|---|---|---|
| `vi_risk035` | collision reward multiplied by 0.35 | safety penalty underweighted after reward retuning |
| `vi_coarse2` | TTC grid quantization 2.0 s instead of 1.0 s | computation-saving temporal coarsening |
| `vi_lane10` | lane-change penalty multiplied by 10 | comfort retuning suppresses evasive changes |

The reference is the unmodified `ValueIterationPolicy` (gamma 0.9,
60 iterations, 1.0 s TTC grid) at 20 Hz. A new `VersionedVIPolicy` adapter
must reproduce reference actions and physical outcomes when all modifiers
equal one. Use the fixed v8 five-mode candidate generator at development seed
20291118 and take the first 16 scrambled-Sobol candidates in each mode,
80 scenarios total. Execute reference and all three frozen variants on every
scenario with identical simulator seed (`bank_seed + original index`). Define
physical events as ego collision or, without ego collision, TTC < 1.5 s or
vehicle-polygon clearance < 1 m. Do not count background-only collisions.

Report reference-safe eligible count/modes and, for *all three variants*,
new failures and ego collisions inside that eligible set, with per-mode
counts. This pilot is only an opportunity and implementation-validity check;
it does not compare selection methods or establish novelty. A subsequent
validation must freeze fresh seeds, the whole 320-candidate pool, B=50,
source-safe eligibility, and direct existing-method baselines before looking
at its target outcomes. Zero-opportunity variants must remain reported.
