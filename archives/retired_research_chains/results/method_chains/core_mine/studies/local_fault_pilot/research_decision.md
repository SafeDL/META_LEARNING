# Local-fault pilot: opportunity gate failed

The three controller faults, trigger constants, scenario set, and gate were
frozen in `docs/core_mine_local_fault_pilot_protocol.md` before target
execution. The fixed 80-scenario source set was rerun and all 80 remained
completed and event-free. Each local fault physically executed the same 80
scenarios at 20 Hz ego control and 20 Hz physics; no target-dependent
scenario filtering was applied.

| Seeded local fault | Triggered scenarios | New failures | Ego collisions | Failure modes |
|---|---:|---:|---:|---|
| 0.60 s cut-in association delay | 32 | 1 | 1 | fast intrusion |
| 0.80 s cut-in brake cap | 32 | 1 | 1 | fast intrusion |
| Brake cap while observed front speed < 18 m/s | 48 | 2 | 2 | fast intrusion, slow lead following |

The predeclared gate required at least two fixed mutants to cause failures
in at least two modes. It failed. **Stop this fault family without adjusting
the trigger thresholds, durations, candidate bounds, or mode mix.** The 80
cases are an opportunity pilot, not a B=50 selector validation; they cannot
support an efficacy or novelty claim. Raw physical observations are in
`records.csv` and the machine-readable counts are in `summary.json`.

This result also guards against a tempting but invalid narrative: a local
fault is not automatically an informative regression benchmark. Most of
these previously safe scenes remained safe despite the fault activation.
