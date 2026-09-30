# v3 qualification protocol

The v3 continuous-borderline proposal uses seed `20271103` only for
qualification.  Development (`20271117`, `20271201`) and validation
(`20271215`, `20271229`, `20280112`) are disjoint and were not executed when
this decision was made.

`qualification_gate.json` preserves the initial all-source-safe scan.  It is
not a valid necessary condition for CoRe-Mine: CoRe represents one historical
response branch per source, so its relevant opportunity is a target failure
with *source disagreement* (at least one source has an event and at least one
is safe).  `qualification_gate_superseded.json` therefore freezes the appropriate
pre-confirmation gates:

1. At least two targets have source-discordant failures in at least two modes.
2. No three-SUT subset has all pairwise event Jaccard overlaps above 0.90.
3. No target places over 75% of critical events in one initial-gap quartile.

The qualification bank satisfies all three gates.  This correction changes
only the qualification interpretation, not the v3 candidate pool, SUT roster,
selection code, development protocol, validation seeds, or evaluation metrics.
