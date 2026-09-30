# v7 qualification decision

The independent qualification seed 20290325 failed the prespecified
one-dimensional gap-dispersion gate. Corrected critical-event counts per
320-candidate target were 31 (idm_mobil), 57 (mcts_cv), 27 (ppo_ece) and 86
(vi_ttc); ego collisions were 22, 29, 9 and 45. All four targets had
source-discordant events in at least two functional modes, and no controller
triplet was redundant. Only IDM exceeded the 75% maximum: 80.6% of its
events were in the shortest-gap quartile. The full gate result is in
`qualification/qualification_gate.json`.

No v7 development or validation banks will be generated. This qualification
bank is development information for a separately versioned protocol. Future
gate design should test headroom in nonredundant discovered events directly;
the gap quartile concentration remains a descriptive diagnostic.
