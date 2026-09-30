# Additional-family development survey

The first 11-family manifest in `nl/` and `ppo/` was frozen before a runner
capability check. Its first S03 execution stopped with an explicit unsupported
template error; **no physical response rows were generated**. It is retained
as an aborted design, not as evidence.

`nl_supported/` and `ppo_supported/` are a revised, development-only survey
using five executable scenario families (S03, S04, S05, S06, S09). S03/S04/S09
were added to the unified runner before the revised manifests were frozen.
These data may guide method design, but any method chosen from them requires
fresh physical contexts for confirmation.

The revised survey completed 45 scenes and 135 physical episodes per SUT
chain. NL V0→V1 had six regressions, all in S04, and no improvements; NL
V1→V2 had no flips. PPO had no flips in either update. Every deterministic
comparator found all six NL regressions within the budget, so this survey
provides no method advantage. The frozen selector source originally wrote
nonfinite missing-TTC audit scores, which stopped JSON report generation;
the replay code was patched to serialize those scores as null without changing
query selection. This post-measure reporting fix is covered by a regression
test. The frozen original source and pre-fix digest remain in each survey root.
