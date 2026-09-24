# v4 sparse-continuous qualification protocol

v4 is a separate experiment from v2 and v3.  In v3's physical qualification,
`vi_ttc` produced a critical event on 498/500 candidates.  It is retained in
the repository's SUT qualification record, but excluded from this sparse-risk
test family because a constant-density source cannot support the stated sparse
mining question.

The remaining SUTs are IDM+MOBIL, MCTS-CV, and PPO-ECE.  v4 shifts only the
outcome-blind continuous headway ranges upward.  Seed `20280203` is used only
for qualification; development (`20280217`, `20280303`) and validation
(`20280317`, `20280331`, `20280414`) remain unexecuted at this decision point.

Earlier gate files are preserved.  `qualification_gate.json` freezes the
confirmation conditions: at least two targets must have branch-discordant
failures in two modes; no three-SUT subset can be jointly redundant; and a
target with at least ten events cannot put more than 75% in one gap quartile.
The minimum-event rule avoids interpreting three MCTS events as a population
concentration statistic.
