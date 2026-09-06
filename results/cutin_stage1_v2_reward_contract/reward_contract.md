# Stage 1 v2 reward contract

This run keeps the existing `frenet_path_longitudinal_v2`, 31-D observation,
event definitions, reward coefficients, and Inner SAC settings unchanged.

## Micro transition

Each `env.step()` invokes `InnerRiskReward.step_with_components()` exactly once
after schedule and semantic state are refreshed.  The returned `reward_total`
is stored as `reward_inner`; the same call stores these auditable components in
the transition: `criticality_previous`, `criticality_current`, `reward_risk`,
`reward_event`, `reward_progress`, `penalty_tracking`, `penalty_shield`,
`penalty_invalid`, `reward_preclip`, `reward_clip_adjustment`, and
`reward_total`.

`reward_preclip = reward_risk + reward_event + reward_progress -
penalty_tracking - penalty_shield - penalty_invalid`; `reward_total` is that
value clipped to `[-3, 12]`.  Formal event labels remain explicit semantic and
execution-valid captures, never a reward threshold.

## Macro transition

One SAC decision block produces one replay row:

`R_k = sum_j gamma^j r_(t_k+j)`, `bootstrap_discount = gamma^H`, with
`gamma = 0.99` and actual block length `H`.  Replay retains the raw 4-D policy
action, while projected planner/control actions are diagnostic only.  Each
episode record includes a `macro_records` audit trace with its start micro
step, duration, undiscounted micro sum, discounted macro reward, discount,
event label, and termination reason.

## Episode metric

The publication curve is the undiscounted sum of all micro `reward_inner`
values in an episode.  It is neither a macro reward nor an SAC loss.  The
JSONL also records the separately discounted episode sum for audit.
