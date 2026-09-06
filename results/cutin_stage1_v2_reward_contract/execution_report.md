# Stage 1 v2 reward-contract execution report

## Scope and status

- `reward_contract_status`: passed — unit checks cover the one-call component
  interface, reset behavior, explicit event labels, and discounted macro
  aggregation.
- `training_status`: completed — one fixed interaction-prior run only.
- `domain_coverage_status`: failed — the deterministic Stage 1 gate passed
  `0/3` Logical Domains because every domain had zero valid events, despite
  valid rate `1.0` in each domain.
- `paired_gain_status`: inconclusive — nine fixed, paired development cases
  are too small for statistical claims; their observed event-rate differences
  are recorded below.

## Executed budget

The run used Cut-in / `idm_normal` / `cutin-g01`, one shared Inner SAC, three
Logical Domains, 40 episodes per domain, five warm-up episodes per domain,
batch size 64, 12 update calls per post-warm-up episode, learning rate 0.0001,
and gamma 0.99.  It consumed 120 episodes, 31,899 environment steps, 6,453
macro transitions, and 1,260 optimizer updates.  No context-meta or Outer
stage was run.

## Reward and control audit

`reward_episode_metrics.jsonl` contains 120 compact episode records and all
required reward-component sums.  Across the run there were 6 clipped micro
rewards, 923 path-speed-infeasible diagnostic steps, 4,786 path-projection
interventions, 255 speed-envelope longitudinal interventions, and 29,640
jerk-projector longitudinal interventions.  These are execution diagnostics,
not success criteria.

Training-time valid-event episodes were 19/40 in `close_closing_early`, 20/40
in `balanced_interaction`, and 7/40 in `late_tight_cutin`.  They do not replace
the fixed-seed post-training gate.

## Deterministic gate

For the centre action, candidate 0, and seeds 11/22/33, all three domains had
valid rate 1.0 and event rate 0.0.  The formal per-domain event requirement was
therefore unmet in every domain; Stage 1 remains blocked.

## Paired SAC/Random engineering comparison

Each domain used three pre-registered in-domain cases; SAC and Random received
the same candidate, Logical action, episode seed, horizon, event semantics,
and v2 executor.  SAC had 0/3 valid events in every domain.  Random had 1/3
events in `balanced_interaction` (collision), 1/3 in
`close_closing_early` (critical near-miss), and 0/3 in
`late_tight_cutin`.  Valid rates were 1.0 for both policies in all domains.
The observed SAC-minus-Random event-rate gains were -1/3, -1/3, and 0,
respectively.  This small batch is reported as inconclusive and does not
support a claim of adversarial improvement.

## Reproducibility and artifacts

Training command:

`conda run -n metadrive python -m mvr.scripts.train_mvr --config mvr/configs/cutin_inner.yaml --output results/cutin_stage1_v2_reward_contract --stop-after interaction_prior`

The output directory contains the checkpoint, manifest, full training metrics,
reward JSONL, gate, fixed paired cases/report, and the 8-episode trailing-mean
PDF/600-dpi PNG.  `run_metadata.json` records the actual commit, dirty-tree
state, configuration, action schema, reward schema, seed, and completed stage.
