# Highway-env SUT qualification report

| SUT | Qualified driving | Common completed | Ego collisions | Decision |
|---|---:|---:|---:|---|
| idm_mobil | 20/20 | 85/96 | 11/96 | retain |
| vi_ttc | 20/20 | 76/96 | 20/96 | retain |
| mcts_cv | 20/20 | 82/96 | 14/96 | retain |
| ppo_ece | 20/20 | 91/96 | 5/96 | retain |

DQN-ECE was rejected after 20 native loading failures. Double DQN-ECE was rejected after 33 ego collisions in 96 common scenarios. Their runtime integrations and checkpoints were removed; the aggregate decisions and checkpoint metadata are retained.

MCTS-CV uses a constant-velocity prediction model and has no access to the scheduled future traffic state.
