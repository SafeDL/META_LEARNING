# Frozen-bank common-seed audit protocol

This audit is exploratory because the target labels in the frozen legacy bank
have already been inspected. It must not be used as confirmatory proof of a
new algorithm advantage.

Before running the new replay, the protocol is fixed as follows:

- Use every available frozen regression task, including zero-failure tasks,
  without filtering by method outcomes. Cross-agent sequences remain in the
  replay but are excluded from paired inference because their predecessor
  histories differ by method.
- Run ten repeats for every regression method at budget 20. In each task and
  repeat, use the same deterministic seed for every method. All methods use
  the same candidate list, target oracle, history snapshot, reward, and
  checkpoints (1, 5, 10, 20).
- Primary descriptive comparison: paired Memory minus NoMemory valid
  discoveries at budget 20, averaged across repeats within each task.
  Report all budget checkpoints and all task-level losses as well as gains.
- For uncertainty, treat the three legacy simulator seeds as clusters and
  the compact regression task as a fourth cluster. Report the exact
  two-sided sign-flip result on cluster means, excluding zero-failure tasks
  where discovery is impossible. Do not claim significance from repeated
  selector seeds as though they were new independent physical episodes.
- Write to `legacy_paired_seed_audit/`; do not overwrite prior replay outputs
  or run any new physical episode.

Execution command:

```powershell
conda run -n metadrive python -m method_chains.failure_memory_regression.replay --offline-only --paired-repeats 10 --output results/method_chains/failure_memory_regression/memory_exploit --test-result '41 passed'
```
