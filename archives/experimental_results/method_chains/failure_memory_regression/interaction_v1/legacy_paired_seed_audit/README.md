# Common-seed frozen-bank audit

The protocol in `../PAIRED_SEED_AUDIT_PROTOCOL.md` was written before this
replay. `manifest.json` records 21 completed tasks, 11,560 logical queries,
unchanged frozen input fingerprints, no missing cache tasks, and no new
physical episodes. Every regression method ran ten repeats at budget 20,
using the same seed within each task/repeat. For example, Memory, NoMemory,
and FailureDistance all used seed `3667806950` on repeat 0 of
`legacy_regression_seed4179801_merge_blind06`.
`seed_ledger.csv` retains all 578 method-run seeds; each of the 110
regression task/repeat groups contains five methods with one common seed.

For the ten nonempty comparable regression tasks, totals across ten repeats
per task are:

| Budget | Memory hits | NoMemory hits | Task-level two-sided p | Seed-cluster two-sided p |
|---:|---:|---:|---:|---:|
| 1 | 31 | 21 | 0.219 | 0.25 |
| 5 | 272 | 184 | 0.0078 | 0.25 |
| 10 | 543 | 406 | 0.0117 | 0.25 |
| 20 | 1056 | 900 | 0.0156 | 0.25 |

Memory's budget-20 mean exceeds NoMemory by 1.8, 1.1, and 2.3 discoveries
per task in the three legacy simulator-seed clusters. The compact regression
task ties. One legacy task loses 0.1 discovery on average; another ties.
`summary_by_task.csv` retains every task and repeat used for the reported
sign-flip calculations.

The task-level p-values treat different target builds on the same simulator
seed and candidate set as independent, which is too optimistic for a strong
claim. Repeated selector seeds are also not new physical episodes. The
cluster-level test uses the three legacy seeds plus the compact task and does
not establish significance. These frozen outcomes were already inspected
before this audit, so even a small p-value here would be exploratory. The
interaction redesign's newer target banks still do not show a Memory
discovery advantage: the age regressions have no target failures and the
guard-off diagnostic ties NoMemory. A prospective independent physical bank
with a nonempty relevant target failure pool is needed for the requested
significant-improvement claim.
