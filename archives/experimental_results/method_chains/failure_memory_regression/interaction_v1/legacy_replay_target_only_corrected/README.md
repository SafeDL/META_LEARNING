# Target-only branch correction audit

This is an offline replay of the original frozen banks. The selector now uses its independent
target-only feature dictionary whenever compatible historical failures are
absent. Previously it set the historical mixture weight to zero while still
fitting and ranking with source-derived coverage and margin features.

The replay completed 21 tasks and 3,640 logical queries with no new physical
episodes. `manifest.json` reports unchanged input bank fingerprints and no
missing cache tasks. Comparing `summary_by_task.csv` against the prior replay
shows zero changed discovery counts at budgets 1, 5, 10, and 20 for Memory and
NoMemory. The overall empirical effect remains mixed. A same-seed ten-query
unit test confirms that pass-only Memory and NoMemory choose exactly the same
scenarios, including after target failures introduce new pattern features.
The subsequent full suite passed 40 tests; the generated replay report records
the 39-test suite that had passed at replay time.

The companion interaction evaluation now uses the common `is_parent_pass`
predicate. Its five cached evaluations retain the same discovery counts; one
budget-10 file changed query order without changing rewards. No target outcome
was used to select a new scenario or mutation parameter.
