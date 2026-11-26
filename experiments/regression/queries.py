"""Process-isolated random queries for the independent policy reference chain."""

import time
from pathlib import Path

import numpy as np

from experiments.costs import measure_resources
from experiments.query import PaidExecution, run_isolated
from experiments.regression.execution import checkpoint_identity, execute_record
from experiments.regression.protocol import COLLISION_TARGET, QUERY_LIMIT, stop_reason
from experiments.resume import ResumableLedger
from experiments.schema import QueryFeedback
from experiments.storage import compact_result, read_json, write_json


class CollisionStoppingLedger(ResumableLedger):
    def query(self, index, *, purpose="target"):
        feedback = super().query(index, purpose=purpose)
        if sum(r["feedback"]["collision"] for r in self.records) >= COLLISION_TARGET:
            self.budget = len(self.records)
        return feedback


@measure_resources
def random_reference_selector(candidates, oracle, *, initial_indices, budget):
    selected, queries = [], []

    def query(index, initial=False):
        feedback = oracle.query(int(index), purpose="initial" if initial else "target")
        selected.append(int(index))
        queries.append({"query": len(queries) + 1, "index": int(index),
                        "feedback": feedback.to_dict()})

    for index in initial_indices:
        query(index, True)
    started = time.perf_counter()
    used = set(initial_indices)
    order = [int(i) for i in np.random.default_rng(11).permutation(len(candidates)) if i not in used]
    ranking_s = time.perf_counter() - started
    for index in order:
        if len(queries) >= budget or stop_reason(sum(q["feedback"]["collision"] for q in queries), len(queries)):
            break
        query(index)
    return {"method": "random", "queries": queries, "selected_indices": selected,
            "actual_queries": len(queries), "maximum_queries": budget,
            "termination": stop_reason(sum(q["feedback"]["collision"] for q in queries), len(queries)),
            "ranking_s": ranking_s}


def query_reference_round(candidates, initial, checkpoint, directory, *, iteration, execution_root,
                          policy="ppo"):
    directory = Path(directory)
    execution_root = Path(execution_root)
    result_path = directory / f"result_{iteration}.json"
    identity = checkpoint_identity(checkpoint)
    specification = {"method": "random", "checkpoint_sha256": identity,
                     "collision_target": COLLISION_TARGET, "query_limit": QUERY_LIMIT,
                     "history_fingerprints": None}
    public = {"candidates": candidates, "initial_indices": initial, "budget": QUERY_LIMIT}

    def physical_path(index):
        candidate = candidates[index]
        return execution_root / "query_executions" / f"{identity}_{candidate.scenario_id.rsplit(':', 1)[-1]}.json.gz"

    def paid(index):
        saved = read_json(physical_path(index))
        return PaidExecution(QueryFeedback.from_dict(saved["feedback"]), saved["trajectory"])

    def execute(scene):
        index = next(i for i, candidate in enumerate(candidates) if candidate == scene)
        saved = execute_record(checkpoint, scene, physical_path(index), purpose="query", policy=policy)
        return PaidExecution(QueryFeedback.from_dict(saved["feedback"]), saved["trajectory"])

    ledger = CollisionStoppingLedger(
        candidates, QUERY_LIMIT, execute, checkpoint_path=directory / f"ledger_{iteration}.json",
        specification=specification, paid_execution=paid, joint_feedback=True,
        allow_registration=False,
    )
    if result_path.exists():
        saved = read_json(result_path)
        if saved["specification"] != specification:
            raise ValueError("Completed query result belongs to another method or policy")
        return saved
    result = run_isolated(random_reference_selector, public, ledger)
    collisions = sum(r["feedback"]["collision"] for r in ledger.records)
    reason = stop_reason(collisions, len(ledger.records))
    if reason is None or collisions > COLLISION_TARGET:
        raise ValueError("Selector did not obey the registered collision and query limits")
    ledger.finish()
    saved = {"specification": specification, "status": "complete", "collisions": collisions,
             "actual_queries": len(ledger.records), "stop_reason": reason,
             "result": compact_result(result)}
    write_json(result_path, saved)
    return saved
