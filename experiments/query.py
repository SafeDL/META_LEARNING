"""Evaluator-owned execution and process-isolated, metered query transport."""

import multiprocessing as mp
import time
import traceback
from dataclasses import asdict, dataclass

from threadpoolctl import threadpool_limits

from .scenarios import validate
from .schema import QueryFeedback, Scenario


class QueryClient:
    def __init__(self, connection):
        self.connection = connection

    def query(self, index, *, purpose="target"):
        self.connection.send(("query", (index, purpose)))
        kind, payload = self.connection.recv()
        if kind == "error":
            raise RuntimeError(payload)
        if kind != "feedback":
            raise RuntimeError("Unexpected evaluator response")
        return QueryFeedback.from_dict(payload)

    def register(self, scenario):
        self.connection.send(("register", asdict(scenario)))
        kind, payload = self.connection.recv()
        if kind == "error":
            raise RuntimeError(payload)
        if kind != "registered":
            raise RuntimeError("Unexpected candidate registration response")
        return int(payload)


@dataclass
class PaidExecution:
    feedback: QueryFeedback
    trajectory: list[dict]


class QueryLedger:
    def __init__(self, candidates, budget, execute, *, account="target"):
        self.candidates = tuple(candidates)
        self.budget = budget
        self.execute = execute
        self.account = account
        self.records = []
        self.trajectories = []

    def register(self, scenario):
        validate(scenario)
        if self.candidates and scenario.kind != self.candidates[0].kind:
            raise ValueError(
                "Continuous candidates must stay in the registered functional scenario"
            )
        for index, existing in enumerate(self.candidates):
            if existing.scenario_id == scenario.scenario_id:
                if existing != scenario:
                    raise ValueError("Candidate identifier was reused for a different coordinate")
                return index
            if existing.coordinate() == scenario.coordinate():
                raise ValueError(
                    "Repeated coordinates must query the existing candidate and pay for a repeat"
                )
        self.candidates = (*self.candidates, scenario)
        return len(self.candidates) - 1

    def query(self, index, *, purpose="target"):
        if type(index) is not int or not 0 <= index < len(self.candidates):
            raise ValueError("Invalid paid query")
        if len(self.records) >= self.budget:
            raise ValueError("Paid execution budget exhausted")
        if purpose not in ("initial", "target", "repeat", "history_certification"):
            raise ValueError("Unknown execution purpose")
        if (self.account == "target" and purpose == "history_certification") or (
            self.account == "history_certification"
            and purpose not in ("history_certification", "repeat")
        ):
            raise ValueError("Historical certification requires its own paid account")
        repeated = any(r["index"] == index for r in self.records)
        if repeated != (purpose == "repeat"):
            raise ValueError("Repeated execution must be explicitly requested and paid")
        scenario = self.candidates[index]
        record = {
            "call": len(self.records) + 1,
            "index": index,
            "scenario_id": scenario.scenario_id,
            "purpose": purpose,
            "account": self.account,
            "status": "running",
        }
        self.records.append(record)
        started = time.perf_counter()
        try:
            execution = self.execute(scenario)
            if execution.feedback.scenario_id != scenario.scenario_id:
                raise ValueError("Executor returned a different scene")
        except Exception as error:
            record.update(status="failed", wall_s=time.perf_counter() - started, error=repr(error))
            raise
        record.update(
            status="complete",
            wall_s=time.perf_counter() - started,
            feedback=execution.feedback.to_dict(),
        )
        self.trajectories.append(execution.trajectory)
        return execution.feedback


def _worker(connection, selector, public_task):
    try:
        with threadpool_limits(1):
            result = selector(oracle=QueryClient(connection), **public_task)
        connection.send(("result", result))
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def run_isolated(selector, public_task, ledger):
    """Only explicitly assembled public inputs are serialized into the selector."""
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_worker, args=(child, selector, public_task))
    process.start()
    child.close()
    try:
        while True:
            if not parent.poll(1):
                if not process.is_alive():
                    raise RuntimeError(f"Selector exited without a result: {process.exitcode}")
                continue
            kind, payload = parent.recv()
            if kind == "register":
                try:
                    index = ledger.register(Scenario(**payload))
                    parent.send(("registered", index))
                except Exception:
                    parent.send(("error", traceback.format_exc()))
                    raise
            elif kind == "query":
                index, purpose = payload
                try:
                    feedback = ledger.query(index, purpose=purpose)
                    parent.send(("feedback", feedback.to_dict()))
                except Exception:
                    parent.send(("error", traceback.format_exc()))
                    raise
            elif kind == "result":
                if len(ledger.records) != ledger.budget or payload["selected_indices"] != [
                    r["index"] for r in ledger.records
                ]:
                    raise ValueError("Selector result differs from the complete paid query ledger")
                if len(payload["queries"]) != len(ledger.records) or any(
                    q["feedback"] != record["feedback"]
                    for q, record in zip(payload["queries"], ledger.records)
                ):
                    raise ValueError("Selector feedback differs from evaluator disclosures")
                return payload
            elif kind == "error":
                raise RuntimeError(payload)
            else:
                raise RuntimeError(f"Unknown selector message: {kind}")
    finally:
        parent.close()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join()
