"""Recover paid feedback by checked replay without querying it a second time."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .query import QueryLedger
from .schema import QueryFeedback
from .storage import compact_feedback, read_json, write_json


class ReplayInterrupted(RuntimeError):
    """A requested test interruption after a successfully persisted paid response."""


def trajectory_hash(trajectory):
    return hashlib.sha256(
        json.dumps(trajectory, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def paid_feedback_hash(records):
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


class ResumableLedger(QueryLedger):
    """Rebuild selector state from its own saved responses; new queries still pay.

    paid_execution must read an existing physical record, never execute a scene.
    This is used with fixed execution banks and separately metered development
    caches. A failed physical call is recorded and cannot be silently resumed.
    """

    def __init__(
        self,
        candidates,
        budget,
        execute,
        *,
        checkpoint_path,
        specification,
        paid_execution,
        stop_after=None,
        **kwargs,
    ):
        super().__init__(candidates, budget, execute, **kwargs)
        self.checkpoint_path = Path(checkpoint_path)
        self.paid_execution = paid_execution
        self.stop_after = stop_after
        self.replayed_count = 0
        self.saved_count = 0
        self.trajectory_hashes = []
        self.specification = json.loads(
            json.dumps(
                {
                    "task": specification,
                    "budget": budget,
                    "account": self.account,
                    "joint_feedback": self.joint_feedback,
                    "candidates": [asdict(s) for s in self.candidates],
                    "domain": None if self.domain is None else self.domain.to_dict(),
                }
            )
        )
        if not self.checkpoint_path.exists():
            return
        saved = read_json(self.checkpoint_path)
        if saved["specification"] != self.specification:
            raise ValueError("Saved paid feedback uses a different task or candidate population")
        records = saved["query_ledger"]
        if saved.get("paid_feedback_hash") != paid_feedback_hash(records):
            raise ValueError("Saved paid feedback failed its integrity check")
        if (
            saved["status"] == "failed"
            or len(records) > budget
            or len(saved["trajectory_hashes"]) != len(records)
            or any(r["status"] != "complete" or r["call"] != i + 1 for i, r in enumerate(records))
        ):
            raise ValueError("Only completed paid responses may enter interruption recovery")
        for record, fingerprint in zip(records, saved["trajectory_hashes"]):
            index = record["index"]
            if (
                not 0 <= index < len(self.candidates)
                or record["scenario_id"] != self.candidates[index].scenario_id
            ):
                raise ValueError("Saved paid feedback uses another candidate")
            physical = paid_execution(index)
            if (
                trajectory_hash(physical.trajectory) != fingerprint
                or physical.feedback.scenario_id != record["scenario_id"]
                or physical.feedback.collision != record["feedback"]["collision"]
            ):
                raise ValueError("Saved paid physical execution changed before recovery")
            feedback = QueryFeedback.from_dict(
                {
                    **record["feedback"],
                    "trajectory": physical.trajectory if self.joint_feedback else None,
                }
            )
            self.records.append({**record, "feedback": feedback.to_dict()})
            self.trajectories.append(physical.trajectory)
            self.trajectory_hashes.append(fingerprint)
        self.saved_count = len(records)

    def _save(self, status):
        records = compact_feedback(self.records)
        write_json(
            self.checkpoint_path,
            {
                "specification": self.specification,
                "status": status,
                "query_ledger": records,
                "paid_feedback_hash": paid_feedback_hash(records),
                "trajectory_hashes": self.trajectory_hashes,
                "successful_paid_responses": len(self.trajectories),
            },
        )

    def query(self, index, *, purpose="target"):
        if self.replayed_count < self.saved_count:
            if type(index) is not int or not 0 <= index < len(self.candidates):
                raise ValueError("Invalid paid query")
            record = self.records[self.replayed_count]
            if record["index"] != index or record["purpose"] != purpose:
                raise ValueError("Recovered selector requested a different paid feedback sequence")
            self.replayed_count += 1
            return QueryFeedback.from_dict(record["feedback"])
        try:
            feedback = super().query(index, purpose=purpose)
        except Exception:
            self._save("failed")
            raise
        self.replayed_count += 1
        self.trajectory_hashes.append(trajectory_hash(self.trajectories[-1]))
        self._save("running")
        if self.stop_after is not None and len(self.records) == self.stop_after:
            self._save("interrupted")
            raise ReplayInterrupted(f"Interrupted after {self.stop_after} persisted paid responses")
        return feedback

    def finish(self):
        if self.replayed_count != self.budget or len(self.records) != self.budget:
            raise ValueError("Recovered run did not consume its complete registered budget")
        self._save("complete")
