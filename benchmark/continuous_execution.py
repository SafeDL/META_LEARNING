"""Persistent actual calls; replaying an interrupted prefix does not execute it twice."""

from dataclasses import asdict
import gzip
import json
from pathlib import Path
import time

from controllers.highway.catalog import BY_ID
from controllers.highway.factory import create
from simulation.highway.collision_env import execute

from .execution_events import analyze_events
from .executions import execution_contract, require_qualified
from .query import PaidExecution
from .schema import QueryFeedback
from .protocol import DEFAULT


class ContinuousExecutor:
    def __init__(self, sut_id, destination, identity, *, environment_seed=81001):
        sut = BY_ID[sut_id]
        require_qualified(sut)
        self.sut_id = sut_id
        self.root = Path(destination)
        self.root.mkdir(parents=True, exist_ok=True)
        self.environment_seed = environment_seed
        self.cursor = 0
        specification = json.loads(json.dumps({"identity": identity,
                         "sut_contract": execution_contract(sut), "environment_seed": environment_seed}))
        path = self.root / "manifest.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != specification:
                raise ValueError("Continuous prefix belongs to another frozen experiment")
        else:
            path.write_text(json.dumps(specification, indent=2), encoding="utf-8")

    def __call__(self, scenario):
        path = self.root / f"query_{self.cursor:04d}.json.gz"
        attempt_path = self.root / f"attempt_{self.cursor:04d}.json"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                saved = json.load(stream)
            if saved["scenario"] != asdict(scenario):
                raise ValueError("A resumed search diverged from its already-paid query prefix")
            execution = PaidExecution(QueryFeedback.from_dict(saved["feedback"]), saved["trajectory"])
        else:
            if attempt_path.exists():
                raise RuntimeError("An earlier physical attempt has no measured feedback; it cannot be silently repeated")
            record = {"query": self.cursor + 1, "scenario": asdict(scenario), "status": "running"}
            attempt_path.write_text(json.dumps(record), encoding="utf-8")
            started = time.perf_counter()
            try:
                execution = execute(create(self.sut_id), scenario, environment_seed=self.environment_seed,
                                     duration_s=DEFAULT.episode_duration_s)
            except Exception as error:
                record.update(status="failed", error=repr(error), wall_s=time.perf_counter() - started)
                attempt_path.write_text(json.dumps(record), encoding="utf-8")
                raise
            call_wall_s = time.perf_counter() - started
            payload = {"scenario": asdict(scenario), "feedback": execution.feedback.to_dict(),
                       "trajectory": execution.trajectory, "events": analyze_events(scenario, execution.trajectory)}
            temporary = path.with_suffix(".tmp")
            with gzip.open(temporary, "wt", encoding="utf-8") as stream:
                json.dump(payload, stream, allow_nan=False)
            temporary.replace(path)
            record.update(status="complete", execution_s=execution.feedback.execution_s, wall_s=call_wall_s)
            attempt_path.write_text(json.dumps(record), encoding="utf-8")
        self.cursor += 1
        return execution
